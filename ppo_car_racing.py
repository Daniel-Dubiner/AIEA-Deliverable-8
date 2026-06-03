import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_NAME = "CarRacing-v3"
EPISODES = 200
MAX_STEPS = 1000
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 6
BATCH_SIZE = 64
LR = 2e-4
ENT_COEF = 0.02
VF_COEF = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Preprocessing ---
def preprocess(obs):
    gray = np.mean(obs, axis=2) / 255.0
    return gray.astype(np.float32)  # shape: (96, 96)

# --- CNN Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, 1, 96, 96)
        cnn_out = self.cnn(dummy).shape[1]
        self.shared = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.shared(x)
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value

    def get_action(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(), entropy

# --- Compute GAE ---
def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# --- PPO Update ---
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)  # (B, 1, 96, 96)
    actions = torch.FloatTensor(np.array(actions)).to(device)
    old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
    returns = torch.FloatTensor(np.array(returns)).to(device)
    advantages = torch.FloatTensor(np.array(advantages)).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(EPOCHS):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), BATCH_SIZE):
            idx = indices[start:start+BATCH_SIZE]
            log_probs, values, entropy = model.evaluate(states[idx], actions[idx])
            ratio = (log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns[idx])
            entropy_loss = -entropy.mean()
            loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

# --- Main ---
def main():
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()
    action_dim = env.action_space.shape[0]
    print(f"Action dim: {action_dim}")

    model = ActorCritic(action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 96, 96)
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            next_state = preprocess(next_obs)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.cpu().item())
            values.append(value.cpu().item())

            state = next_state
            total_reward += reward

            if done:
                break

        with torch.no_grad():
            _, _, next_value = model.get_action(torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device))
        next_value = next_value.cpu().item()

        advantages, returns = compute_gae(rewards, values, dones, next_value)
        ppo_update(model, optimizer, states, actions, log_probs, returns, advantages)

        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f}")

    env.close()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.plot(np.convolve(episode_rewards, np.ones(5)/5, mode='valid'), label="5-ep Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO (CNN) from Scratch on CarRacing-v3")
    plt.legend()
    plt.savefig("/home/ubuntu/persistent/ppo_cnn_results.png")
    plt.close()
    print("Plot saved to ppo_cnn_results.png")

if __name__ == "__main__":
    main()
