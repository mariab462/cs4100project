# train_ppo_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.env.project_env import ProjectEnv
from src.train_lstm import GlucoseLSTM
from collections import deque


# === Load pre-trained LSTM ===
LSTM_PATH = "models/glucose_lstm.pth"
lstm_model = GlucoseLSTM(input_size=5, hidden_size=64, num_layers=2)
lstm_model.load_state_dict(torch.load(LSTM_PATH))
lstm_model.eval()
print("Loaded LSTM model.")


# PPO Policy and Value Networks
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, states, actions):
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns



# Environment Setup
env = ProjectEnv()
obs_dim = env.observation_space.shape[0] + 1
action_dim = env.action_space.shape[0]

policy = PolicyNetwork(obs_dim, action_dim)
value_net = ValueNetwork(obs_dim)

policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

seq_buffer = deque(maxlen=6)

gamma = 0.99
lam = 0.95
epsilon = 0.2
entropy_coef = 0.01
K_epochs = 4
mini_batch_size = 64
num_episodes = 500  # longer training for better learning

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False

    states, actions, rewards, dones = [], [], [], []

    old_log_probs = []
    raw_values = []

    total_reward = 0
    all_actual_glucose = []

    while not done:
        seq_buffer.append(obs[:5])

        if len(seq_buffer) < 6:
            lstm_value = obs[0]  # fallback until buffer fills
        else:
            seq_array = np.array(seq_buffer, dtype=np.float32)
            seq_tensor = torch.tensor(seq_array).unsqueeze(0)

            with torch.no_grad():
                lstm_value = lstm_model(seq_tensor).item()

        lstm_feat = np.array([lstm_value / 500.0], dtype=np.float32)

        obs_np = np.array(obs, dtype=np.float32) / 500.0
        obs_full = np.concatenate([obs_np, lstm_feat])
        obs_tensor = torch.tensor(obs_full, dtype=torch.float32)

        with torch.no_grad():
            action, log_prob = policy.get_action(obs_tensor)
            value = value_net(obs_tensor).item()

        #  PPO action
        action_np = action.detach().numpy()

        next_obs, env_reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        # Time-in-Range reward shaping
        glucose = next_obs[0]
        if glucose < 70:
            reward = env_reward - 2.0
        elif glucose > 180:
            reward = env_reward - min((glucose - 180) / 50.0, 5.0)
        else:
            reward = env_reward + 1.0

        states.append(obs_tensor)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        old_log_probs.append(log_prob.detach())
        raw_values.append(value)

        total_reward += reward
        all_actual_glucose.append(glucose)
        obs = next_obs

    #  Compute returns and advantages
    advantages, returns = compute_gae(rewards, raw_values, dones, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    returns = returns.view(-1, 1)

    dataset_size = len(states)
    for _ in range(K_epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]

            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_old_lp = old_log_probs[idx]
            mb_returns = returns[idx]
            mb_advantages = advantages[idx]

            new_log_probs, entropy = policy.evaluate_actions(mb_states, mb_actions)
            values = value_net(mb_states)

            ratio = torch.exp(new_log_probs - mb_old_lp)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            policy_loss = policy_loss - entropy_coef * entropy.mean()

            value_loss = nn.MSELoss()(values, mb_returns)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()


    time_in_range = ((np.array(all_actual_glucose) >= 70) & (np.array(all_actual_glucose) <= 180)).mean() * 100
    print(f"Episode {ep + 1}: Total Reward={total_reward:.2f}, Time-in-range={time_in_range:.2f}%")

# Save Models 
torch.save(policy.state_dict(), "models/ppo_policy_lstm.pth")
torch.save(value_net.state_dict(), "models/ppo_value_lstm.pth")
print("Saved PPO policy and value networks.")