# train_ppo_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.env.project_env import ProjectEnv
from src.train_lstm import GlucoseLSTM
from collections import deque
import joblib


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


# Environment Setup
env = ProjectEnv()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = PolicyNetwork(obs_dim, action_dim)
value_net = ValueNetwork(obs_dim)

policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

scaler = joblib.load("models/scaler.save")
seq_buffer = deque(maxlen=6)

gamma = 0.99
num_episodes = 500  # longer training for better learning

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False

    states, actions, rewards, log_probs = [], [], [], []
    total_reward = 0
    all_actual_glucose = []

    while not done:
        seq_buffer.append(obs[:5])

        if len(seq_buffer) < 6:
            lstm_pred = obs[0]  # fallback until buffer fills
        else:
            seq_array = np.array(seq_buffer, dtype=np.float32)
            seq_scaled = scaler.transform(seq_array)
            lstm_tensor = torch.tensor(seq_scaled).unsqueeze(0)

            with torch.no_grad():
                lstm_pred = lstm_model(lstm_tensor).item()

        # Replace actual glucose with LSTM prediction
        obs_aug = np.copy(obs)
        obs_aug[0] = lstm_pred

        #  Normalize observation
        obs_norm = np.array(obs_aug, dtype=np.float32) / 500.0  # scale numeric inputs
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)

        #  PPO action
        action, log_prob = policy.get_action(obs_tensor)
        next_obs, reward, terminated, truncated, _ = env.step(action.detach().numpy())
        done = terminated or truncated

        # Time-in-Range reward shaping
        glucose = next_obs[0]
        env_reward = reward

        if 70 <= glucose <= 180:
            reward = env_reward + 1.0
        else:
            reward = env_reward - 1.0

        states.append(obs_tensor)
        actions.append(action)
        rewards.append(torch.tensor([reward], dtype=torch.float32))
        log_probs.append(log_prob)

        total_reward += reward
        all_actual_glucose.append(glucose)
        obs = next_obs

    #  Compute returns and advantages
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    log_probs = torch.stack(log_probs)

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.stack(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    returns = returns.view(-1, 1)

    values = value_net(states)
    advantage = returns - values

    new_log_probs, entropy = policy.evaluate_actions(states, actions)
    policy_loss = -(advantage.detach() * new_log_probs).mean()
    value_loss = nn.MSELoss()(values, returns)

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