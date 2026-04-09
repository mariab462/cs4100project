# run_combined_lstm_ppo.py
import torch
import numpy as np
import pandas as pd
from env.project_env import ProjectEnv
from train_lstm import GlucoseLSTM
from train_ppo import PolicyNetwork, ValueNetwork  

# Load models 
LSTM_PATH = "models/glucose_lstm.pth"
PPO_POLICY_PATH = "models/ppo_policy_lstm.pth"
PPO_VALUE_PATH = "models/ppo_value_lstm.pth"

lstm_model = GlucoseLSTM(input_size=5, hidden_size=64, num_layers=2)
lstm_model.load_state_dict(torch.load(LSTM_PATH))
lstm_model.eval()

# Setup environment
env = ProjectEnv()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = PolicyNetwork(obs_dim, action_dim)
value_net = ValueNetwork(obs_dim)
policy.load_state_dict(torch.load(PPO_POLICY_PATH))
value_net.load_state_dict(torch.load(PPO_VALUE_PATH))
policy.eval()
value_net.eval()
print("Loaded LSTM and PPO models.")

num_episodes = 5
results = []
all_actual, all_pred, all_rewards = [], [], []

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    hypo_events = 0
    hyper_events = 0

    while not done:
        #  LSTM prediction 
        lstm_input = np.array(obs[:5], dtype=np.float32)
        lstm_tensor = torch.tensor(lstm_input).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            lstm_pred = lstm_model(lstm_tensor).item()

        # Replace glucose
        obs[0] = lstm_pred
        obs_tensor = torch.tensor(np.array(obs, dtype=np.float32) / 500.0)

        #  PPO action 
        action, _ = policy.get_action(obs_tensor)
        next_obs, reward, terminated, truncated, _ = env.step(action.detach().numpy())
        done = terminated or truncated

        # Reward 
        glucose = next_obs[0]
        if glucose < 70:
            hypo_events += 1
        elif glucose > 180:
            hyper_events += 1
        if 70 <= glucose <= 180:
            reward = 1.0
        else:
            reward = -1.0

        total_reward += reward
        step_count += 1

        all_actual.append(glucose)
        all_pred.append(lstm_pred)
        all_rewards.append(reward)

        results.append({
            "episode": ep+1,
            "step": step_count,
            "actual_glucose": glucose,
            "predicted_glucose": lstm_pred,
            "action": action[0].item(),
            "reward": reward
        })
        obs = next_obs

    print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={step_count}, "
          f"Hypo={hypo_events}, Hyper={hyper_events}")


all_actual = np.array(all_actual)
all_pred = np.array(all_pred)
all_rewards = np.array(all_rewards)

mae = np.mean(np.abs(all_actual - all_pred))
rmse = np.sqrt(np.mean((all_actual - all_pred) ** 2))
r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - np.mean(all_actual)) ** 2)
time_in_range = np.mean((all_actual >= 70) & (all_actual <= 180)) * 100
hypo = np.sum(all_actual < 70)
hyper = np.sum(all_actual > 180)

print("\n=== FINAL COMBINED MODEL PERFORMANCE ===")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
print(f"Time in range (70-180): {time_in_range:.2f}%")
print(f"Hypoglycemia events: {hypo}")
print(f"Hyperglycemia events: {hyper}")
print(f"Average Reward: {np.mean(all_rewards):.2f}")

df = pd.DataFrame(results)
df.to_csv("models/combined_results_lstm_ppo.csv", index=False)
print("Saved results to models/combined_results_lstm_ppo.csv")