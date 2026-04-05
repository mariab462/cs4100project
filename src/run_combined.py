# run_combined.py

import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env.project_env import ProjectEnv
from train_lstm import GlucoseLSTM

LSTM_PATH = "models/glucose_lstm.pth"
PPO_PATH = "models/ppo_diabetes.zip"

lstm_model = GlucoseLSTM(input_size=5, hidden_size=64, num_layers=2)
lstm_model.load_state_dict(torch.load(LSTM_PATH))
lstm_model.eval()
print("Loaded LSTM model.")


ppo_model = PPO.load(PPO_PATH)
print("Loaded PPO model.")


env = ProjectEnv()

num_episodes = 5
results = []

all_actual = []
all_pred = []
all_rewards = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False

    total_reward = 0
    step_count = 0
    hypo_events = 0
    hyper_events = 0

    while not done:
        lstm_input = obs[:5]  

        obs_tensor = torch.tensor(lstm_input, dtype=torch.float32) \
                        .unsqueeze(0) \
                        .unsqueeze(1)  

        with torch.no_grad():
            lstm_pred = lstm_model(obs_tensor).item()

       
        ppo_obs = obs.copy()
        ppo_obs[0] = lstm_pred 

       
        action, _ = ppo_model.predict(ppo_obs, deterministic=True)

        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

       
        total_reward += reward
        step_count += 1

        if obs[0] < 70:
            hypo_events += 1
        if obs[0] > 180:
            hyper_events += 1

        all_actual.append(obs[0])
        all_pred.append(lstm_pred)
        all_rewards.append(reward)

        results.append({
            "episode": ep + 1,
            "step": step_count,
            "actual_glucose": obs[0],
            "predicted_glucose": lstm_pred,
            "action": action[0],
            "reward": reward
        })

    print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={step_count}, "
          f"Hypo={hypo_events}, Hyper={hyper_events}")


all_actual = np.array(all_actual)
all_pred = np.array(all_pred)
all_rewards = np.array(all_rewards)

mae = np.mean(np.abs(all_actual - all_pred))
rmse = np.sqrt(np.mean((all_actual - all_pred) ** 2))

# R2
r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - np.mean(all_actual)) ** 2)

time_in_range = np.mean((all_actual >= 70) & (all_actual <= 180)) * 100
hypo = np.sum(all_actual < 70)
hyper = np.sum(all_actual > 180)

print("\n=== FINAL COMBINED MODEL PERFORMANCE ===")
print(f"Time in range (70-180): {time_in_range:.2f}%")
print(f"Hypoglycemia events: {hypo}")
print(f"Hyperglycemia events: {hyper}")
print(f"Average Reward: {np.mean(all_rewards):.2f}")

df = pd.DataFrame(results)
df.to_csv("models/combined_results.csv", index=False)

print(" Saved results to models/combined_results.csv")