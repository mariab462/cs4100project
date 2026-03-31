# evaluate_rl.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.project_env import ProjectEnv  # your environment

# Load environment
env = DummyVecEnv([lambda: ProjectEnv()])

# Load trained model
model = PPO.load("ppo_diabetic")

num_episodes = 5
target_range = (70, 180)

all_rewards = []
all_hypo = 0
all_hyper = 0
all_in_target = 0
total_steps = 0

for ep in range(num_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    ep_hypo = 0
    ep_hyper = 0
    ep_target = 0
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  

        obs_val = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs

        glucose = obs_val[0]  
        if glucose < target_range[0]:
            ep_hypo += 1
        elif glucose > target_range[1]:
            ep_hyper += 1
        else:
            ep_target += 1

    
        ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        step_count += 1

    all_rewards.append(ep_reward)
    all_hypo += ep_hypo
    all_hyper += ep_hyper
    all_in_target += ep_target
    total_steps += step_count

    print(f"Episode {ep+1}: Total Reward = {ep_reward:.2f}, Steps = {step_count}")

# Overall safety metrics
print("\nOverall Safety Metrics:")
print(f"Time in target range ({target_range[0]}-{target_range[1]}): {all_in_target/total_steps*100:.2f}%")
print(f"Hypoglycemia events (<{target_range[0]}): {all_hypo}")
print(f"Hyperglycemia events (>{target_range[1]}): {all_hyper}")
print(f"Average Reward per Episode: {np.mean(all_rewards):.2f}")