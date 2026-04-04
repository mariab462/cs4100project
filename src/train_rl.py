# train_rl.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.project_env import ProjectEnv
# Create environment
env = DummyVecEnv([lambda: ProjectEnv()]) 

# Initialize PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.99,
    clip_range=0.2,
)

# Train agent
timesteps = 100_000  
model.learn(total_timesteps=timesteps)


model.save("ppo_diabetic")

print("Training finished and model saved as 'ppo_diabetic'.")