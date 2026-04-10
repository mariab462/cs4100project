# Fixed ProjectEnv.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.env.diabetic_env import Diabetic1Env

class ProjectEnv(gym.Env): 
    metadata = {"render.modes": ["human"]}

    def __init__(self, minute_interval=5):
        super().__init__()  

        #  diabetic environment
        self.env = Diabetic1Env()
        self.env.set_episode_length(minute_interval=minute_interval)

        # Normalize action: RL agent outputs in [-1,1]
        self.action_low = -1.0
        self.action_high = 1.0
        self.env_action_low = self.env.action_space.low[0]   
        self.env_action_high = self.env.action_space.high[0] 

        
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -0.5, 0.5).flatten()[0]
        scaled_action = action * self.env_action_high

        obs, reward, terminated, truncated, info = self.env.step([scaled_action])
        obs = np.nan_to_num(obs, nan=100.0, posinf=400.0, neginf=40.0)
        obs = np.clip(obs, 40, 400)

        reward = np.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0)

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        glucose = self.env.G[-1]
        step = self.env.curr_step
        print(f"Step {step}, Glucose: {glucose:.1f} mg/dL")

    def close(self):
        self.env.close()