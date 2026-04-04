import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.diabetic_env import Diabetic1Env
from src.lstm_predictor import LSTMPredictor


class HybridEnv(gym.Env):
    def __init__(self, minute_interval=5):
        super().__init__()

        self.env = Diabetic1Env()
        self.env.set_episode_length(minute_interval=minute_interval)

        self.predictor = LSTMPredictor(
            model_path="../models/glucose_lstm.pth",
            data_csv="../data/processed/patient_data.csv"
        )

        self.action_space = self.env.action_space

        base_obs_low = self.env.observation_space.low
        base_obs_high = self.env.observation_space.high

        low = np.append(base_obs_low, 0.0)
        high = np.append(base_obs_high, 1.0)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset()

        self.predictor = LSTMPredictor(
            model_path="../models/glucose_lstm.pth",
            data_csv="../data/processed/patient_data.csv"
        )

        return self._augment_obs(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        glucose = obs[0]
        basal = 0.0
        bolus = 0.0
        meal = obs[6]
        exercise = 0.0

        self.predictor.update(glucose, basal, bolus, meal, exercise)
        prediction = self.predictor.predict()

        obs = self._augment_obs(obs, prediction)

        return obs, reward, terminated, truncated, info

    def _augment_obs(self, obs, prediction=None):
        if prediction is None:
            prediction = self.predictor.predict()

        return np.append(obs, prediction)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()