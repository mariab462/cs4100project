from src.env.diabetic_env import Diabetic1Env

class ProjectEnv:
    def __init__(self):
        # create the underlying environment
        self.env = Diabetic1Env()

        self.env.set_episode_length(minute_interval=10)

        # expose these so RL can access them
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)