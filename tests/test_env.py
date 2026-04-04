from src.env.project_env import ProjectEnv

env = ProjectEnv()

obs, _ = env.reset()
print("Initial observation:", obs)

for i in range(10):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i}")
    print("Obs:", obs)
    print("Reward:", reward)

    if terminated or truncated:
        print("Episode ended early")
        break