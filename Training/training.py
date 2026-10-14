from stable_baselines3 import PPO

from environment import CustomEnv


def main():
    env = CustomEnv()

    # CnnPolicy is for image observations:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=25_000)
    model.save("ppo_galaga")

    model = PPO.load("ppo_galaga")
    observation, _ = env.reset()
    for _ in range(1_000):
        action, _ = model.predict(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
