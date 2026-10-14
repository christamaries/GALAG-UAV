import gymnasium as gym
import numpy as np

from mame import ACTIONS, FRAMES_PER_STEP, MameBridge


# Gymnasium custom environment API:
# https://gymnasium.farama.org/introduction/create_custom_env/
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # Placeholder until game observations are implemented.
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.mame = MameBridge()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mame.start()
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.mame.advance(ACTIONS[action], FRAMES_PER_STEP)

        # Observation, reward, and episode-ending logic will be added later.
        observation = np.zeros(1, dtype=np.float32)
        return observation, 0.0, False, False, {
            "frame": self.mame.frame,
            "input_readback": self.mame.input_readback,
        }

    def close(self):
        self.mame.close()
