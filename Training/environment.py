import gymnasium as gym
import numpy as np

from mame import ACTIONS, FRAMES_PER_STEP, MameBridge
from pathlib import Path


# Gymnasium custom environment API:
# https://gymnasium.farama.org/introduction/create_custom_env/
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # Placeholder until game observations are implemented.
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.mame = MameBridge()
        self.screenshot_paths = []
        self.screenshots = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mame.start()
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_state = self.mame.get_game_state()

        self.mame.advance(ACTIONS[action], FRAMES_PER_STEP)
        state = self.mame.get_game_state()

        frame_numbers = range(state["frame"] - 3, state["frame"] + 1)
        self.screenshot_paths = [
            Path(f"/tmp/galaga88-frame-{(frame - 1) % 4}.png")
            for frame in frame_numbers
        ]

        self.screenshots = [
            path.read_bytes() for path in self.screenshot_paths
        ]

        score_increase = max(0, state["score"] - previous_state["score"])
        lives_lost = max(
            0,
            previous_state["reserve_lives"] - state["reserve_lives"],
        )
        
        frames_elapsed = state["frame"] - previous_state["frame"]
        terminated = state["game_over"]

        if terminated:
            reward -= 3

        reward = score_increase / 500
        reward += frames_elapsed * 0.001
        reward -= lives_lost * 2

        # Observation, reward, and episode-ending logic will be added later.
        observation = np.zeros(1, dtype=np.float32)
        return observation, reward, state["game_over"], False, {
            **state,
            "input_readback": self.mame.input_readback,
        }

    def close(self):
        self.mame.close()
