import tempfile

import gymnasium as gym
import numpy as np

from mame import ACTIONS, FRAMES_PER_STEP, MameBridge
from pathlib import Path
from reward_function import GalagaReward


# Gymnasium custom environment API:
# https://gymnasium.farama.org/introduction/create_custom_env/
class CustomEnv(gym.Env):
    # reward: "offense", "defense", "balanced" or "greedy" (see PRESETS in reward_function.py)
    def __init__(self, reward="balanced"):
        super().__init__()
        self.reward_fn = GalagaReward(reward)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # Placeholder until game observations are implemented.
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.mame = MameBridge()
        self.screenshot_paths = []
        self.screenshots = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mame.start()
        self.reward_fn.reset()
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_state = self.mame.get_game_state()

        self.mame.advance(ACTIONS[action], FRAMES_PER_STEP)
        state = self.mame.get_game_state()

        frame_numbers = range(state["frame"] - 3, state["frame"] + 1)
        self.screenshot_paths = [
            Path(tempfile.gettempdir()) / f"galaga88-frame-{(frame - 1) % 4}.png"
            for frame in frame_numbers
        ]

        self.screenshots = [
            path.read_bytes() for path in self.screenshot_paths
        ]

        # Reward weights live in reward_function.py (old reward = "balanced" preset).
        reward, reward_parts = self.reward_fn(previous_state, state)

        # Observation logic will be added later.
        observation = np.zeros(1, dtype=np.float32)
        return observation, reward, state["game_over"], False, {
            **state,
            "input_readback": self.mame.input_readback,
            "reward_parts": reward_parts,
        }

    def close(self):
        self.mame.close()
