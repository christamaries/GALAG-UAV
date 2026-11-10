import tempfile

import gymnasium as gym
import numpy as np

from mame import ACTIONS, FRAMES_PER_STEP, MameBridge
from pathlib import Path
from reward_function import GalagaReward
from preprocessing import FRAME_SIZE, STACK_SIZE, build_observation


# Gymnasium custom environment API:
# https://gymnasium.farama.org/introduction/create_custom_env/
class CustomEnv(gym.Env):
    # reward: "offense", "defense", "balanced" or "greedy" (see PRESETS in reward_function.py)
    def __init__(self, reward="balanced"):
        super().__init__()
        self.reward_fn = GalagaReward(reward)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # 4 stacked grayscale frames, 84x84, pixel values 0-255.
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(STACK_SIZE, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
        )
        self.mame = MameBridge()
        self.screenshot_paths = []
        self.screenshots = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mame.start()
        self.reward_fn.reset()
        observation = build_observation(self._read_screenshots())
        return observation, {}

    def _read_screenshots(self):
        # The last 4 frames MAME saved (oldest first).
        frame = self.mame.get_game_state()["frame"]
        frame_numbers = range(frame - 3, frame + 1)
        self.screenshot_paths = [
            Path(tempfile.gettempdir()) / f"galaga88-frame-{(number - 1) % 4}.png"
            for number in frame_numbers
        ]
        return [path.read_bytes() for path in self.screenshot_paths]

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_state = self.mame.get_game_state()

        self.mame.advance(ACTIONS[action], FRAMES_PER_STEP)
        state = self.mame.get_game_state()

        self.screenshots = self._read_screenshots()

        # Reward weights live in reward_function.py (old reward = "balanced" preset).
        reward, reward_parts = self.reward_fn(previous_state, state)

        observation = build_observation(self.screenshots)
        return observation, reward, state["game_over"], False, {
            **state,
            "input_readback": self.mame.input_readback,
            "reward_parts": reward_parts,
        }

    def close(self):
        self.mame.close()