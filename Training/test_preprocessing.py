import cv2
import numpy as np

from environment import CustomEnv

STEPS = 1000

env = CustomEnv()
observation, _ = env.reset()
print("Reset observation:", observation.shape, observation.dtype)

total_reward = 0.0
for step in range(STEPS):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Last observation:", observation.shape, observation.dtype,
      "min", observation.min(), "max", observation.max())
print("Score:", info["score"], " Total reward:", round(total_reward, 3))

frames = [cv2.resize(frame, (252, 252), interpolation=cv2.INTER_NEAREST) for frame in observation]
cv2.imwrite("observation_frames.png", np.hstack(frames))
print("Saved observation_frames.png")
env.close()