"""
Random-action baseline for comparing against the trained agent.
Plays full games through the same CustomEnv the PPO agent trains on (same 6 actions,
same 4-frame action hold, same reward and game-over rules), choosing every action
uniformly at random.

"""
import statistics

from environment import CustomEnv

EPISODES = 10  # change as needed
REWARD = "balanced"  # ["offense", "defense", "balanced", "greedy"]


def main():
    env = CustomEnv(reward=REWARD)
    print(f"reward preset: {REWARD}")
    rewards, scores, frames = [], [], []
    for ep in range(EPISODES):
        env.reset()  # launches MAME and plays coin/start, like the training loop
        total, done = 0.0, False
        while not done:
            # The "policy": sample any action with equal probability.
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        rewards.append(total)
        scores.append(info["score"])     # in-game points (easier to explain than reward)
        frames.append(info["frame"])     # emulator frame at game over ~ survival time
        print(f"episode {ep}: reward={total:.2f} score={info['score']} frames={info['frame']}")
    env.close()

    # Summary to put next to the trained agent's numbers (evaluate it with the same EPISODES).
    for name, vals in (("reward", rewards), ("score", scores), ("frames", frames)):
        print(f"{name}: mean={statistics.mean(vals):.1f} stdev={statistics.pstdev(vals):.1f}")


if __name__ == "__main__":
    main()
