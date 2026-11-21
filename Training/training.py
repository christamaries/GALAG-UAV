from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule

from environment import CustomEnv


REWARD = "balanced"
ROOT = Path(__file__).resolve().parent.parent

SOURCE_MODEL = ROOT / "ppo_galaga_balanced_stabilized_v3.zip"
FINAL_MODEL = ROOT / "ppo_galaga_balanced_stabilized_v4.zip"
CHECKPOINT_DIR = ROOT / "checkpoints"

ADDITIONAL_STEPS = 500_000


def main():
    env = CustomEnv(reward=REWARD)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_galaga_stabilized",
    )

    model = PPO.load(
        str(SOURCE_MODEL),
        env=env,
        verbose=1,
        device="cpu",
    )

    model.learning_rate = 3e-5
    model.lr_schedule = FloatSchedule(3e-5)
    model.clip_range = FloatSchedule(0.1)
    model.n_epochs = 3
    model.target_kl = 0.03

    print("Continuing from step:", model.num_timesteps)

    try:
        model.learn(
            total_timesteps=ADDITIONAL_STEPS,
            reset_num_timesteps=False,
            callback=checkpoint_callback,
        )
        model.save(str(FINAL_MODEL))
    finally:
        env.close()


if __name__ == "__main__":
    main()