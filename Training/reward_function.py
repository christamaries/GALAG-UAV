"""
GALAG-UAV's Reward function with presets based on intended strategy.
1) offense    - points gained matter mosts
2) balanced   - middle ground 
3) defense    - staying alive matters most
4) greedy     - only points matter 

Note: Stage cleared, captures, rescues and missed depend on RAM addreses
"""

from dataclasses import dataclass

FPS = 60  # the game runs at 60 frames per second

@dataclass
class Weights:
    score_divisor: float       # points are divided by this (the lower the divisor = more points)
    score_clip: float          # max limit of score reward in one step (prevents from big risks and life loss)
    survival_per_frame: float  # small bonus for every frame alive
    life_lost: float           # penalty for each life lost (negative)
    game_over: float           # one-time penalty when the game ends (negative)
    idle_grace_s: float        # seconds without scoring before the idle penalty starts
    idle_per_s: float          # idle penalty per second after the grace period

PRESETS = {
    # Preset 1: Offense
    # Agent prioritizes killing enemies. It takes more risks since penalty for death is less.
    # Strong idle penalty, 
    "offense": Weights(score_divisor=300, score_clip=2.0, survival_per_frame=0.0,
                      life_lost=-1.0, game_over=-1.0,
                      idle_grace_s=3, idle_per_s=0.02),

    # Preset 2: Defense
    # Agent prioritizes stay alive. Points are worth less and dying costs a lot.
    # Weak idle penalty,to prevent dodging forever without shooting
    "defense": Weights(score_divisor=1000, score_clip=1.0, survival_per_frame=0.002,
                          life_lost=-5.0, game_over=-5.0,
                          idle_grace_s=20, idle_per_s=0.005),
    
    # Preset 3: Balanced
    # Equal consideration for offense and defense
    "balanced": Weights(score_divisor=500, score_clip=1.0, survival_per_frame=0.001,
                        life_lost=-2.0, game_over=-3.0,
                        idle_grace_s=10, idle_per_s=0.01),

    # Preset 4: Greedy (control group)
    # Only points are rewarded. 
    # The agent has to learn on its own that dying is bad (it can't score after game over).
    "greedy": Weights(score_divisor=500, score_clip=1.0, survival_per_frame=0.0,
                      life_lost=0.0, game_over=0.0,
                      idle_grace_s=0, idle_per_s=0.0),
}

# Reward function
class GalagaReward:
    def __init__(self, preset="balanced"):
        if preset not in PRESETS:
            raise ValueError(f"Unknown reward preset {preset!r}, pick one of {list(PRESETS)}")
        self.preset = preset
        self.w = PRESETS[preset]
        self.reset()

    # 
    def reset(self):
        # Frame when the score last went up (used by the idle penalty)
        self.last_score_frame = None

    # Checks previous and current game frames to see if score changes
    def __call__(self, prev, curr):
        w = self.w
        frames = curr["frame"] - prev["frame"]
        if self.last_score_frame is None:
            self.last_score_frame = prev["frame"]
        variables = {}

        # Points scored this step (capped)
        points = max(curr["score"] - prev["score"], 0)
        variables["score"] = min(points / w.score_divisor, w.score_clip)

        # Bonus for staying alive
        variables["survival"] = w.survival_per_frame * frames

        # Penalty for each life lost
        lives_lost = max(prev["reserve_lives"] - curr["reserve_lives"], 0)
        variables["life_lost"] = w.life_lost * lives_lost

        # Penalty for going too long without scoring
        if points > 0:
            self.last_score_frame = curr["frame"]
        idle_s = (curr["frame"] - self.last_score_frame) / FPS
        variables["idle"] = -w.idle_per_s * frames / FPS if idle_s > w.idle_grace_s else 0.0

        # One-time penalty when the game ends
        variables["game_over"] = w.game_over if curr["game_over"] and not prev["game_over"] else 0.0

        return sum(variables.values()), variables


# Testing game events using reward function
if __name__ == "__main__":
    base = {"frame": 1000, "score": 0, "reserve_lives": 2, "game_over": False}

    # Creates fake game events
    def after(**changes):
        # The next step: 4 frames later, plus whatever changed.
        return {**base, "frame": base["frame"] + 4, **changes}

    events = {
        "kill a bee (50 pts)": (base, after(score=50)),
        "boss combo (1600 pts)": (base, after(score=1600)),
        "nothing happens": (base, after()),
        "idle 11 s": (base, after()),
        "lose a life": (base, after(reserve_lives=1)),
        "game over": ({**base, "reserve_lives": 0}, after(reserve_lives=0, game_over=True)),
    }

    # Checks the rewards against these events
    def reward_for(preset, name):
        R = GalagaReward(preset)
        if name == "idle 11 s":
            R.last_score_frame = base["frame"] - 11 * FPS
        return R(*events[name])[0]

    # Exact checks on the preset.
    expected = {
        "kill a bee (50 pts)": 0.1 + 0.004,
        "boss combo (1600 pts)": 1.0 + 0.004,    # clipped at 1.0
        "nothing happens": 0.004,
        "idle 11 s": 0.004 - 0.01 * 4 / FPS,
        "lose a life": -2.0 + 0.004,
        "game over": -3.0 + 0.004,
    }
    for name, value in expected.items():
        got = reward_for("balanced", name)
        assert abs(got - value) < 1e-9, f"balanced / {name}: got {got}, expected {value}"

    # Table of all presets to check functionality
    print(f"{'event':<24}" + "".join(f"{p:>10}" for p in PRESETS))
    for name in events:
        print(f"{name:<24}" + "".join(f"{reward_for(p, name):>+10.3f}" for p in PRESETS))
    print("All checks passed.")
