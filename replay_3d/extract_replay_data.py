"""Record the data behind the 3D replay (boxing_replay_3d.html).

Plays one greedy round with the final model (dqn_model.pth), the same round as
the gameplay video (seed 1004), and saves for every step:
    agent x, agent y, CPU x, CPU y, action, reward, agent score, CPU score, clock
Positions and scores are read straight from the Atari console's memory (RAM).

Run from the repository root:
    uv run python replay_3d/extract_replay_data.py
"""

import json
import os
import sys

import numpy as np
import torch

# Make common.py (in the repository root) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import QNetwork, make_env  # noqa: E402

# Boxing RAM addresses (from the AtariARI annotations)
CLOCK, PLAYER_SCORE, ENEMY_SCORE = 17, 18, 19
PLAYER_X, ENEMY_X, PLAYER_Y, ENEMY_Y = 32, 33, 34, 35


def main():
    env = make_env()
    q_network = QNetwork(env.action_space.n)
    q_network.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
    q_network.eval()
    ram = env.unwrapped.ale

    state, _ = env.reset(seed=1004)
    frames, total, done = [], 0.0, False
    while not done:
        with torch.no_grad():  # greedy action, exactly like play.py
            action = int(q_network(torch.from_numpy(np.asarray(state)).unsqueeze(0)).argmax(dim=1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
        r = ram.getRAM()
        frames.append([int(r[PLAYER_X]), int(r[PLAYER_Y]), int(r[ENEMY_X]), int(r[ENEMY_Y]),
                       action, int(reward), int(r[PLAYER_SCORE]), int(r[ENEMY_SCORE]), int(r[CLOCK])])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_data.json")
    with open(out, "w") as f:
        json.dump({"actions": env.unwrapped.get_action_meanings(), "frames": frames}, f, separators=(",", ":"))
    print(f"Score {total:+.0f} over {len(frames)} steps. Saved {out}")


if __name__ == "__main__":
    main()
