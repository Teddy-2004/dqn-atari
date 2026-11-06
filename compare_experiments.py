"""Plot all of one member's experiments on one chart, to compare them.

Example:
    uv run compare_experiments.py --member Teddy

Saves models/<member>/comparison_plot.png
"""

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window=20):
    """Average of the last `window` values at each point (shorter at the start)."""
    return [np.mean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=str, required=True)
    args = parser.parse_args()

    member_dir = os.path.join("models", args.member)
    exp_dirs = sorted(glob.glob(os.path.join(member_dir, "exp*")),
                      key=lambda d: int(os.path.basename(d)[3:]))  # exp1, exp2, ..., exp10

    fig, ax = plt.subplots(figsize=(11, 6))
    for exp_dir in exp_dirs:
        log_path = os.path.join(exp_dir, "training_log.csv")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        steps = [int(row["total_steps"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]

        # Label each line with the experiment name and its hyperparameters
        label = os.path.basename(exp_dir)
        summary_path = os.path.join(exp_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                h = json.load(f)["hyperparameters"]
            label += (f": lr={h['lr']:g}, γ={h['gamma']}, batch={h['batch_size']}, "
                      f"ε_end={h['eps_end']}, decay={h['eps_decay']:,}")
        ax.plot(steps, moving_average(rewards), label=label)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")  # 0 = even with the opponent
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Reward (20-episode moving average)")
    ax.set_title(f"{args.member}: all experiments compared")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out_path = os.path.join(member_dir, "comparison_plot.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
