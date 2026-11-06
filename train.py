"""Train a DQN agent on an Atari game.

Example (one experiment):
    uv run train.py --member Teddy --exp 1 --lr 1e-4 --gamma 0.99 --batch_size 32 \
        --eps_start 1.0 --eps_end 0.01 --eps_decay 100000

Everything for that run is saved in models/<member>/exp<N>/:
    dqn_model.pth       trained network weights
    training_log.csv    one row per episode (episode length, total steps, reward, epsilon)
    training_plot.png   reward and episode-length curves
    summary.json        hyperparameters used + final results (for the experiment table)
"""

import argparse
import csv
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import DEFAULT_ENV_ID, QNetwork, get_device, make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on an Atari game")
    # Who / which experiment (decides the output folder)
    parser.add_argument("--member", type=str, default="test", help="group member name")
    parser.add_argument("--exp", type=int, default=0, help="experiment number (1-10)")
    parser.add_argument("--env", type=str, default=DEFAULT_ENV_ID)
    # The hyperparameters we tune (the columns of the experiment table)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--eps_decay", type=int, default=100_000,
                        help="number of steps over which epsilon goes from eps_start to eps_end")
    # Other training settings (kept fixed across experiments)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=50_000,
                        help="replay buffer capacity (about 56 MB of RAM per 1,000 transitions)")
    parser.add_argument("--learning_starts", type=int, default=10_000,
                        help="random steps collected before training begins")
    parser.add_argument("--train_freq", type=int, default=4, help="do a gradient update every N steps")
    parser.add_argument("--target_update", type=int, default=1_000,
                        help="copy online network into target network every N steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ReplayBuffer:
    """Fixed-size memory of past transitions (state, action, reward, next_state, done).

    Sampling random mini-batches from it breaks the correlation between
    consecutive frames, which makes training much more stable.
    When the buffer is full, the oldest transitions are overwritten.
    """

    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0  # where the next transition will be written
        self.size = 0   # how many transitions are stored

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity  # wrap around (circular buffer)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Linear decay: eps_start at step 0, eps_end after eps_decay steps, then constant."""
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(state, epsilon, q_network, num_actions, device):
    """Epsilon-greedy: random action with probability epsilon, otherwise the best action."""
    if random.random() < epsilon:
        return random.randrange(num_actions)  # explore
    with torch.no_grad():
        state_tensor = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)  # add batch dim
        q_values = q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())  # exploit


def train_step(q_network, target_network, optimizer, buffer, batch_size, gamma, device):
    """One gradient update of the online network on a random mini-batch."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)

    # Q(s, a) predicted by the online network, for the actions that were actually taken
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + gamma * max_a' Q_target(s', a')
    # (1 - done) removes the future term when the episode ended at s'
    with torch.no_grad():
        max_next_q = target_network(next_states).max(dim=1).values
        targets = rewards + gamma * max_next_q * (1 - dones)

    # Huber loss: like MSE for small errors, like L1 for large ones (more stable)
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()  # backpropagation
    optimizer.step()  # update the online network's weights
    return loss.item()


def save_plot(episode_rewards, episode_lengths, path, title):
    """Plot reward and episode length per episode, with a 20-episode moving average."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    window = 20

    ax1.plot(episode_rewards, alpha=0.4, label="Episode reward")
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"{window}-episode average")
    ax1.set_ylabel("Reward")
    ax1.legend()

    ax2.plot(episode_lengths, color="tab:orange")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode length (steps)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()

    # Output folder for this experiment
    out_dir = os.path.join("models", args.member, f"exp{args.exp}")
    os.makedirs(out_dir, exist_ok=True)

    # Seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device} | Output: {out_dir}")

    # 1. Environment
    env = make_env(args.env)
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape  # (4, 84, 84)

    # 2. Online and target networks (target starts as an exact copy of the online network)
    q_network = QNetwork(num_actions).to(device)
    target_network = QNetwork(num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()  # the target network is never trained directly

    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr)

    # 3. Replay buffer
    buffer = ReplayBuffer(args.buffer_size, state_shape)

    # Logging
    log_path = os.path.join(out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "episode_length", "total_steps", "reward", "epsilon"])
    episode_rewards, episode_lengths = [], []

    state, _ = env.reset(seed=args.seed)
    episode_reward, episode_length = 0.0, 0
    start_time = time.time()

    # 4. Main training loop: one iteration = one environment step
    for step in range(1, args.total_steps + 1):
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(state, epsilon, q_network, num_actions, device)

        next_state, reward, terminated, truncated, _ = env.step(action)
        # Reward clipping (as in the DQN paper): we store only the sign of the
        # reward (+1, 0, -1) so large rewards can't cause huge, unstable updates.
        # In Boxing every reward is already +1 or -1, so this changes nothing here,
        # but it keeps the code correct for any Atari game. The real score is logged.
        clipped_reward = np.sign(reward)
        # Only "terminated" (game over) stops the Bellman bootstrap.
        # "truncated" (time limit) is not a real end state, so we still look ahead.
        buffer.add(state, action, clipped_reward, next_state, terminated)

        state = next_state
        episode_reward += reward
        episode_length += 1

        # Learn from replayed experience (after the buffer has some data)
        if step > args.learning_starts and step % args.train_freq == 0:
            train_step(q_network, target_network, optimizer, buffer,
                       args.batch_size, args.gamma, device)

        # Periodically sync the target network with the online network
        if step % args.target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        # End of an episode: log it and start a new one
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            log_writer.writerow([len(episode_rewards), episode_length, step, episode_reward, round(epsilon, 4)])
            log_file.flush()

            avg_reward = np.mean(episode_rewards[-20:])
            minutes = (time.time() - start_time) / 60
            print(f"Episode {len(episode_rewards):4d} | step {step:8d} | reward {episode_reward:6.1f} | "
                  f"avg(20) {avg_reward:6.2f} | length {episode_length:5d} | eps {epsilon:.3f} | {minutes:.1f} min")

            state, _ = env.reset()
            episode_reward, episode_length = 0.0, 0

        # Save a checkpoint every 100k steps so a crash doesn't lose everything
        if step % 100_000 == 0:
            torch.save(q_network.state_dict(), os.path.join(out_dir, "dqn_model.pth"))

    env.close()
    log_file.close()

    # 5. Save final model, plot and summary
    torch.save(q_network.state_dict(), os.path.join(out_dir, "dqn_model.pth"))
    save_plot(episode_rewards, episode_lengths, os.path.join(out_dir, "training_plot.png"),
              title=f"{args.env} | {args.member} exp {args.exp}")

    summary = {
        "hyperparameters": vars(args),
        "episodes": len(episode_rewards),
        "mean_reward_last_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else None,
        "best_episode_reward": float(max(episode_rewards)) if episode_rewards else None,
        "mean_episode_length_last_20": float(np.mean(episode_lengths[-20:])) if episode_lengths else None,
        "training_minutes": round((time.time() - start_time) / 60, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Mean reward (last 20 episodes): {summary['mean_reward_last_20']}")
    print(f"Saved model, log, plot and summary to {out_dir}")


if __name__ == "__main__":
    main()
