"""Load a trained DQN model and watch it play.

Example:
    uv run play.py                                         # uses dqn_model.pth
    uv run play.py --model models/Teddy/exp3/dqn_model.pth --episodes 3
    uv run play.py --fps 30                                # play twice as fast

Controls in the game window: press Q to quit.
"""

import argparse

import cv2
import numpy as np
import torch

from common import DEFAULT_ENV_ID, QNetwork, get_device, make_env

WINDOW_NAME = "DQN Agent"


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play")
    parser.add_argument("--model", type=str, default="dqn_model.pth", help="path to saved weights")
    parser.add_argument("--env", type=str, default=DEFAULT_ENV_ID, help="must match the training env")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=15,
                        help="agent decisions per second (15 = real game speed, since each decision lasts 4 frames)")
    parser.add_argument("--scale", type=int, default=4, help="window size multiplier")
    return parser.parse_args()


def show_frame(frame, scale, status_text, fps):
    """Enlarge the game frame, add a status bar under it and display it.

    Returns False if the user pressed Q (quit), True otherwise.
    """
    height, width = frame.shape[:2]
    # INTER_NEAREST keeps the pixel-art look sharp instead of blurry
    big = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR colour order

    # Black bar at the bottom for the status text
    big = cv2.copyMakeBorder(big, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(big, status_text, (10, big.shape[0] - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, big)
    key = cv2.waitKey(int(1000 / fps))  # also controls the playing speed
    return key not in (ord("q"), ord("Q"))


def main():
    args = parse_args()
    device = get_device()

    # Same environment and preprocessing as training.
    # "rgb_array" gives us the colour frames so we can draw our own window.
    env = make_env(args.env, render_mode="rgb_array")
    num_actions = env.action_space.n
    action_names = env.unwrapped.get_action_meanings()

    # Rebuild the same network architecture and load the trained weights
    q_network = QNetwork(num_actions).to(device)
    q_network.load_state_dict(torch.load(args.model, map_location=device))
    q_network.eval()  # evaluation mode: no training

    keep_playing = True
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward, steps, done = 0.0, 0, False

        while not done and keep_playing:
            # Greedy action: always pick the action with the highest Q-value (no exploration)
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)
                action = int(q_network(state_tensor).argmax(dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            status = (f"Episode {episode}/{args.episodes}   Score: {total_reward:+.0f}   "
                      f"Step: {steps}   Action: {action_names[action]}")
            keep_playing = show_frame(env.render(), args.scale, status, args.fps)

        print(f"Episode {episode}: score = {total_reward:+.0f}, length = {steps} steps")
        if not keep_playing:
            break

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()