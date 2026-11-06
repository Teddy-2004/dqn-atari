"""Code shared by train.py and play.py: the environment setup and the Q-network.

Keeping these in one place guarantees that play.py rebuilds exactly the same
environment and network architecture that train.py used.
"""

import ale_py
import gymnasium as gym
import torch
import torch.nn as nn

# Make the ALE/... Atari environments available to gym.make
gym.register_envs(ale_py)

DEFAULT_ENV_ID = "ALE/Boxing-v5"


def make_env(env_id=DEFAULT_ENV_ID, render_mode=None):
    """Create an Atari environment with the standard DQN preprocessing.

    - frame_skip=4: the agent picks an action every 4 frames (the action is repeated).
    - Frames are converted to grayscale and resized to 84x84.
    - The last 4 frames are stacked so the network can see motion (e.g. ball direction).
    Final observation shape: (4, 84, 84), dtype uint8 (values 0-255).
    """
    # frameskip=1 here because AtariPreprocessing does the frame skipping itself
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        noop_max=30,  # random number of "do nothing" actions at the start, for varied starts
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """CNN that maps a stack of 4 frames to one Q-value per action.

    Architecture from the DQN Nature paper (Mnih et al., 2015):
        Conv(32, 8x8, stride 4) -> ReLU
        Conv(64, 4x4, stride 2) -> ReLU
        Conv(64, 3x3, stride 1) -> ReLU
        Flatten -> Linear(3136 -> 512) -> ReLU -> Linear(512 -> num_actions)
    """

    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # (4, 84, 84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),                                # -> 64 * 7 * 7 = 3136
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # Frames arrive as uint8 in [0, 255]; scale them to [0, 1] floats
        x = x.float() / 255.0
        return self.fc(self.conv(x))


def get_device():
    """Use a GPU if one is available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")
