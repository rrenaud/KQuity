"""Symmetry augmentation for KQuity feature vectors and event streams.

Killer Queen has perfect Blue/Gold team symmetry: any game state where Blue
has resources X and Gold has resources Y is strategically equivalent (from the
opposite perspective) to Blue having Y and Gold having X.

This module provides two approaches:
1. swap_teams(X, y) - fast numpy feature vector swap (for data augmentation)
2. swap_event_stream(events) - event-level swap (for cross-verification)
"""

import numpy as np
import numpy.typing as npt

from preprocess import GameEvent

# --- Feature vector swap constants (52 features, no ratings) ---

# Feature layout (52 features):
#   [0:20]   Blue team
#   [20:40]  Gold team
#   [40:45]  Maiden control (+1=Blue, -1=Gold, 0=Neutral)
#   [45:49]  Map one-hot (unchanged)
#   [49]     snail_position (positive = toward Blue goal)
#   [50]     snail_velocity (positive = toward Blue goal)
#   [51]     berries_available (team-neutral)

# Permutation: swap blue[0:20] <-> gold[20:40], keep rest in place
SWAP_PERM: list[int] = list(range(20, 40)) + list(range(0, 20)) + list(range(40, 52))

# Sign flips: negate maiden control and snail pos/vel
SWAP_SIGN: npt.NDArray[np.float32] = np.ones(52, dtype=np.float32)
SWAP_SIGN[40:45] = -1.0  # Maiden control: Blue(+1) <-> Gold(-1)
SWAP_SIGN[49:51] = -1.0  # Snail pos & vel: flip perspective

# --- Feature vector swap constants (62 features, with ratings) ---

# Feature layout (62 features):
#   [0:25]   Blue team (4 stats + queen_mu + 4 workers × 5 attrs)
#   [25:50]  Gold team (same)
#   [50:55]  Maiden control
#   [55:59]  Map one-hot
#   [59]     snail_position
#   [60]     snail_velocity
#   [61]     berries_available

SWAP_PERM_62: list[int] = list(range(25, 50)) + list(range(0, 25)) + list(range(50, 62))

SWAP_SIGN_62: npt.NDArray[np.float32] = np.ones(62, dtype=np.float32)
SWAP_SIGN_62[50:55] = -1.0  # Maiden control
SWAP_SIGN_62[59:61] = -1.0  # Snail pos & vel


def swap_teams(X: npt.NDArray[np.float32],
               y: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Swap Blue/Gold perspective on materialized feature matrices.

    Args:
        X: numpy array of shape (N, 52) or (N, 62), feature vectors
        y: numpy array of shape (N,), labels (1=Blue wins, 0=Gold wins)

    Returns:
        (swapped_X, swapped_y): same shapes, with teams swapped
    """
    if X.shape[1] == 62:
        return X[:, SWAP_PERM_62] * SWAP_SIGN_62, 1 - y
    return X[:, SWAP_PERM] * SWAP_SIGN, 1 - y


def swap_event_stream(events: list[GameEvent]) -> list[GameEvent]:
    """Swap Blue/Gold teams on a list of GameEvent objects.

    Calls each event's swap_teams() method, which is implemented
    on the GameEvent subclasses in preprocess.py.

    Args:
        events: list of GameEvent objects for a single game

    Returns:
        list of new GameEvent objects with teams swapped
    """
    return [event.swap_teams() for event in events]
