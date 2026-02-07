"""Symmetry augmentation for KQuity feature vectors and event streams.

Killer Queen has perfect Blue/Gold team symmetry: any game state where Blue
has resources X and Gold has resources Y is strategically equivalent (from the
opposite perspective) to Blue having Y and Gold having X.

This module provides two approaches:
1. swap_teams(X, y) - fast numpy feature vector swap (for data augmentation)
2. swap_event_stream(events) - event-level swap (for cross-verification)
"""

from typing import List

import numpy as np

from preprocess import GameEvent

# --- Feature vector swap constants ---

# Feature layout (52 features):
#   [0:20]   Blue team
#   [20:40]  Gold team
#   [40:45]  Maiden control (+1=Blue, -1=Gold, 0=Neutral)
#   [45:49]  Map one-hot (unchanged)
#   [49]     snail_position (positive = toward Blue goal)
#   [50]     snail_velocity (positive = toward Blue goal)
#   [51]     berries_available (team-neutral)

# Permutation: swap blue[0:20] <-> gold[20:40], keep rest in place
SWAP_PERM = list(range(20, 40)) + list(range(0, 20)) + list(range(40, 52))

# Sign flips: negate maiden control and snail pos/vel
SWAP_SIGN = np.ones(52, dtype=np.float32)
SWAP_SIGN[40:45] = -1.0  # Maiden control: Blue(+1) <-> Gold(-1)
SWAP_SIGN[49:51] = -1.0  # Snail pos & vel: flip perspective


def swap_teams(X, y):
    """Swap Blue/Gold perspective on materialized feature matrices.

    Args:
        X: numpy array of shape (N, 52), feature vectors
        y: numpy array of shape (N,), labels (1=Blue wins, 0=Gold wins)

    Returns:
        (swapped_X, swapped_y): same shapes, with teams swapped
    """
    return X[:, SWAP_PERM] * SWAP_SIGN, 1 - y


def swap_event_stream(events: List[GameEvent]) -> List[GameEvent]:
    """Swap Blue/Gold teams on a list of GameEvent objects.

    Calls each event's swap_teams() method, which is implemented
    on the GameEvent subclasses in preprocess.py.

    Args:
        events: list of GameEvent objects for a single game

    Returns:
        list of new GameEvent objects with teams swapped
    """
    return [event.swap_teams() for event in events]
