"""Symmetry augmentation for KQuity feature vectors and event streams.

Killer Queen has perfect Blue/Gold team symmetry: any game state where Blue
has resources X and Gold has resources Y is strategically equivalent (from the
opposite perspective) to Blue having Y and Gold having X.

This module provides two approaches:
1. swap_teams(X, y) - fast numpy feature vector swap (for data augmentation)
2. swap_event_stream(raw_events) - event-level swap (for cross-verification)
"""

import numpy as np

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


# --- Event stream swap ---

def _swap_pid(pid):
    """Swap a position ID between Blue and Gold teams.

    Blue PIDs are even (2,4,6,8,10), Gold PIDs are odd (1,3,5,7,9).
    Swap rule: even -> odd (pid+1), odd -> even (pid-1).
    """
    if pid % 2 == 0:
        return pid + 1
    else:
        return pid - 1


def swap_event_stream(raw_events):
    """Transform raw event tuples to swap Blue/Gold teams.

    Args:
        raw_events: list of (datetime, event_type, values_str) tuples
            where values_str is like "{val1,val2,...}"

    Returns:
        list of (datetime, event_type, values_str) tuples with teams swapped
    """
    swapped = []
    for dt, event_type, values_str in raw_events:
        if event_type == 'mapstart':
            # Flip gold_on_left
            vals = values_str[1:-1].split(',')
            gold_on_left = vals[1] == 'True'
            vals[1] = str(not gold_on_left)
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'spawn':
            # Swap pid
            vals = values_str[1:-1].split(',')
            vals[0] = str(_swap_pid(int(vals[0])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'carryFood':
            # Swap pid
            vals = values_str[1:-1].split(',')
            vals[0] = str(_swap_pid(int(vals[0])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'berryDeposit':
            # vals: hole_x, hole_y, pid
            # Swap pid, keep coords physical
            vals = values_str[1:-1].split(',')
            vals[2] = str(_swap_pid(int(vals[2])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'berryKickIn':
            # vals: hole_x, hole_y, pid, own_team
            # Swap pid, own_team stays the same (it's relative to the kicker)
            vals = values_str[1:-1].split(',')
            vals[2] = str(_swap_pid(int(vals[2])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'blessMaiden':
            # vals: x, y, color
            # Flip maiden x (1920 - x), swap color
            vals = values_str[1:-1].split(',')
            vals[0] = str(1920 - int(vals[0]))
            if vals[2] == 'Blue':
                vals[2] = 'Gold'
            else:
                vals[2] = 'Blue'
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'useMaiden':
            # vals: x, y, maiden_type, pid
            # Swap pid (maiden type unchanged)
            vals = values_str[1:-1].split(',')
            vals[3] = str(_swap_pid(int(vals[3])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'playerKill':
            # vals: x, y, killer_pid, killed_pid, category
            # Swap both pids, category unchanged
            vals = values_str[1:-1].split(',')
            vals[2] = str(_swap_pid(int(vals[2])))
            vals[3] = str(_swap_pid(int(vals[3])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'getOnSnail':
            # vals: snail_x, snail_y, rider_pid
            # Swap pid, keep coords physical
            vals = values_str[1:-1].split(',')
            vals[2] = str(_swap_pid(int(vals[2])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type == 'snailEat':
            # vals: snail_x, snail_y, rider_pid, eaten_pid
            # Swap both pids
            vals = values_str[1:-1].split(',')
            vals[2] = str(_swap_pid(int(vals[2])))
            vals[3] = str(_swap_pid(int(vals[3])))
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        elif event_type in ('getOffSnail', 'snailEscape'):
            # _process_game only reads vals[0] (snail x position) for these
            # events and sets velocity to 0. No pid swap needed.
            swapped.append((dt, event_type, values_str))

        elif event_type == 'victory':
            # Swap winning team
            vals = values_str[1:-1].split(',')
            if vals[0] == 'Blue':
                vals[0] = 'Gold'
            else:
                vals[0] = 'Blue'
            swapped.append((dt, event_type, '{' + ','.join(vals) + '}'))

        else:
            # gamestart and any others: pass through unchanged
            swapped.append((dt, event_type, values_str))

    return swapped
