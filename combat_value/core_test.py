#!/usr/bin/env python3
"""Unit tests for combat_value.core (no model / data required).

Run: pytest combat_value/core_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from combat_value import core


def make_state(blue, gold):
    """Build one 52-dim state; team spec: {eggs, workers=[(wings,speed),...]}."""
    x = np.zeros(core.NUM_FEATURES, np.float32)
    for team, spec in (('blue', blue), ('gold', gold)):
        off = core.TEAM_OFFSET[team]
        x[off + core.EGGS] = spec.get('eggs', 2)
        workers = spec.get('workers', [])
        x[off + core.N_VANILLA] = sum(1 for w, s in workers if w and not s)
        x[off + core.N_SPEED] = sum(1 for w, s in workers if w and s)
        for i in range(4):
            w, s = workers[i] if i < len(workers) else (0, 0)
            b = off + core.WORKER_BASE + 4 * i
            x[b + core.W_WINGS] = float(w)
            x[b + core.W_SPEED] = float(s)
    return x


# --------------------------------------------------------------------------
# State edits
# --------------------------------------------------------------------------
def test_kill_queen_terminal_and_nonterminal():
    x = make_state({'eggs': 2}, {'eggs': 0})[None, :]
    xk, term = core.kill_queen(x, 'gold')
    assert xk[0, core.TEAM_OFFSET['gold'] + core.EGGS] == -1.0
    assert term[0]
    xk2, term2 = core.kill_queen(x, 'blue')
    assert xk2[0, core.EGGS] == 1.0 and not term2[0]


def test_kill_worker_dewings_and_decrements():
    x = make_state({'workers': [(1, 0)]}, {})[None, :]
    assert core.worker_type_count(x, 'blue', 'vanilla_warrior')[0] == 1
    xd, applic = core.kill_worker(x, 'blue', 'vanilla_warrior')
    assert applic[0]
    assert xd[0, core.N_VANILLA] == 0.0
    off = core.TEAM_OFFSET['blue'] + core.WORKER_BASE
    assert sum(xd[0, off + 4 * i + core.W_WINGS] for i in range(4)) == 0.0


def test_kill_worker_not_applicable_when_absent():
    x = make_state({'workers': [(1, 0)]}, {})[None, :]
    _, applic = core.kill_worker(x, 'gold', 'vanilla_warrior')
    assert not applic[0]


def test_speed_warrior_kill_decrements_speed_aggregate():
    x = make_state({'workers': [(1, 1)]}, {})[None, :]
    xd, applic = core.kill_worker(x, 'blue', 'speed_warrior')
    assert applic[0]
    assert xd[0, core.N_SPEED] == 0.0


def test_resort_keeps_workers_power_sorted():
    # blue: one speed warrior in slot 0, drones elsewhere -> after any edit,
    # blocks must be ascending by power.
    x = make_state({'workers': [(1, 1), (0, 0), (0, 0), (0, 0)]}, {})[None, :]
    core._resort_workers(x, 'blue')
    p = core._worker_power(x, 'blue')[0]
    assert np.all(np.diff(p) >= 0)


# --------------------------------------------------------------------------
# Break-even math
# --------------------------------------------------------------------------
def test_break_even_values():
    vS = np.array([0.5, 0.9, 0.1], np.float32)
    vK = np.array([0.8, 0.8, 0.8], np.float32)
    vD = np.array([0.2, 0.2, 0.2], np.float32)
    p = core.break_even(vS, vK, vD)
    assert abs(p[0] - 0.5) < 1e-4
    assert p[1] > 1.0   # status quo already above kill value -> never fight
    assert p[2] < 0.0   # status quo below death value -> always fight


def test_break_even_degenerate_is_nan():
    vS = np.array([0.5], np.float32)
    vK = np.array([0.5001], np.float32)
    vD = np.array([0.4999], np.float32)  # swing 2e-4 < 1e-3
    assert np.isnan(core.break_even(vS, vK, vD)[0])


# --------------------------------------------------------------------------
# End-to-end symmetry property with a symmetric stub model
# --------------------------------------------------------------------------
def _symmetric_predict(X):
    """P(blue wins) as a monotone function of (blue warriors - gold warriors).

    Symmetric by construction, so a mirror matchup must break even at 0.5.
    """
    bw = X[:, core.N_VANILLA] + X[:, core.N_SPEED]
    gw = X[:, core.TEAM_OFFSET['gold'] + core.N_VANILLA] + \
        X[:, core.TEAM_OFFSET['gold'] + core.N_SPEED]
    return 1.0 / (1.0 + np.exp(-0.7 * (bw - gw)))


def test_mirror_matchup_breaks_even_at_half():
    # Symmetric board: both teams 2 vanilla warriors + 2 drones, eggs 2.
    x = make_state({'workers': [(1, 0), (1, 0), (0, 0), (0, 0)]},
                   {'workers': [(1, 0), (1, 0), (0, 0), (0, 0)]})[None, :]
    res = core.evaluate_matchup(x, _symmetric_predict, 'blue',
                                'vanilla_warrior', 'vanilla_warrior')
    assert res.applicable[0]
    assert abs(res.pstar[0] - 0.5) < 1e-5


def test_both_sides_doubles_and_mirrors():
    x = make_state({'workers': [(1, 0), (0, 0), (0, 0), (0, 0)]},
                   {'workers': [(1, 0), (0, 0), (0, 0), (0, 0)]})[None, :]
    res = core.evaluate_matchup_both_sides(
        x, _symmetric_predict, 'vanilla_warrior', 'vanilla_warrior')
    assert len(res.pstar) == 2
    # symmetric board -> both perspectives give 0.5
    assert np.allclose(res.pstar, 0.5, atol=1e-5)


if __name__ == '__main__':
    import subprocess
    raise SystemExit(subprocess.call(['pytest', '-q', __file__]))
