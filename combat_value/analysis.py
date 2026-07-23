#!/usr/bin/env python3
"""Fine-grained and coarse-grained summaries of break-even results.

Fine-grained: per-state p* distribution over a dataset, and a per-game p*
curve over one game's event sequence.

Coarse-grained: bucket states by decision-relevant variables and report the
break-even computed *from bucket-mean V's* (stable), alongside the median and
IQR of the per-state p* within the bucket (spread). Averaging the V's first,
then dividing, avoids letting a few tiny-denominator states blow up the mean.
"""
from __future__ import annotations

import numpy as np

from . import core


# --------------------------------------------------------------------------
# Fine-grained
# --------------------------------------------------------------------------
def valid_mask(res: core.MatchupResult) -> np.ndarray:
    """Applicable states with a finite p* (excludes degenerate swings)."""
    return res.applicable & np.isfinite(res.pstar)


def summarize(res: core.MatchupResult) -> dict:
    """Distribution summary of p* over all valid states in a matchup."""
    m = valid_mask(res)
    ps = res.pstar[m]
    if len(ps) == 0:
        return {'label': res.label(), 'n': 0}
    pct = np.percentile(ps, [5, 25, 50, 75, 95])
    return {
        'label': res.label(),
        'n': int(m.sum()),
        'n_applicable': int(res.applicable.sum()),
        'n_terminal_kill': int((res.terminal_kill & res.applicable).sum()),
        'pstar_mean': float(np.mean(ps)),
        'pstar_p5': float(pct[0]),
        'pstar_p25': float(pct[1]),
        'pstar_median': float(pct[2]),
        'pstar_p75': float(pct[3]),
        'pstar_p95': float(pct[4]),
        'frac_always_fight': float(np.mean(ps <= 0.0)),
        'frac_never_fight': float(np.mean(ps >= 1.0)),
        'mean_V_status_quo': float(np.mean(res.vS[m])),
        'mean_V_kill': float(np.mean(res.vK[m])),
        'mean_V_death': float(np.mean(res.vD[m])),
    }


def game_curve(res: core.MatchupResult, game_id: int) -> dict:
    """Per-event p* curve for a single game (needs game_ids/timestamps).

    Only meaningful for a single-attacker result (attacker in {'blue','gold'});
    a 'both' result stacks each state twice.
    """
    if res.game_ids is None or res.timestamps is None:
        raise ValueError("game_curve needs game_ids and timestamps")
    sel = (res.game_ids == game_id) & valid_mask(res)
    order = np.argsort(res.timestamps[sel], kind='stable')
    return {
        'game_id': int(game_id),
        'label': res.label(),
        't': res.timestamps[sel][order].astype(float).tolist(),
        'pstar': res.pstar[sel][order].astype(float).tolist(),
        'V_status_quo': res.vS[sel][order].astype(float).tolist(),
        'V_kill': res.vK[sel][order].astype(float).tolist(),
        'V_death': res.vD[sel][order].astype(float).tolist(),
    }


# --------------------------------------------------------------------------
# Coarse-grained bucketing
# --------------------------------------------------------------------------
def bucket_table(res: core.MatchupResult, keys: list[str]) -> list[dict]:
    """Group valid states by the given decision-feature keys.

    For each group report count, mean V's, break-even p* computed from those
    means, and the median / IQR of per-state p* within the group.
    `keys` are names in res.feats (e.g. 'def_eggs', 'net_warriors').
    """
    m = valid_mask(res)
    if m.sum() == 0:
        return []
    key_arrs = [res.feats[k][m] for k in keys]
    vS, vK, vD, ps = res.vS[m], res.vK[m], res.vD[m], res.pstar[m]

    # Build a composite grouping via unique rows of the stacked keys.
    stacked = np.stack(key_arrs, axis=1)
    uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
    inv = inv.ravel()

    rows = []
    for g in range(len(uniq)):
        gm = inv == g
        n = int(gm.sum())
        mvS, mvK, mvD = float(vS[gm].mean()), float(vK[gm].mean()), float(vD[gm].mean())
        swing = mvK - mvD
        pstar_means = (mvS - mvD) / swing if abs(swing) > 1e-3 else float('nan')
        gp = ps[gm]
        row = {keys[i]: int(uniq[g, i]) for i in range(len(keys))}
        row.update({
            'n': n,
            'pstar': pstar_means,               # from bucket-mean V's (stable)
            'pstar_median': float(np.median(gp)),
            'pstar_p25': float(np.percentile(gp, 25)),
            'pstar_p75': float(np.percentile(gp, 75)),
            'mean_V_status_quo': mvS,
            'mean_V_kill': mvK,
            'mean_V_death': mvD,
        })
        rows.append(row)
    rows.sort(key=lambda r: tuple(r[k] for k in keys))
    return rows


def format_bucket_table(rows: list[dict], keys: list[str], min_n: int = 50) -> str:
    """Render a bucket table as aligned text (skips groups below min_n)."""
    shown = [r for r in rows if r['n'] >= min_n]
    if not shown:
        return "(no buckets above min_n)"
    head = keys + ['n', 'p*', 'p*_med', 'IQR', 'V(S)', 'V_kill', 'V_death']
    widths = [max(len(h), 8) for h in head]
    lines = ['  '.join(h.ljust(w) for h, w in zip(head, widths))]
    for r in shown:
        cells = [str(r[k]) for k in keys]
        cells += [
            f"{r['n']:,}",
            f"{r['pstar']:.3f}",
            f"{r['pstar_median']:.3f}",
            f"[{r['pstar_p25']:.2f},{r['pstar_p75']:.2f}]",
            f"{r['mean_V_status_quo']:.3f}",
            f"{r['mean_V_kill']:.3f}",
            f"{r['mean_V_death']:.3f}",
        ]
        lines.append('  '.join(c.ljust(w) for c, w in zip(cells, widths)))
    return '\n'.join(lines)
