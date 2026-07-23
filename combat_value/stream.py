#!/usr/bin/env python3
"""Streaming bucket aggregator for the model reports at full scale.

The model reports (heatmap, lines, vdecomp) each need, per matchup, the same
per-bucket table that analysis.bucket_table produces — but on 30M+ states that
can't be materialized at once. This module streams the dataset in batches and
accumulates, per (def_eggs, net_warriors) bucket:
  * V-sums (exact means -> p* from means, mean V's)
  * a p* histogram (median / IQR without holding all states)
plus a per-matchup global summary. The result is cached to JSON so all three
reports render from it instantly.

Usage:
  python -m combat_value.stream [--data ...] [--model ...] [--max-games N]
                                [--batch-games 4000] [--out cache.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from combat_value import core, analysis
from combat_value.html_report import MATRIX_MATCHUPS

# p* histogram: fixed range covers the meaningful decision band; the rare
# always-fight (<-1) / never-fight (>2) tails fall into under/overflow.
HIST_LO, HIST_HI, HIST_BINS = -1.0, 2.0, 600
_BINW = (HIST_HI - HIST_LO) / HIST_BINS
_CENTERS = HIST_LO + (np.arange(HIST_BINS) + 0.5) * _BINW
# value support for weighted quantiles: [LO, centers..., HI]
_QVALUES = np.concatenate([[HIST_LO], _CENTERS, [HIST_HI]])

NW_MAX = 6
NWN = 2 * NW_MAX + 1
EGG_MAX = 2


def matchup_key(atk: str, dfn: str) -> str:
    return f"{atk}|{dfn}"


def _new_accum():
    return {
        'svS': np.zeros((EGG_MAX + 1, NWN), np.float64),
        'svK': np.zeros((EGG_MAX + 1, NWN), np.float64),
        'svD': np.zeros((EGG_MAX + 1, NWN), np.float64),
        'cn':  np.zeros((EGG_MAX + 1, NWN), np.float64),
        'hist': np.zeros((EGG_MAX + 1, NWN, HIST_BINS), np.float64),
        'under': np.zeros((EGG_MAX + 1, NWN), np.float64),
        'over':  np.zeros((EGG_MAX + 1, NWN), np.float64),
        'n_applicable': 0,
        'n_terminal_kill': 0,
    }


def _accumulate(acc, res):
    app = res.applicable
    acc['n_applicable'] += int(app.sum())
    acc['n_terminal_kill'] += int((res.terminal_kill & app).sum())

    m = analysis.valid_mask(res)
    de = np.clip(res.feats['def_eggs'][m], 0, EGG_MAX)
    nw = np.clip(res.feats['net_warriors'][m], -NW_MAX, NW_MAX) + NW_MAX
    ps = res.pstar[m]
    np.add.at(acc['svS'], (de, nw), res.vS[m])
    np.add.at(acc['svK'], (de, nw), res.vK[m])
    np.add.at(acc['svD'], (de, nw), res.vD[m])
    np.add.at(acc['cn'], (de, nw), 1.0)

    under = ps < HIST_LO
    over = ps >= HIST_HI
    inside = ~(under | over)
    bi = np.clip(((ps - HIST_LO) / _BINW).astype(np.int64), 0, HIST_BINS - 1)
    np.add.at(acc['hist'], (de[inside], nw[inside], bi[inside]), 1.0)
    np.add.at(acc['under'], (de[under], nw[under]), 1.0)
    np.add.at(acc['over'], (de[over], nw[over]), 1.0)


def _wq(weights, q):
    """Weighted quantile over _QVALUES given a weight vector (len HIST_BINS+2)."""
    total = weights.sum()
    if total <= 0:
        return float('nan')
    cum = np.cumsum(weights)
    i = int(np.searchsorted(cum, q * total, side='left'))
    return float(_QVALUES[min(i, len(_QVALUES) - 1)])


def _bucket_weights(acc, de, j):
    return np.concatenate([[acc['under'][de, j]], acc['hist'][de, j], [acc['over'][de, j]]])


def _finalize_rows(acc) -> list[dict]:
    rows = []
    for de in range(EGG_MAX + 1):
        for j in range(NWN):
            n = acc['cn'][de, j]
            if n < 1:
                continue
            mS = acc['svS'][de, j] / n
            mK = acc['svK'][de, j] / n
            mD = acc['svD'][de, j] / n
            swing = mK - mD
            pstar = (mS - mD) / swing if abs(swing) > 1e-3 else float('nan')
            w = _bucket_weights(acc, de, j)
            rows.append({
                'def_eggs': de, 'net_warriors': j - NW_MAX, 'n': int(n),
                'pstar': float(pstar),
                'pstar_median': _wq(w, 0.50),
                'pstar_p25': _wq(w, 0.25),
                'pstar_p75': _wq(w, 0.75),
                'mean_V_status_quo': float(mS),
                'mean_V_kill': float(mK),
                'mean_V_death': float(mD),
            })
    rows.sort(key=lambda r: (r['def_eggs'], r['net_warriors']))
    return rows


def _finalize_summary(acc, label: str) -> dict:
    n = acc['cn'].sum()
    if n < 1:
        return {'label': label, 'n': 0}
    w = (np.concatenate([[acc['under'].sum()], acc['hist'].sum(axis=(0, 1)),
                         [acc['over'].sum()]]))
    total = w.sum()
    cum = np.cumsum(w)
    # fraction of p* <= 0 and >= 1 from the histogram support
    le0 = float(cum[np.searchsorted(_QVALUES, 0.0, side='right') - 1] / total)
    ge1 = float((total - cum[np.searchsorted(_QVALUES, 1.0, side='left') - 1]) / total)
    return {
        'label': label,
        'n': int(n),
        'n_applicable': acc['n_applicable'],
        'n_terminal_kill': acc['n_terminal_kill'],
        'pstar_mean': float((_CENTERS * acc['hist'].sum(axis=(0, 1))).sum()
                            / max(acc['hist'].sum(), 1e-9)),
        'pstar_p5': _wq(w, 0.05),
        'pstar_p25': _wq(w, 0.25),
        'pstar_median': _wq(w, 0.50),
        'pstar_p75': _wq(w, 0.75),
        'pstar_p95': _wq(w, 0.95),
        'frac_always_fight': le0,
        'frac_never_fight': ge1,
        'mean_V_status_quo': float(acc['svS'].sum() / n),
        'mean_V_kill': float(acc['svK'].sum() / n),
        'mean_V_death': float(acc['svD'].sum() / n),
    }


def compute_cache(data_path, model, matchups, batch_games=4000, max_games=None,
                  progress=True) -> dict:
    from event_codec import read_packed_games, materialize_entries

    accums = {matchup_key(a, d): _new_accum() for a, d, *_ in matchups}
    batch, n_games, n_states = [], 0, 0

    def flush(entries):
        nonlocal n_states
        if not entries:
            return
        X, y, gids, ts = materialize_entries(entries)
        n_states += len(X)
        # Share predictions across matchups: the unique (piece, team) kills are
        # predicted once and reused, ~10x fewer model queries than evaluating
        # each matchup independently, with identical results.
        results = core.evaluate_matchups_shared(X, model.predict, matchups,
                                                game_ids=gids, timestamps=ts)
        for a, d, *_ in matchups:
            _accumulate(accums[matchup_key(a, d)], results[(a, d)])

    for game_id, encoded in read_packed_games(data_path):
        if max_games is not None and n_games >= max_games:
            break
        batch.append((game_id, encoded))
        n_games += 1
        if len(batch) >= batch_games:
            flush(batch)
            batch = []
            if progress:
                print(f"\r  {n_games:,} games · {n_states:,} states", end='', flush=True)
    flush(batch)
    if progress:
        print(f"\r  {n_games:,} games · {n_states:,} states")

    out = {'matchups': {}}
    for a, d, *rest in matchups:
        k = matchup_key(a, d)
        label = f"{a} vs {d}"
        out['matchups'][k] = {
            'attacker_piece': a, 'defender_piece': d,
            'rows': _finalize_rows(accums[k]),
            'summary': _finalize_summary(accums[k], label),
        }
    out['meta'] = {'data': data_path, 'n_games': n_games, 'n_states': n_states}
    return out


def load_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cache_grid(cache: dict, atk: str, dfn: str) -> dict:
    """{(def_eggs, net_warriors): row} for a matchup, from a loaded cache."""
    entry = cache['matchups'][matchup_key(atk, dfn)]
    return {(r['def_eggs'], r['net_warriors']): r for r in entry['rows']}


def cache_summary(cache: dict, atk: str, dfn: str) -> dict:
    return cache['matchups'][matchup_key(atk, dfn)]['summary']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=None)
    ap.add_argument('--batch-games', type=int, default=4000)
    ap.add_argument('--out', default='combat_value/_bucket_cache.json')
    args = ap.parse_args()

    import lightgbm as lgb
    print(f"Loading model {args.model}")
    model = lgb.Booster(model_file=args.model)
    print(f"Streaming {args.data} (batch {args.batch_games}, "
          f"limit {args.max_games or 'all'}, {len(MATRIX_MATCHUPS)} matchups)")
    cache = compute_cache(args.data, model, MATRIX_MATCHUPS,
                          batch_games=args.batch_games, max_games=args.max_games)
    with open(args.out, 'w') as f:
        json.dump(cache, f)
    print(f"\nWrote {args.out}  ({cache['meta']['n_games']:,} games, "
          f"{cache['meta']['n_states']:,} states)")
    for a, d, *_ in MATRIX_MATCHUPS:
        s = cache_summary(cache, a, d)
        med = s.get('pstar_median')
        print(f"  {a:16s} vs {d:16s} median p*="
              f"{('%.3f' % med) if med is not None else 'n/a':>6}  n={s.get('n', 0):,}")


if __name__ == '__main__':
    main()
