#!/usr/bin/env python3
"""Per-matchup empirical win-rate and model p*, bucketed by P(attacker wins).

For every real kill we score the pre-kill state with the win-probability model,
orient it to the attacker (the killer's team), and drop the event into one of
``--bins`` buckets along P(attacker wins) in [0, 1]. Within each bucket, per
matchup A vs B:

  win-rate = A-kills-B / (A-kills-B + B-kills-A)        empirical, from real kills
  p*       = mean model break-even at those same states  estimated, model-based

Both are oriented to the attacker, so every kill contributes a *win* + a p* to
matchup (killer, victim) at bucket P(killer wins), and a *loss* + the
complementary p* to the reverse matchup (victim, killer) at bucket
1 - P(killer wins). The two curves therefore live over the same population (the
states where A and B actually fought) and can be read against each other:
empirical above p* means those shots paid off.

Usage:
  python -m combat_value.kd_by_winprob [--data ...] [--bins 50] [--out path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb

from event_codec import (read_packed_games, walk_game_events, OP_PLAYER_KILL,
                         _game_state_to_vectorize_args, _vectorize_state,
                         NUM_FEATURES)
from combat_value import core
from combat_value.kills import _piece_of, PIECES

MIN_TS = 5.0            # skip the opening seconds, matching training materialization


def _apply_per_row_kill(X, piece_types, teams):
    """Kill piece_types[i] on teams[i] in row i. Returns (edited, terminal)."""
    out = X.copy()
    terminal = np.zeros(len(X), np.bool_)
    tname = {0: 'blue', 1: 'gold'}
    for t in (0, 1):
        for piece in PIECES:
            m = (teams == t) & (piece_types == piece)
            if not m.any():
                continue
            rows = np.where(m)[0]
            edited, _applic, term = core._apply_kill(X[rows], tname[t], piece)
            out[rows] = edited
            terminal[rows] = term
    return out, terminal


def _flush(model, rows, K_type, K_team, V_type, V_team, nb, acc):
    if not rows:
        return
    X = np.asarray(rows, np.float32)
    Kt = np.asarray(K_type, object)
    Vt = np.asarray(V_type, object)
    Kteam = np.asarray(K_team, np.int64)
    Vteam = np.asarray(V_team, np.int64)

    X_kill, term_kill = _apply_per_row_kill(X, Vt, Vteam)   # defender (victim) dies
    X_death, term_death = _apply_per_row_kill(X, Kt, Kteam)  # attacker (killer) dies
    N = len(X)
    preds = np.asarray(model.predict(np.concatenate([X, X_kill, X_death], axis=0)),
                       dtype=np.float64)
    pS, pK, pD = preds[:N], preds[N:2 * N], preds[2 * N:]

    # orient to the killer's team: P(attacker wins)
    att_blue = (Kteam == 0)
    vS = np.where(att_blue, pS, 1.0 - pS)
    vK = np.where(att_blue, pK, 1.0 - pK)
    vD = np.where(att_blue, pD, 1.0 - pD)
    vK = np.where(term_kill, 1.0, vK)     # defender's last queen died -> attacker wins
    vD = np.where(term_death, 0.0, vD)    # attacker's last queen died -> attacker loses

    swing = vK - vD
    with np.errstate(divide='ignore', invalid='ignore'):
        pstar = (vS - vD) / swing
    pstar = np.where(np.abs(swing) < 1e-3, np.nan, pstar)

    bwin = np.clip((vS * nb).astype(np.int64), 0, nb - 1)
    bloss = np.clip(((1.0 - vS) * nb).astype(np.int64), 0, nb - 1)
    pstar_rev = 1.0 - pstar

    for i in range(N):
        a = acc[(Kt[i], Vt[i])]
        a['wins'][bwin[i]] += 1
        if np.isfinite(pstar[i]):
            a['ps_sum'][bwin[i]] += pstar[i]
            a['ps_cnt'][bwin[i]] += 1
        r = acc[(Vt[i], Kt[i])]
        r['losses'][bloss[i]] += 1
        if np.isfinite(pstar_rev[i]):
            r['ps_sum'][bloss[i]] += pstar_rev[i]
            r['ps_cnt'][bloss[i]] += 1


def compute(data_path, model, nb=50, max_games=None, batch_games=4000,
            progress=True):
    acc = {(A, B): {'wins': np.zeros(nb), 'losses': np.zeros(nb),
                    'ps_sum': np.zeros(nb), 'ps_cnt': np.zeros(nb)}
           for A in PIECES for B in PIECES}
    rows, K_type, K_team, V_type, V_team = [], [], [], [], []
    n_games = n_kills = 0

    def flush():
        _flush(model, rows, K_type, K_team, V_type, V_team, nb, acc)
        rows.clear(); K_type.clear(); K_team.clear(); V_type.clear(); V_team.clear()

    for _gid, enc in read_packed_games(data_path):
        if max_games is not None and n_games >= max_games:
            break
        n_games += 1
        for rel_ts, op, payload, gs in walk_game_events(enc):
            if op != OP_PLAYER_KILL or rel_ts <= MIN_TS:
                continue
            kp, vp = payload >> 4, payload & 0xF
            K, V = _piece_of(gs, kp), _piece_of(gs, vp)
            if K is None or V is None:
                continue
            buf = np.empty((1, NUM_FEATURES), np.float32)
            (w, eggs, food, md, mi, sx, sv, st, ba, gsym) = \
                _game_state_to_vectorize_args(gs)
            _vectorize_state(buf, 0, w, eggs, food, md, mi, sx, sv, st,
                             rel_ts, ba, gsym)
            rows.append(buf[0])
            K_type.append(K); K_team.append(kp % 2)
            V_type.append(V); V_team.append(vp % 2)
            n_kills += 1
        if n_games % batch_games == 0:
            flush()
            if progress:
                print(f"\r  {n_games:,} games · {n_kills:,} kills", end='', flush=True)
    flush()
    if progress:
        print(f"\r  {n_games:,} games · {n_kills:,} kills")

    out = {'bins': nb, 'pieces': PIECES, 'matchups': {}}
    for (A, B), a in acc.items():
        if a['wins'].sum() + a['losses'].sum() == 0:
            continue
        out['matchups'][f'{A}|{B}'] = {
            'wins': a['wins'].astype(int).tolist(),
            'losses': a['losses'].astype(int).tolist(),
            'pstar_sum': [round(x, 4) for x in a['ps_sum'].tolist()],
            'pstar_cnt': a['ps_cnt'].astype(int).tolist(),
        }
    out['meta'] = {'data': data_path, 'n_games': n_games, 'n_kills': n_kills}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--bins', type=int, default=50)
    ap.add_argument('--max-games', type=int, default=None)
    ap.add_argument('--out', default='combat_value/_kd_winprob.json')
    args = ap.parse_args()

    print(f"Loading model {args.model}")
    model = lgb.Booster(model_file=args.model)
    print(f"Scoring kills from {args.data} into {args.bins} P(attacker-wins) bins")
    result = compute(args.data, model, nb=args.bins, max_games=args.max_games)
    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f"Wrote {args.out}  ({result['meta']['n_kills']:,} kills, "
          f"{len(result['matchups'])} matchups)")


if __name__ == '__main__':
    main()
