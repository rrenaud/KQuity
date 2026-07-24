#!/usr/bin/env python3
"""Empirical kill / death matrix over real games.

Model-free counterpart to the break-even p*. For every playerKill in the data we
classify the killer and the victim by piece type -- read from the tracked game
state at the instant of the kill, *before* the victim's upgrades are stripped --
and tally a killer x victim count matrix. From it we read, per matchup A vs B:

  kills_for = # times an A killed a B          (A is the attacker/killer)
  kills_ag  = # times a B killed an A          (A is the victim)
  win_rate  = kills_for / (kills_for + kills_ag)   empirical P(A wins the exchange)
  kd        = kills_for / kills_ag                 empirical kill/death ratio

Piece type is (has_wings, has_speed); the queen is its own type. Bare and speed
drones can't strike, so their killer rows are ~empty (snail / queen-kick / other
attributions aside); their signal is in the victim columns. Killers/victims that
aren't a queen or worker pid (environmental, snail) are counted as `n_other`.

Usage:
  python -m combat_value.kills [--data ...] [--max-games N] [--out path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_codec import read_packed_games, walk_game_events, OP_PLAYER_KILL

# Killer/victim rows and columns, in the vdecomp axis order + bare drone last.
PIECES = ['queen', 'speed_warrior', 'vanilla_warrior', 'speed_drone', 'drone']


def _piece_of(game_state, pid: int) -> str | None:
    """Piece type of `pid` in the current state, or None if not a player unit."""
    if pid == 1 or pid == 2:
        return 'queen'
    if 3 <= pid <= 10:
        w = game_state.teams[pid % 2].workers[(pid - 3) // 2]
        if w.has_wings:
            return 'speed_warrior' if w.has_speed else 'vanilla_warrior'
        return 'speed_drone' if w.has_speed else 'drone'
    return None


def tally_kills(data_path: str, max_games: int | None = None,
                progress: bool = True) -> dict:
    counts = {k: {v: 0 for v in PIECES} for k in PIECES}
    n_games = n_kills = n_other = 0
    for _game_id, encoded in read_packed_games(data_path):
        if max_games is not None and n_games >= max_games:
            break
        n_games += 1
        for _rel_ts, opcode, payload, gs in walk_game_events(encoded):
            if opcode != OP_PLAYER_KILL:
                continue
            killer = _piece_of(gs, payload >> 4)
            victim = _piece_of(gs, payload & 0xF)
            if killer is None or victim is None:
                n_other += 1
                continue
            counts[killer][victim] += 1
            n_kills += 1
        if progress and n_games % 5000 == 0:
            print(f"\r  {n_games:,} games · {n_kills:,} kills", end='', flush=True)
    if progress:
        print(f"\r  {n_games:,} games · {n_kills:,} classified kills · "
              f"{n_other:,} environmental/other")
    return {'pieces': PIECES, 'counts': counts,
            'meta': {'data': data_path, 'n_games': n_games,
                     'n_kills': n_kills, 'n_other': n_other}}


def matchup_stats(counts: dict, atk: str, dfn: str) -> dict:
    """Empirical exchange stats for attacker `atk` vs defender `dfn`."""
    kills_for = counts[atk][dfn]
    kills_ag = counts[dfn][atk]
    total = kills_for + kills_ag
    return {
        'kills_for': kills_for,
        'kills_ag': kills_ag,
        'n': total,
        'win_rate': (kills_for / total) if total else None,
        'kd': (kills_for / kills_ag) if kills_ag else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=None)
    ap.add_argument('--out', default='combat_value/_kill_matrix.json')
    args = ap.parse_args()

    print(f"Tallying kills from {args.data} (limit {args.max_games or 'all'})")
    result = tally_kills(args.data, max_games=args.max_games)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=0)
    print(f"Wrote {args.out}")

    counts = result['counts']
    print(f"\n  killer \\ victim  {'  '.join(f'{p[:5]:>6}' for p in PIECES)}")
    for k in PIECES:
        row = '  '.join(f"{counts[k][v]:>6,}" for v in PIECES)
        print(f"  {k:16s} {row}")

    print("\n  matchup (A vs B)          A->B    B->A   win%    K/D")
    for a in PIECES:
        for d in PIECES:
            s = matchup_stats(counts, a, d)
            if not s['n']:
                continue
            wr = f"{s['win_rate']:.3f}" if s['win_rate'] is not None else '  -  '
            kd = f"{s['kd']:.2f}" if s['kd'] is not None else '  inf'
            print(f"  {a:15s} vs {d:15s} {s['kills_for']:>6,} {s['kills_ag']:>6,}"
                  f"  {wr}  {kd:>6}")


if __name__ == '__main__':
    main()
