#!/usr/bin/env python3
"""Break-even combat value CLI.

Examples:
  # Vanilla warrior attacking a queen, coarse table by defender queen lives:
  python -m combat_value --attacker vanilla_warrior --defender queen \
      --coarse def_eggs net_warriors

  # Warrior-vs-warrior, dump full results to JSON:
  python -m combat_value --attacker vanilla_warrior --defender vanilla_warrior \
      --json out.json

  # Per-game p* curve for one game (single attacking side):
  python -m combat_value --attacker vanilla_warrior --defender queen \
      --game-id 12345 --side blue
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb

from combat_value import core, analysis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=2000)
    ap.add_argument('--attacker', default='vanilla_warrior', choices=core.ALL_PIECES,
                    help="attacking piece type")
    ap.add_argument('--defender', default='queen', choices=core.ALL_PIECES,
                    help="defending piece type")
    ap.add_argument('--side', default='both', choices=('blue', 'gold', 'both'),
                    help="attacking team (both = use every state twice)")
    ap.add_argument('--coarse', nargs='+', default=['def_eggs', 'net_warriors'],
                    help="decision-feature keys to bucket by")
    ap.add_argument('--min-n', type=int, default=50,
                    help="min states per bucket to display")
    ap.add_argument('--game-id', type=int, default=None,
                    help="print a per-event p* curve for this game")
    ap.add_argument('--json', default=None, help="write full summary JSON here")
    args = ap.parse_args()

    print(f"Loading model {args.model}")
    model = lgb.Booster(model_file=args.model)

    print(f"Materializing up to {args.max_games} games from {args.data}")
    from event_codec import fast_materialize_from_codec
    X, y, gids, ts = fast_materialize_from_codec(args.data, max_games=args.max_games)
    print(f"  {len(X):,} states")

    if args.side == 'both':
        res = core.evaluate_matchup_both_sides(
            X, model.predict, args.attacker, args.defender,
            game_ids=gids, timestamps=ts)
    else:
        res = core.evaluate_matchup(
            X, model.predict, args.side, args.attacker, args.defender,
            game_ids=gids, timestamps=ts)

    summary = analysis.summarize(res)
    print(f"\n=== {res.label()} ===")
    print(f"applicable states: {summary.get('n_applicable', 0):,}   "
          f"valid (finite p*): {summary['n']:,}   "
          f"last-life-queen kills: {summary.get('n_terminal_kill', 0):,}")
    if summary['n']:
        print(f"p*: median={summary['pstar_median']:.3f}  "
              f"mean={summary['pstar_mean']:.3f}  "
              f"IQR=[{summary['pstar_p25']:.3f}, {summary['pstar_p75']:.3f}]  "
              f"5-95%=[{summary['pstar_p5']:.3f}, {summary['pstar_p95']:.3f}]")
        print(f"always-fight (p*<=0): {summary['frac_always_fight']*100:.1f}%   "
              f"never-fight (p*>=1): {summary['frac_never_fight']*100:.1f}%")
        print(f"mean V(S)={summary['mean_V_status_quo']:.3f}  "
              f"V_kill={summary['mean_V_kill']:.3f}  "
              f"V_death={summary['mean_V_death']:.3f}")

    rows = analysis.bucket_table(res, args.coarse)
    print(f"\n=== coarse break-even by {', '.join(args.coarse)} "
          f"(min_n={args.min_n}) ===")
    print(analysis.format_bucket_table(rows, args.coarse, min_n=args.min_n))

    curve = None
    if args.game_id is not None:
        curve = analysis.game_curve(res, args.game_id)
        print(f"\n=== p* curve for game {args.game_id} "
              f"({len(curve['t'])} points) ===")
        for t, p in zip(curve['t'], curve['pstar']):
            print(f"  t={t:7.1f}s  p*={p:.3f}")

    if args.json:
        out = {
            'matchup': res.label(),
            'attacker_piece': args.attacker,
            'defender_piece': args.defender,
            'side': args.side,
            'data': args.data,
            'max_games': args.max_games,
            'summary': summary,
            'coarse_keys': args.coarse,
            'coarse_buckets': rows,
        }
        if curve is not None:
            out['game_curve'] = curve
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
