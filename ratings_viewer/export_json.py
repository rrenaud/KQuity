"""Export rating history to JSON for the ratings viewer UI.

Usage:
    python ratings_viewer/export_json.py

Generates ratings_viewer/ratings_history.json from logged_in_games/ data.
"""

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute_ratings import compute_ratings, load_data, RatingResult


def main() -> None:
    usergame, game_cabinets, outcomes = load_data()

    print('Computing ratings with history...')
    result: RatingResult = compute_ratings(
        outcomes, usergame, game_cabinets, record_history=True)
    assert result.history is not None, 'record_history=True should populate history'
    print(f'  {len(result.history)} games with rating updates')

    # Build player index with per-role stats
    # Count games per (user_id, role) from history
    game_counts: dict[tuple[Any, str], int] = {}  # (user_id, role) -> count
    for event in result.history:
        for p in event['participants']:
            key = (p['user_id'], p['role'])
            game_counts[key] = game_counts.get(key, 0) + 1

    players: dict[str, dict[str, Any]] = {}
    for (user_id, role), rating in result.player_ratings.items():
        uid_str = str(user_id)
        if uid_str not in players:
            players[uid_str] = {
                'user_id': user_id,
                'name': result.user_names.get(user_id, f'user_{user_id}'),
                'roles': {},
            }
        players[uid_str]['roles'][role] = {
            'final_mu': round(rating.mu, 2),
            'final_sigma': round(rating.sigma, 2),
            'game_count': game_counts.get((user_id, role), 0),
        }

    # Add cabinet pseudo-players
    for (cab, role), rating in result.cabinet_anon_ratings.items():
        uid_str = f'cab_{cab}_{role}'
        if uid_str not in players:
            # Derive a display name from the cabinet name
            display_name = f'Anon @ {cab}'
            players[uid_str] = {
                'user_id': uid_str,
                'name': display_name,
                'roles': {},
                'is_cabinet': True,
            }
        players[uid_str]['roles'][role] = {
            'final_mu': round(rating.mu, 2),
            'final_sigma': round(rating.sigma, 2),
            'game_count': None,
        }

    # Round floats in game history
    games: list[dict[str, Any]] = []
    for event in result.history:
        participants: list[dict[str, Any]] = []
        for p in event['participants']:
            participants.append({
                'user_id': p['user_id'],
                'name': p['name'],
                'position': p['position'],
                'role': p['role'],
                'team': p['team'],
                'mu_before': round(p['mu_before'], 2),
                'mu_after': round(p['mu_after'], 2),
                'sigma': round(p['sigma'], 2),
            })
        games.append({
            'game_id': event['game_id'],
            'timestamp': event['timestamp'],
            'winner': event['winner'],
            'participants': participants,
        })

    output: dict[str, Any] = {
        'metadata': {
            'total_games': len(result.history),
            'total_players': len(players),
        },
        'players': players,
        'games': games,
        'cabinet_snapshots': result.cabinet_snapshots,
    }

    output_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'ratings_history.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)

    size_mb: float = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Wrote {output_path} ({size_mb:.1f} MB)')
    print(f'  {len(players)} players, {len(games)} games')


if __name__ == '__main__':
    main()
