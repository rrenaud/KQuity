"""Compute per-player OpenSkill ratings from game history.

Processes logged_in_games/ event CSVs chronologically, updating Plackett-Luce
ratings for logged-in players. Uses composite (user_id, role) keys where role
is 'queen' or 'drone', so each player has independent ratings for each role.

Outputs {game_id: np.array(10)} containing pre-game mu values for each
position (1-10), with queen positions getting queen-role mu and drone
positions getting drone-role mu.
"""

import csv
import datetime
from dataclasses import dataclass
import glob
import gzip
import os
import pickle

import numpy as np
from openskill.models import PlackettLuce


@dataclass
class RatingResult:
    ratings_by_game: dict
    player_ratings: dict
    user_names: dict
    cabinet_anon_ratings: dict
    history: list | None = None
    cabinet_snapshots: list | None = None


def load_usergame(path='unfiltered_partitioned/usergame.csv'):
    """Load usergame.csv -> {game_id: {position_id: (user_id, name)}}."""
    usergame = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header: id, game_id, position_id, user_id, name, scene
        for row in reader:
            game_id = int(row[1])
            position_id = int(row[2])
            user_id = int(row[3])
            name = row[4]
            if game_id not in usergame:
                usergame[game_id] = {}
            usergame[game_id][position_id] = (user_id, name)
    return usergame


def load_game_cabinets(path='unfiltered_partitioned/game.csv'):
    """Load game.csv -> {game_id: cabinet_name}."""
    cabinets = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            game_id = int(row[0])
            cabinet_name = row[8]
            cabinets[game_id] = cabinet_name
    return cabinets


def extract_game_outcomes(csv_pattern='logged_in_games/gameevents_*.csv.gz'):
    """Scan event CSVs to extract game_id -> (timestamp, winner).

    Returns dict mapping game_id to (first_event_timestamp, winning_team)
    where winning_team is 'Blue' or 'Gold'.
    """
    games = {}  # game_id -> {'min_ts': datetime, 'winner': str}

    for filename in sorted(glob.glob(csv_pattern)):
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                game_id = int(row[4])
                event_type = row[2]

                if event_type == 'gamestart':
                    ts = datetime.datetime.fromisoformat(row[1])
                    if game_id not in games:
                        games[game_id] = {'min_ts': ts, 'winner': None}
                    else:
                        if ts < games[game_id]['min_ts']:
                            games[game_id]['min_ts'] = ts

                elif event_type == 'victory':
                    vals = row[3][1:-1].split(',')
                    winner = vals[0]  # 'Blue' or 'Gold'
                    if game_id not in games:
                        ts = datetime.datetime.fromisoformat(row[1])
                        games[game_id] = {'min_ts': ts, 'winner': winner}
                    else:
                        games[game_id]['winner'] = winner

    # Filter to games with valid outcomes
    outcomes = {}
    for game_id, info in games.items():
        if info['winner'] is not None:
            outcomes[game_id] = (info['min_ts'], info['winner'])

    return outcomes


def _role_for_position(pos):
    """Return 'queen' for queen positions (1, 2) or 'drone' for workers."""
    return 'queen' if pos in (1, 2) else 'drone'


def compute_ratings(outcomes, usergame, game_cabinets, record_history=False):
    """Compute OpenSkill ratings chronologically with per-role composite keys.

    Each player gets independent ratings for queen and drone roles, keyed by
    (user_id, 'queen') or (user_id, 'drone'). Anonymous (non-logged-in) players
    are represented by shared per-cabinet ratings keyed ('anon', cabinet, role),
    ensuring teams are always 5v5 for zero-sum rating updates.

    Args:
        outcomes: {game_id: (timestamp, winning_team)}
        usergame: {game_id: {position_id: (user_id, name)}}
        game_cabinets: {game_id: cabinet_name}
        record_history: if True, populate history and cabinet_snapshots fields
            on the returned RatingResult.

    Returns:
        RatingResult with fields:
            ratings_by_game: {game_id: np.array(10, dtype=float32)} - pre-game mu
                values indexed by position 1-10 (array index 0 = position 1).
            player_ratings: {(user_id, role): PlackettLuceRating} - final ratings.
            user_names: {user_id: name} - most recent name for each user.
            cabinet_anon_ratings: {(cabinet, role): PlackettLuceRating} - final
                anonymous ratings per cabinet/role.
            history: list of per-game rating events (if record_history=True).
            cabinet_snapshots: periodic samples of cabinet anonymous ratings.
    """
    model = PlackettLuce()

    # Sort games chronologically
    sorted_games = sorted(outcomes.items(), key=lambda x: x[1][0])

    player_ratings = {}  # (user_id, role) -> PlackettLuceRating
    user_names = {}  # user_id -> name (most recent)
    cabinet_anon_ratings = {}  # (cabinet, role) -> PlackettLuceRating

    # Cabinet running averages for first-time logged-in player initialization.
    # Intentionally frequency-weighted (each player-game adds an entry), so
    # frequent players pull the average toward their skill level. This better
    # reflects the mu a new player will face at that cabinet.
    cabinet_mu_sum = {}  # (cabinet_name, role) -> sum of player-game mu values
    cabinet_mu_count = {}  # (cabinet_name, role) -> count of player-game entries

    ratings_by_game = {}
    history = [] if record_history else None
    cabinet_snapshots = [] if record_history else None

    for game_idx, (game_id, (ts, winner)) in enumerate(sorted_games):
        game_users = usergame.get(game_id, {})
        cabinet = game_cabinets.get(game_id, '')

        # Compute role-specific cabinet average mu for first-time logged-in players
        def cabinet_avg_for_role(role):
            key = (cabinet, role)
            if cabinet and key in cabinet_mu_count and cabinet_mu_count[key] > 0:
                return cabinet_mu_sum[key] / cabinet_mu_count[key]
            return 25.0

        # Record pre-game mu for all 10 positions
        pre_game_mu = np.zeros(10, dtype=np.float32)
        for pos in range(1, 11):
            role = _role_for_position(pos)
            if pos in game_users:
                user_id, name = game_users[pos]
                user_names[user_id] = name
                rating_key = (user_id, role)
                if rating_key in player_ratings:
                    pre_game_mu[pos - 1] = player_ratings[rating_key].mu
                else:
                    # First-time player in this role: use cabinet average
                    pre_game_mu[pos - 1] = cabinet_avg_for_role(role)
            else:
                # Anonymous player: use cabinet anonymous rating
                cab_key = (cabinet, role)
                if cab_key in cabinet_anon_ratings:
                    pre_game_mu[pos - 1] = cabinet_anon_ratings[cab_key].mu
                else:
                    pre_game_mu[pos - 1] = 25.0

        ratings_by_game[game_id] = pre_game_mu

        # Build teams for rating update — always 5v5
        # Blue team: even PIDs (2, 4, 6, 8, 10)
        # Gold team: odd PIDs (1, 3, 5, 7, 9)
        blue_positions = [2, 4, 6, 8, 10]
        gold_positions = [1, 3, 5, 7, 9]

        def build_team(positions):
            keys = []
            ratings = []
            for pos in positions:
                role = _role_for_position(pos)
                if pos in game_users:
                    user_id, _ = game_users[pos]
                    rating_key = (user_id, role)
                    if rating_key not in player_ratings:
                        player_ratings[rating_key] = model.rating(
                            mu=cabinet_avg_for_role(role))
                    keys.append(rating_key)
                    ratings.append(player_ratings[rating_key])
                else:
                    cab_key = (cabinet, role)
                    if cab_key not in cabinet_anon_ratings:
                        cabinet_anon_ratings[cab_key] = model.rating()
                    anon_r = cabinet_anon_ratings[cab_key]
                    keys.append(('anon', cabinet, role))
                    ratings.append(
                        model.rating(mu=anon_r.mu, sigma=anon_r.sigma))
            return keys, ratings

        blue_keys, blue_ratings = build_team(blue_positions)
        gold_keys, gold_ratings = build_team(gold_positions)

        # Always 5v5, always rate
        if winner == 'Blue':
            teams = [blue_ratings, gold_ratings]
            team_keys = [blue_keys, gold_keys]
        else:
            teams = [gold_ratings, blue_ratings]
            team_keys = [gold_keys, blue_keys]

        result = model.rate(teams=teams)

        # Separate results into player updates and anonymous updates
        anon_updates = {}  # (cabinet, role) -> [(mu, sigma)]
        for team_idx in range(2):
            for i, rating_key in enumerate(team_keys[team_idx]):
                new_rating = result[team_idx][i]
                if len(rating_key) == 3:
                    _, cab, role = rating_key
                    cab_key = (cab, role)
                    if cab_key not in anon_updates:
                        anon_updates[cab_key] = []
                    anon_updates[cab_key].append(
                        (new_rating.mu, new_rating.sigma))
                else:
                    player_ratings[rating_key] = new_rating

        # Average anonymous updates per (cabinet, role)
        for cab_key, updates in anon_updates.items():
            avg_mu = sum(u[0] for u in updates) / len(updates)
            avg_sigma = sum(u[1] for u in updates) / len(updates)
            cabinet_anon_ratings[cab_key] = model.rating(
                mu=avg_mu, sigma=avg_sigma)

        # Update cabinet running averages for logged-in players (per role)
        if cabinet:
            for rating_key in blue_keys + gold_keys:
                if len(rating_key) == 3:
                    continue
                _, role = rating_key
                cab_key = (cabinet, role)
                if cab_key not in cabinet_mu_sum:
                    cabinet_mu_sum[cab_key] = 0.0
                    cabinet_mu_count[cab_key] = 0
                cabinet_mu_sum[cab_key] += player_ratings[rating_key].mu
                cabinet_mu_count[cab_key] += 1

        if record_history:
            # Only record logged-in participants in game events
            participants = []
            all_positions = [(pos, 'blue') for pos in blue_positions] + \
                            [(pos, 'gold') for pos in gold_positions]
            for pos, team in all_positions:
                if pos in game_users:
                    user_id, name = game_users[pos]
                    role = _role_for_position(pos)
                    rating_key = (user_id, role)
                    participants.append({
                        'user_id': user_id,
                        'name': name,
                        'position': pos,
                        'role': role,
                        'team': team,
                        'mu_before': float(pre_game_mu[pos - 1]),
                        'mu_after': float(player_ratings[rating_key].mu),
                        'sigma': float(player_ratings[rating_key].sigma),
                    })
            history.append({
                'game_id': game_id,
                'timestamp': ts.isoformat(),
                'winner': winner,
                'participants': participants,
            })

            # Sample cabinet snapshots every 500 games
            if game_idx % 500 == 499 or game_idx == len(sorted_games) - 1:
                snap = {'timestamp': ts.isoformat(), 'ratings': {}}
                for (cab, role), r in cabinet_anon_ratings.items():
                    snap['ratings'][f'{cab}_{role}'] = {
                        'mu': round(r.mu, 2),
                        'sigma': round(r.sigma, 2),
                    }
                cabinet_snapshots.append(snap)

    return RatingResult(
        ratings_by_game=ratings_by_game,
        player_ratings=player_ratings,
        user_names=user_names,
        cabinet_anon_ratings=cabinet_anon_ratings,
        history=history,
        cabinet_snapshots=cabinet_snapshots,
    )


def evaluate_prediction(ratings_by_game, outcomes):
    """Compute correlation between team rating advantage and win rate.

    Returns (pearson_r, advantages, actuals) arrays.
    """
    advantages = []
    actuals = []
    for game_id, (ts, winner) in outcomes.items():
        if game_id not in ratings_by_game:
            continue
        mu = ratings_by_game[game_id]
        # Blue positions: indices 1,3,5,7,9 (positions 2,4,6,8,10)
        blue_avg = np.mean([mu[1], mu[3], mu[5], mu[7], mu[9]])
        gold_avg = np.mean([mu[0], mu[2], mu[4], mu[6], mu[8]])
        advantages.append(blue_avg - gold_avg)
        actuals.append(1.0 if winner == 'Blue' else 0.0)

    advantages = np.array(advantages)
    actuals = np.array(actuals)
    corr = np.corrcoef(advantages, actuals)[0, 1]
    return corr, advantages, actuals


def print_mu_over_time(history, player_ratings):
    """Print table and save chart of mean mu over time.

    Reveals rating deflation caused by mu leaking to untracked anonymous players.
    """
    player_mu = {}  # (user_id, role) -> current mu
    snapshots = []  # (game_index, date, n_players, mean_mu)

    sample_interval = max(1, len(history) // 80)  # ~80 rows

    for i, event in enumerate(history):
        for p in event['participants']:
            player_mu[(p['user_id'], p['role'])] = p['mu_after']

        if i % sample_interval == sample_interval - 1 or i == len(history) - 1:
            mus = list(player_mu.values())
            snapshots.append((
                i + 1,
                event['timestamp'][:10],
                len(mus),
                sum(mus) / len(mus),
            ))

    print('\nMean mu over time (rated games only):')
    print(f'  {"Game":>7s}  {"Date":>12s}  {"Players":>8s}  {"Mean mu":>8s}')
    # Print ~20 evenly-spaced rows for the table
    table_step = max(1, len(snapshots) // 20)
    for j, (game_idx, date, n, mean_mu) in enumerate(snapshots):
        if j % table_step == 0 or j == len(snapshots) - 1:
            print(f'  {game_idx:7d}  {date:>12s}  {n:8d}  {mean_mu:8.2f}')

    total_deflation = snapshots[-1][3] - snapshots[0][3]
    print(f'\n  Total deflation: {total_deflation:+.2f} '
          f'({snapshots[0][3]:.1f} -> {snapshots[-1][3]:.1f})')

    # Save chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(12, 5))

        dates = [s[1] for s in snapshots]
        mean_mus = [s[3] for s in snapshots]
        n_players = [s[2] for s in snapshots]

        color_mu = '#5ba3ec'
        ax1.plot(range(len(dates)), mean_mus, color=color_mu, linewidth=2)
        ax1.set_ylabel('Mean mu', color=color_mu)
        ax1.tick_params(axis='y', labelcolor=color_mu)
        ax1.set_xlabel('Date')

        # Show ~10 date labels
        tick_step = max(1, len(dates) // 10)
        tick_positions = list(range(0, len(dates), tick_step))
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([dates[i] for i in tick_positions],
                            rotation=45, ha='right', fontsize=8)

        ax2 = ax1.twinx()
        color_n = '#ffd700'
        ax2.plot(range(len(dates)), n_players, color=color_n, linewidth=1,
                 linestyle='--', alpha=0.7)
        ax2.set_ylabel('Tracked players', color=color_n)
        ax2.tick_params(axis='y', labelcolor=color_n)

        ax1.set_title('Rating Deflation: Mean Mu Over Time')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        import os
        chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'mu_over_time.png')
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        print(f'  Chart saved to {chart_path}')
    except ImportError:
        print('  (matplotlib not available, skipping chart)')


def print_validation(ratings_by_game, player_ratings, user_names, outcomes):
    """Print full validation stats."""
    # Top-rated players by queen and drone separately
    queen_ratings = {k: v for k, v in player_ratings.items() if k[1] == 'queen'}
    drone_ratings = {k: v for k, v in player_ratings.items() if k[1] == 'drone'}

    for role, role_ratings in [('queen', queen_ratings), ('drone', drone_ratings)]:
        top = sorted(role_ratings.items(),
                     key=lambda x: x[1].mu, reverse=True)[:20]
        print(f'\nTop 20 {role} players by mu:')
        for (user_id, _), rating in top:
            name = user_names.get(user_id, f'user_{user_id}')
            print(f'  {name:30s}  mu={rating.mu:.2f}  sigma={rating.sigma:.2f}')

    # Rating distribution
    all_mu = [r.mu for r in player_ratings.values()]
    print(f'\nRating distribution (mu) - all roles:')
    print(f'  mean={np.mean(all_mu):.2f}  std={np.std(all_mu):.2f}  '
          f'min={np.min(all_mu):.2f}  max={np.max(all_mu):.2f}')
    for role, role_ratings in [('queen', queen_ratings), ('drone', drone_ratings)]:
        mu_vals = [r.mu for r in role_ratings.values()]
        print(f'  {role}: mean={np.mean(mu_vals):.2f}  std={np.std(mu_vals):.2f}  '
              f'n={len(mu_vals)}')

    corr, advantages, actuals = evaluate_prediction(ratings_by_game, outcomes)

    # Bin by advantage quintiles
    print('\nTeam rating advantage vs win rate:')
    print(f'  {"Advantage bin":>20s}  {"Blue win%":>10s}  {"N games":>8s}')
    percentiles = [0, 20, 40, 60, 80, 100]
    for i in range(len(percentiles) - 1):
        lo = np.percentile(advantages, percentiles[i])
        hi = np.percentile(advantages, percentiles[i + 1])
        if i == len(percentiles) - 2:
            mask = (advantages >= lo) & (advantages <= hi)
        else:
            mask = (advantages >= lo) & (advantages < hi)
        if mask.sum() > 0:
            wr = actuals[mask].mean()
            print(f'  [{lo:+6.2f}, {hi:+6.2f}]  {wr:10.1%}  {mask.sum():8d}')

    print(f'\n  Pearson correlation: {corr:.4f}')


def load_data():
    """Load all input data (expensive I/O, do once)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print('Loading usergame.csv...')
    usergame = load_usergame(
        os.path.join(base_dir, 'unfiltered_partitioned/usergame.csv'))
    print(f'  {len(usergame)} games with logged-in users')

    print('Loading game.csv...')
    game_cabinets = load_game_cabinets(
        os.path.join(base_dir, 'unfiltered_partitioned/game.csv'))
    print(f'  {len(game_cabinets)} games with cabinet info')

    print('Extracting game outcomes from logged_in_games/...')
    outcomes = extract_game_outcomes(
        os.path.join(base_dir, 'logged_in_games/gameevents_*.csv.gz'))
    print(f'  {len(outcomes)} games with valid outcomes')

    return usergame, game_cabinets, outcomes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute player skill ratings')
    parser.add_argument('--output', type=str, default='ratings_queen_drone.pkl',
                        help='Output pickle filename (default: ratings_queen_drone.pkl)')
    args = parser.parse_args()

    usergame, game_cabinets, outcomes = load_data()

    print('Computing ratings...')
    result = compute_ratings(outcomes, usergame, game_cabinets, record_history=True)
    print(f'  {len(result.ratings_by_game)} games rated')
    print(f'  {len(result.player_ratings)} unique players')

    # Save ratings
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, args.output)
    with open(output_path, 'wb') as f:
        pickle.dump(result.ratings_by_game, f)
    print(f'Saved ratings to {output_path}')

    print('\n=== Rating Deflation ===')
    print_mu_over_time(result.history, result.player_ratings)

    print('\n=== Validation ===')
    print_validation(result.ratings_by_game, result.player_ratings,
                     result.user_names, outcomes)


if __name__ == '__main__':
    main()
