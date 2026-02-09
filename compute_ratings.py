"""Compute per-player OpenSkill ratings from game history.

Processes logged_in_games/ event CSVs chronologically, updating Plackett-Luce
ratings for logged-in players. Outputs ratings.pkl: {game_id: np.array(10)}
containing pre-game mu values for each position (1-10).
"""

import csv
import datetime
import glob
import gzip
import os
import pickle

import numpy as np
from openskill.models import PlackettLuce


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


def compute_ratings(outcomes, usergame, game_cabinets, anonymous_discount=6.0):
    """Compute OpenSkill ratings chronologically.

    Args:
        outcomes: {game_id: (timestamp, winning_team)}
        usergame: {game_id: {position_id: (user_id, name)}}
        game_cabinets: {game_id: cabinet_name}
        anonymous_discount: mu penalty for anonymous (non-logged-in) players
            relative to cabinet average. Logged-in players who create accounts
            are likely stronger than anonymous players.

    Returns:
        ratings_by_game: {game_id: np.array(10, dtype=float32)} - pre-game mu
            values indexed by position 1-10 (array index 0 = position 1).
        player_ratings: {user_id: PlackettLuceRating} - final ratings.
        user_names: {user_id: name} - most recent name for each user.
    """
    model = PlackettLuce()

    # Sort games chronologically
    sorted_games = sorted(outcomes.items(), key=lambda x: x[1][0])

    player_ratings = {}  # user_id -> PlackettLuceRating
    user_names = {}  # user_id -> name (most recent)

    # Cabinet running averages for fallback
    cabinet_mu_sum = {}  # cabinet_name -> sum of known mu values
    cabinet_mu_count = {}  # cabinet_name -> count

    ratings_by_game = {}

    for game_id, (ts, winner) in sorted_games:
        game_users = usergame.get(game_id, {})
        cabinet = game_cabinets.get(game_id, '')

        # Compute cabinet average mu for fallback
        if cabinet and cabinet in cabinet_mu_count and cabinet_mu_count[cabinet] > 0:
            cabinet_avg = cabinet_mu_sum[cabinet] / cabinet_mu_count[cabinet]
        else:
            cabinet_avg = 25.0

        # Record pre-game mu for all 10 positions
        pre_game_mu = np.zeros(10, dtype=np.float32)
        for pos in range(1, 11):
            if pos in game_users:
                user_id, name = game_users[pos]
                user_names[user_id] = name
                if user_id in player_ratings:
                    pre_game_mu[pos - 1] = player_ratings[user_id].mu
                else:
                    # First-time player: use cabinet average
                    pre_game_mu[pos - 1] = cabinet_avg
            else:
                # Anonymous player: cabinet average minus discount
                # TODO: replace flat discount with per-cabinet-position EMA
                # of implied anonymous skill from game outcomes.
                pre_game_mu[pos - 1] = cabinet_avg - anonymous_discount

        ratings_by_game[game_id] = pre_game_mu

        # Build teams for rating update
        # Blue team: even PIDs (2, 4, 6, 8, 10)
        # Gold team: odd PIDs (1, 3, 5, 7, 9)
        blue_positions = [2, 4, 6, 8, 10]
        gold_positions = [1, 3, 5, 7, 9]

        blue_user_ids = []
        blue_ratings = []
        gold_user_ids = []
        gold_ratings = []

        for pos in blue_positions:
            if pos in game_users:
                user_id, _ = game_users[pos]
                if user_id not in player_ratings:
                    player_ratings[user_id] = model.rating(mu=cabinet_avg)
                blue_user_ids.append(user_id)
                blue_ratings.append(player_ratings[user_id])

        for pos in gold_positions:
            if pos in game_users:
                user_id, _ = game_users[pos]
                if user_id not in player_ratings:
                    player_ratings[user_id] = model.rating(mu=cabinet_avg)
                gold_user_ids.append(user_id)
                gold_ratings.append(player_ratings[user_id])

        # Only update if at least one logged-in player per team
        if blue_ratings and gold_ratings:
            if winner == 'Blue':
                teams = [blue_ratings, gold_ratings]
                team_user_ids = [blue_user_ids, gold_user_ids]
            else:
                teams = [gold_ratings, blue_ratings]
                team_user_ids = [gold_user_ids, blue_user_ids]

            result = model.rate(teams=teams)

            # Update player ratings
            for team_idx in range(2):
                for i, user_id in enumerate(team_user_ids[team_idx]):
                    player_ratings[user_id] = result[team_idx][i]

            # Update cabinet running average
            if cabinet:
                if cabinet not in cabinet_mu_sum:
                    cabinet_mu_sum[cabinet] = 0.0
                    cabinet_mu_count[cabinet] = 0
                for user_id in blue_user_ids + gold_user_ids:
                    cabinet_mu_sum[cabinet] += player_ratings[user_id].mu
                    cabinet_mu_count[cabinet] += 1

    return ratings_by_game, player_ratings, user_names


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


def print_validation(ratings_by_game, player_ratings, user_names, outcomes):
    """Print full validation stats."""
    # Top-rated players
    top_players = sorted(player_ratings.items(),
                         key=lambda x: x[1].mu, reverse=True)[:20]
    print('\nTop 20 players by mu:')
    for user_id, rating in top_players:
        name = user_names.get(user_id, f'user_{user_id}')
        print(f'  {name:30s}  mu={rating.mu:.2f}  sigma={rating.sigma:.2f}')

    # Rating distribution
    all_mu = [r.mu for r in player_ratings.values()]
    print(f'\nRating distribution (mu):')
    print(f'  mean={np.mean(all_mu):.2f}  std={np.std(all_mu):.2f}  '
          f'min={np.min(all_mu):.2f}  max={np.max(all_mu):.2f}')

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
    os.chdir(base_dir)

    print('Loading usergame.csv...')
    usergame = load_usergame()
    print(f'  {len(usergame)} games with logged-in users')

    print('Loading game.csv...')
    game_cabinets = load_game_cabinets()
    print(f'  {len(game_cabinets)} games with cabinet info')

    print('Extracting game outcomes from logged_in_games/...')
    outcomes = extract_game_outcomes()
    print(f'  {len(outcomes)} games with valid outcomes')

    return usergame, game_cabinets, outcomes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute player skill ratings')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep anonymous_discount values instead of single run')
    parser.add_argument('--anonymous-discount', type=float, default=6.0,
                        help='Mu penalty for anonymous players (default: 6.0)')
    args = parser.parse_args()

    usergame, game_cabinets, outcomes = load_data()

    if args.sweep:
        print('\n=== Discount Sweep ===')
        print(f'  {"discount":>10s}  {"correlation":>12s}')
        best_corr = -1.0
        best_discount = 0.0
        for discount in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
            ratings_by_game, _, _ = compute_ratings(
                outcomes, usergame, game_cabinets,
                anonymous_discount=float(discount))
            corr, _, _ = evaluate_prediction(ratings_by_game, outcomes)
            marker = ''
            if corr > best_corr:
                best_corr = corr
                best_discount = discount
                marker = '  <-- best'
            print(f'  {discount:10.1f}  {corr:12.4f}{marker}')

        print(f'\nRecomputing with best discount={best_discount:.1f}...')
        args.anonymous_discount = best_discount

    print('Computing ratings...')
    ratings_by_game, player_ratings, user_names = compute_ratings(
        outcomes, usergame, game_cabinets,
        anonymous_discount=args.anonymous_discount)
    print(f'  {len(ratings_by_game)} games rated')
    print(f'  {len(player_ratings)} unique players')

    # Save ratings
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'ratings.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ratings_by_game, f)
    print(f'Saved ratings to {output_path}')

    print('\n=== Validation ===')
    print_validation(ratings_by_game, player_ratings, user_names, outcomes)


if __name__ == '__main__':
    main()
