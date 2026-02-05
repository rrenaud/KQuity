#!/usr/bin/env python3
"""
Extract game events for the last 5 matches of tournaments with >15 teams.

This creates a high-quality tournament data subset representing late-stage
tournament games where teams are likely playing at their best.
"""

import csv
import gzip
import os
from collections import defaultdict
from typing import Set, Dict, List
import pandas as pd

from tournament_clustering import load_clustering, TournamentMetadata


def get_late_tournament_game_ids(
    games_df: pd.DataFrame,
    tournaments: List[TournamentMetadata],
    min_teams: int = 15,
    last_n_matches: int = 5,
) -> Set[int]:
    """
    Get game IDs from the last N matches of tournaments with >min_teams teams.

    Args:
        games_df: DataFrame with game data
        tournaments: List of TournamentMetadata from clustering
        min_teams: Minimum number of teams for a tournament to qualify
        last_n_matches: Number of last matches to include per tournament

    Returns:
        Set of game IDs for late tournament games
    """
    # Filter to tournaments with enough teams
    qualifying_tournaments = [t for t in tournaments if t.num_teams > min_teams]
    print(f"Tournaments with >{min_teams} teams: {len(qualifying_tournaments)}")

    # For each qualifying tournament, find the last N matches by time
    late_game_ids = set()

    for tournament in qualifying_tournaments:
        # Get all games for this tournament's matches
        tournament_games = games_df[
            games_df['tournament_match_id'].isin(tournament.match_ids)
        ].copy()

        if len(tournament_games) == 0:
            continue

        # Parse timestamps
        tournament_games['start_time'] = pd.to_datetime(
            tournament_games['start_time'], format='mixed'
        )

        # Get the last start time for each match (to rank matches by when they ended)
        match_end_times = tournament_games.groupby('tournament_match_id')['start_time'].max()

        # Sort matches by end time and take the last N
        sorted_matches = match_end_times.sort_values(ascending=False)
        last_matches = sorted_matches.head(last_n_matches).index.tolist()

        # Get all game IDs from those matches
        late_games = tournament_games[
            tournament_games['tournament_match_id'].isin(last_matches)
        ]['id'].tolist()

        late_game_ids.update(late_games)

    return late_game_ids


def extract_game_events(
    input_dir: str,
    output_path: str,
    game_ids: Set[int],
):
    """
    Extract game events for specific game IDs from partitioned files.

    Args:
        input_dir: Directory with partitioned CSV files
        output_path: Output path for filtered events (gzipped)
        game_ids: Set of game IDs to extract
    """
    print(f"Extracting events for {len(game_ids):,} games...")

    # Find all partition files
    partition_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.csv.gz')
    ])
    print(f"Found {len(partition_files)} partition files")

    total_events = 0
    games_found = set()
    fieldnames = None

    with gzip.open(output_path, 'wt', newline='') as out_f:
        writer = None

        for partition_file in partition_files:
            partition_path = os.path.join(input_dir, partition_file)
            print(f"  Processing {partition_file}...")

            with gzip.open(partition_path, 'rt') as in_f:
                reader = csv.DictReader(in_f)

                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()

                for row in reader:
                    game_id = int(row['game_id'])
                    if game_id in game_ids:
                        writer.writerow(row)
                        total_events += 1
                        games_found.add(game_id)

    print(f"Extracted {total_events:,} events from {len(games_found):,} games")
    missing = game_ids - games_found
    if missing:
        print(f"Warning: {len(missing)} game IDs not found in event data")

    return len(games_found), total_events


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract late tournament game events"
    )
    parser.add_argument(
        "--game-csv", "-g",
        default="new_data_partitioned/game.csv",
        help="Path to game.csv"
    )
    parser.add_argument(
        "--clustering", "-c",
        default="tournament_clustering.json",
        help="Path to tournament clustering JSON"
    )
    parser.add_argument(
        "--events-dir", "-e",
        default="new_data_partitioned",
        help="Directory with partitioned event files"
    )
    parser.add_argument(
        "--output", "-o",
        default="late_tournament_game_events.csv.gz",
        help="Output path for filtered events"
    )
    parser.add_argument(
        "--min-teams", "-t",
        type=int, default=15,
        help="Minimum teams per tournament (default: 15)"
    )
    parser.add_argument(
        "--last-matches", "-n",
        type=int, default=5,
        help="Number of last matches per tournament (default: 5)"
    )
    args = parser.parse_args()

    # Load data
    print("Loading game data...")
    games_df = pd.read_csv(args.game_csv)
    print(f"  Total games: {len(games_df):,}")

    print("\nLoading tournament clustering...")
    match_to_tournament, tournaments = load_clustering(args.clustering)
    print(f"  Tournaments: {len(tournaments):,}")

    # Get late tournament game IDs
    print(f"\nFinding late tournament games (>{args.min_teams} teams, last {args.last_matches} matches)...")
    late_game_ids = get_late_tournament_game_ids(
        games_df, tournaments,
        min_teams=args.min_teams,
        last_n_matches=args.last_matches
    )
    print(f"  Late tournament games: {len(late_game_ids):,}")

    # Save game IDs for reference
    game_ids_path = args.output.replace('.csv.gz', '_game_ids.txt')
    with open(game_ids_path, 'w') as f:
        for gid in sorted(late_game_ids):
            f.write(f"{gid}\n")
    print(f"  Game IDs saved to {game_ids_path}")

    # Extract events
    print(f"\nExtracting events to {args.output}...")
    num_games, num_events = extract_game_events(
        args.events_dir, args.output, late_game_ids
    )

    # Summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    qualifying_tournaments = [t for t in tournaments if t.num_teams > args.min_teams]
    print(f"Tournaments with >{args.min_teams} teams: {len(qualifying_tournaments)}")
    print(f"Last {args.last_matches} matches per tournament")
    print(f"Total late tournament games: {len(late_game_ids):,}")
    print(f"Games found in events: {num_games:,}")
    print(f"Total events extracted: {num_events:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
