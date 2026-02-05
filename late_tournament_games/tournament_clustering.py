"""
Tournament Clustering Library

Clusters tournament_match_ids (individual match series) into actual tournaments
using heuristics:
1. Same cabinet + within ±1 day → same tournament
2. Same team name + within ±1 day (across any cabinet) → same tournament
"""

import json
import pandas as pd
from datetime import timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, FrozenSet


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self) -> Dict:
        """Return dict mapping root -> list of items in cluster."""
        clusters = {}
        for item in self.parent:
            root = self.find(item)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(item)
        return clusters


@dataclass
class TournamentMetadata:
    """Metadata for a clustered tournament."""
    tournament_id: int
    start_time: str  # ISO format
    end_time: str    # ISO format
    num_games: int
    num_matches: int  # Number of match series
    num_teams: int
    teams: List[str]
    primary_cabinet: str
    cabinets: List[str]
    is_weekend: bool
    match_ids: List[float]  # tournament_match_ids in this tournament


def extract_match_info(tournament_games: pd.DataFrame) -> pd.DataFrame:
    """Extract per-match info needed for clustering."""
    match_info = tournament_games.groupby('tournament_match_id').agg(
        cabinet_id=('cabinet_id', 'first'),
        cabinet_name=('cabinet_name', 'first'),
        min_date=('start_time', 'min'),
        max_date=('start_time', 'max'),
    ).reset_index()

    def get_teams(group) -> FrozenSet[str]:
        teams = set(group['blue_team'].dropna()) | set(group['gold_team'].dropna())
        return frozenset(teams)

    match_teams = tournament_games.groupby('tournament_match_id').apply(
        get_teams
    ).reset_index(name='teams')

    match_info = match_info.merge(match_teams, on='tournament_match_id')
    match_info['date'] = match_info['min_date'].dt.date

    return match_info


def cluster_matches(match_info: pd.DataFrame) -> Dict[float, int]:
    """
    Cluster tournament matches into tournaments.

    Returns:
        Dict mapping tournament_match_id -> tournament_id
    """
    uf = UnionFind(match_info['tournament_match_id'].tolist())

    # Rule 1: Same cabinet + within ±1 day → merge
    cabinet_date_index = defaultdict(list)
    for _, row in match_info.iterrows():
        cabinet_date_index[(row['cabinet_id'], row['date'])].append(row['tournament_match_id'])

    for _, row in match_info.iterrows():
        match_id = row['tournament_match_id']
        cabinet = row['cabinet_id']
        date = row['date']

        for delta in [timedelta(days=0), timedelta(days=1), timedelta(days=-1)]:
            check_date = date + delta
            for other_id in cabinet_date_index.get((cabinet, check_date), []):
                if other_id != match_id:
                    uf.union(match_id, other_id)

    # Rule 2: Same team name + within ±1 day → merge
    team_date_index = defaultdict(list)
    for _, row in match_info.iterrows():
        for team in row['teams']:
            team_date_index[(team, row['date'])].append(row['tournament_match_id'])

    for _, row in match_info.iterrows():
        match_id = row['tournament_match_id']
        date = row['date']

        for team in row['teams']:
            for delta in [timedelta(days=0), timedelta(days=1), timedelta(days=-1)]:
                check_date = date + delta
                for other_id in team_date_index.get((team, check_date), []):
                    if other_id != match_id:
                        uf.union(match_id, other_id)

    # Build mapping from match_id to cluster_id
    clusters = uf.get_clusters()
    match_to_tournament = {}
    for tournament_id, (root, match_ids) in enumerate(clusters.items()):
        for match_id in match_ids:
            match_to_tournament[match_id] = tournament_id

    return match_to_tournament


def compute_tournament_metadata(
    tournament_games: pd.DataFrame,
    match_to_tournament: Dict[float, int]
) -> List[TournamentMetadata]:
    """Compute metadata for each clustered tournament."""

    # Add tournament_id to games
    games = tournament_games.copy()
    games['tournament_id'] = games['tournament_match_id'].map(match_to_tournament)

    # Ensure is_weekend column exists
    if 'is_weekend' not in games.columns:
        games['day_of_week'] = games['start_time'].dt.dayofweek
        games['is_weekend'] = games['day_of_week'].isin([5, 6])

    tournaments = []

    for tournament_id, group in games.groupby('tournament_id'):
        # Get all teams
        teams = set(group['blue_team'].dropna()) | set(group['gold_team'].dropna())

        # Get all cabinets
        cabinets = group['cabinet_name'].dropna().unique().tolist()

        # Primary cabinet (most common)
        primary_cabinet = group['cabinet_name'].mode()
        primary_cabinet = primary_cabinet.iloc[0] if len(primary_cabinet) > 0 else None

        # Weekend determination (majority of games)
        is_weekend = group['is_weekend'].sum() > (len(group) / 2)

        # Match IDs in this tournament
        match_ids = group['tournament_match_id'].unique().tolist()

        metadata = TournamentMetadata(
            tournament_id=int(tournament_id),
            start_time=group['start_time'].min().isoformat(),
            end_time=group['start_time'].max().isoformat(),
            num_games=len(group),
            num_matches=group['tournament_match_id'].nunique(),
            num_teams=len(teams),
            teams=sorted(teams),
            primary_cabinet=primary_cabinet,
            cabinets=sorted(cabinets),
            is_weekend=is_weekend,
            match_ids=match_ids,
        )
        tournaments.append(metadata)

    return tournaments


def cluster_tournaments(games_df: pd.DataFrame) -> tuple:
    """
    Main entry point: cluster tournament matches into tournaments.

    Args:
        games_df: DataFrame with game data (must have tournament_match_id column)

    Returns:
        Tuple of (match_to_tournament mapping, list of TournamentMetadata)
    """
    # Filter to tournament games
    tournament_games = games_df[
        games_df['tournament_match_id'].notna() &
        (games_df['tournament_match_id'] != '')
    ].copy()

    # Parse timestamps
    tournament_games['start_time'] = pd.to_datetime(
        tournament_games['start_time'], format='mixed'
    )

    # Extract match info and cluster
    match_info = extract_match_info(tournament_games)
    match_to_tournament = cluster_matches(match_info)

    # Compute tournament metadata
    tournaments = compute_tournament_metadata(tournament_games, match_to_tournament)

    return match_to_tournament, tournaments


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_clustering(
    match_to_tournament: Dict[float, int],
    tournaments: List[TournamentMetadata],
    output_path: str = "tournament_clustering.json"
):
    """Save clustering results to JSON file."""
    data = {
        "match_to_tournament": {str(k): v for k, v in match_to_tournament.items()},
        "tournaments": [_convert_to_native(asdict(t)) for t in tournaments],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved clustering to {output_path}")
    print(f"  - {len(match_to_tournament):,} match mappings")
    print(f"  - {len(tournaments):,} tournaments")


def load_clustering(input_path: str = "tournament_clustering.json") -> tuple:
    """Load clustering results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    match_to_tournament = {float(k): v for k, v in data["match_to_tournament"].items()}
    tournaments = [TournamentMetadata(**t) for t in data["tournaments"]]

    return match_to_tournament, tournaments


def main():
    """Run clustering on game.csv and save results."""
    import argparse

    parser = argparse.ArgumentParser(description="Cluster tournament matches into tournaments")
    parser.add_argument(
        "--input", "-i",
        default="new_data_partitioned/game.csv",
        help="Path to game.csv file"
    )
    parser.add_argument(
        "--output", "-o",
        default="tournament_clustering.json",
        help="Output path for clustering results"
    )
    args = parser.parse_args()

    print(f"Loading games from {args.input}...")
    games_df = pd.read_csv(args.input)
    print(f"  Total games: {len(games_df):,}")

    print("\nClustering tournament matches...")
    match_to_tournament, tournaments = cluster_tournaments(games_df)

    # Print summary
    print(f"\n=== Clustering Summary ===")
    print(f"Match series: {len(match_to_tournament):,}")
    print(f"Tournaments: {len(tournaments):,}")

    if tournaments:
        games_per = [t.num_games for t in tournaments]
        teams_per = [t.num_teams for t in tournaments]
        print(f"\nGames per tournament: median={sorted(games_per)[len(games_per)//2]}, max={max(games_per)}")
        print(f"Teams per tournament: median={sorted(teams_per)[len(teams_per)//2]}, max={max(teams_per)}")

        # Major tournaments
        major = [t for t in tournaments if t.num_teams >= 12 and t.num_games >= 20 and t.is_weekend]
        print(f"Major tournaments (>=12 teams, >=20 games, weekend): {len(major)}")

    print()
    save_clustering(match_to_tournament, tournaments, args.output)


if __name__ == "__main__":
    main()
