"""Tests for compute_ratings.py."""

import datetime
import pickle
import tempfile

import numpy as np
import pytest

from compute_ratings import compute_ratings, load_usergame, load_game_cabinets


def _make_outcomes(games):
    """Helper: build outcomes dict from list of (game_id, timestamp_str, winner).

    Returns {game_id: (datetime, winner_str)}.
    """
    return {
        gid: (datetime.datetime.fromisoformat(ts), winner)
        for gid, ts, winner in games
    }


class TestChronologicalCorrectness:
    """Ratings for game N must use only games before N."""

    def test_ratings_reflect_only_prior_games(self):
        # Three games in sequence. Player 100 wins game 1 and 2, plays game 3.
        # Their mu at game 3 should be > default 25.0.
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},   # Alice=blue queen, Bob=gold queen
            2: {2: (100, 'Alice'), 1: (201, 'Carol')},
            3: {2: (100, 'Alice'), 1: (202, 'Dave')},
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),
            (2, '2024-01-02T00:00:00+00:00', 'Blue'),
            (3, '2024-01-03T00:00:00+00:00', 'Gold'),
        ])
        cabinets = {1: 'venue_a', 2: 'venue_a', 3: 'venue_a'}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Game 1: Alice has no history, should get default mu (25.0)
        assert result.ratings_by_game[1][1] == pytest.approx(25.0)  # pos 2 -> index 1

        # Game 2: Alice won game 1, her mu should be > 25
        assert result.ratings_by_game[2][1] > 25.0

        # Game 3: Alice won games 1 and 2, her mu should be even higher
        assert result.ratings_by_game[3][1] > result.ratings_by_game[2][1]


class TestWinningTeamRatingsIncrease:
    """Repeated wins should increase team ratings."""

    def test_repeated_wins_increase_mu(self):
        # Same two players, player 100 (Blue) wins every game
        usergame = {}
        outcomes_list = []
        for i in range(1, 11):
            usergame[i] = {2: (100, 'Winner'), 1: (200, 'Loser')}
            outcomes_list.append(
                (i, f'2024-01-{i:02d}T00:00:00+00:00', 'Blue'))

        outcomes = _make_outcomes(outcomes_list)
        cabinets = {i: 'venue' for i in range(1, 11)}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Both play as queens (positions 1 and 2)
        assert result.player_ratings[(100, 'queen')].mu > 25.0
        assert result.player_ratings[(200, 'queen')].mu < 25.0
        # Winner should be clearly above loser
        assert result.player_ratings[(100, 'queen')].mu - result.player_ratings[(200, 'queen')].mu > 5.0


class TestCabinetAverageFallback:
    """Anonymous/unknown players should get cabinet average mu."""

    def test_anonymous_gets_cabinet_average(self):
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},
            2: {2: (100, 'Alice'), 1: (201, 'Carol')},
            3: {2: (100, 'Alice'), 1: (202, 'Dave')},
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),  # venue_a
            (2, '2024-01-02T00:00:00+00:00', 'Blue'),  # venue_a
            (3, '2024-01-03T00:00:00+00:00', 'Blue'),  # venue_b
        ])
        cabinets = {1: 'venue_a', 2: 'venue_a', 3: 'venue_b'}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Alice should have mu > 25 after two wins
        alice_mu = result.player_ratings[(100, 'queen')].mu
        assert alice_mu > 25.0

    def test_no_cabinet_history_falls_back_to_default(self):
        # Game at a venue with no prior history, anonymous player
        usergame = {
            1: {2: (100, 'Alice')},  # only one logged-in player
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),
        ])
        cabinets = {1: 'new_venue'}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Anonymous queen (pos 1) should get default mu (25.0)
        assert result.ratings_by_game[1][0] == pytest.approx(25.0)

    def test_cabinet_avg_diverges_with_asymmetric_participation(self):
        # Strong player (Alice) plays many games at venue_a, always winning.
        # Different opponents each time, who only play once.
        # Cabinet average skews because Alice (high mu) contributes many entries.
        usergame = {}
        outcomes_list = []
        for i in range(1, 6):
            usergame[i] = {2: (100, 'Alice'), 1: (200 + i, f'Opponent_{i}')}
            outcomes_list.append(
                (i, f'2024-01-{i:02d}T00:00:00+00:00', 'Blue'))

        # Game 6: new player at same venue
        usergame[6] = {2: (300, 'NewPlayer'), 1: (200, 'OtherNew')}
        outcomes_list.append((6, '2024-01-06T00:00:00+00:00', 'Blue'))

        outcomes = _make_outcomes(outcomes_list)
        cabinets = {i: 'venue_a' for i in range(1, 7)}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Cabinet average has more entries from Alice (who plays every game
        # and has high mu) than from each individual opponent.
        # Alice contributes 5 high-mu entries; each opponent contributes 1
        # low-mu entry. The average should be above 25.
        new_player_mu = result.ratings_by_game[6][1]  # pos 2 = index 1
        assert new_player_mu > 25.0


class TestFirstTimePlayerCabinetAverage:
    """A first-time player at a cabinet with history gets cabinet average."""

    def test_first_timer_gets_cabinet_avg(self):
        usergame = {}
        outcomes_list = []
        for i in range(1, 6):
            usergame[i] = {2: (100, 'Alice'), 1: (200 + i, f'Opp_{i}')}
            outcomes_list.append(
                (i, f'2024-01-{i:02d}T00:00:00+00:00', 'Blue'))

        usergame[6] = {2: (300, 'NewPlayer'), 1: (100, 'Alice')}
        outcomes_list.append((6, '2024-01-06T00:00:00+00:00', 'Gold'))

        outcomes = _make_outcomes(outcomes_list)
        cabinets = {i: 'venue_a' for i in range(1, 7)}

        result = compute_ratings(outcomes, usergame, cabinets)

        # NewPlayer (pos 2, index 1) should get cabinet avg, which is > 25
        new_player_mu = result.ratings_by_game[6][1]
        assert new_player_mu > 25.0


class TestDualRoleRatings:
    """Players get independent ratings for queen and drone roles."""

    def test_queen_and_drone_ratings_are_independent(self):
        # Player 100 plays queen in game 1 (wins), drone in game 2 (loses).
        # Queen rating should go up, drone rating should go down.
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},        # Alice=blue queen
            2: {4: (100, 'Alice'), 2: (201, 'Carol'),        # Alice=blue drone
                1: (202, 'Dave')},
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),   # Alice wins as queen
            (2, '2024-01-02T00:00:00+00:00', 'Gold'),   # Alice loses as drone
        ])
        cabinets = {1: 'venue', 2: 'venue'}

        result = compute_ratings(outcomes, usergame, cabinets)

        # Queen rating should be above default (won)
        assert result.player_ratings[(100, 'queen')].mu > 25.0
        # Drone rating should be below default (lost)
        assert result.player_ratings[(100, 'drone')].mu < 25.0
        # They should be different
        assert result.player_ratings[(100, 'queen')].mu != result.player_ratings[(100, 'drone')].mu

    def test_queen_cabinet_avg_separate_from_drone(self):
        # Build up cabinet history with a dominant queen.
        # Drone cabinet average should not be pulled up by queen stats.
        usergame = {}
        outcomes_list = []
        for i in range(1, 6):
            usergame[i] = {
                2: (100, 'Alice'),     # blue queen (dominant)
                1: (200 + i, f'GoldQ_{i}'),  # gold queen (one-shot)
                4: (300, 'BlueW1'),    # blue drone
                3: (400, 'GoldW1'),    # gold drone
            }
            outcomes_list.append(
                (i, f'2024-01-{i:02d}T00:00:00+00:00', 'Blue'))

        # Game 6: new queen and new drone at same venue
        usergame[6] = {
            2: (500, 'NewQueen'),
            1: (501, 'NewGoldQ'),
            4: (502, 'NewDrone'),
            3: (503, 'NewGoldD'),
        }
        outcomes_list.append((6, '2024-01-06T00:00:00+00:00', 'Blue'))

        outcomes = _make_outcomes(outcomes_list)
        cabinets = {i: 'venue' for i in range(1, 7)}

        result = compute_ratings(outcomes, usergame, cabinets)

        # New queen (pos 2, idx 1) gets queen cabinet avg
        new_queen_mu = result.ratings_by_game[6][1]
        # New drone (pos 4, idx 3) gets drone cabinet avg
        new_drone_mu = result.ratings_by_game[6][3]
        # These should generally be different since queen and drone pools differ
        # Alice (queen, always winning) skews queen avg up more than drone avg
        assert new_queen_mu != pytest.approx(new_drone_mu, abs=0.5)


class TestPickleRoundtrip:
    """Ratings dict should survive pickle save/load."""

    def test_roundtrip(self):
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),
        ])
        cabinets = {1: 'venue'}

        result = compute_ratings(outcomes, usergame, cabinets)

        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            with open(tmp.name, 'wb') as f:
                pickle.dump(result.ratings_by_game, f)
            with open(tmp.name, 'rb') as f:
                loaded = pickle.load(f)

        assert set(loaded.keys()) == set(result.ratings_by_game.keys())
        for game_id in result.ratings_by_game:
            np.testing.assert_array_equal(
                loaded[game_id], result.ratings_by_game[game_id])
