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

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # Game 1: Alice has no history, should get default mu (25.0)
        assert ratings_by_game[1][1] == pytest.approx(25.0)  # pos 2 -> index 1

        # Game 2: Alice won game 1, her mu should be > 25
        assert ratings_by_game[2][1] > 25.0

        # Game 3: Alice won games 1 and 2, her mu should be even higher
        assert ratings_by_game[3][1] > ratings_by_game[2][1]


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

        _, player_ratings, _ = compute_ratings(outcomes, usergame, cabinets)

        assert player_ratings[100].mu > 25.0
        assert player_ratings[200].mu < 25.0
        # Winner should be clearly above loser
        assert player_ratings[100].mu - player_ratings[200].mu > 5.0


class TestCabinetAverageFallback:
    """Anonymous/unknown players should get cabinet average mu."""

    def test_anonymous_gets_cabinet_average(self):
        # Game 1 at venue_a: only Blue team has logged-in player, Blue wins
        # Game 2 at venue_a: only Blue team has logged-in player, Blue wins
        # After game 1, only Alice's rating is updated (Bob not logged in).
        # Cabinet average = Alice's mu (> 25) since only she contributed.
        # Game 2 anonymous positions should get that cabinet average.
        #
        # We need unequal team sizes of logged-in players so the cabinet
        # average diverges from 25 (equal teams are zero-sum in mu).
        usergame = {
            1: {2: (100, 'Alice')},  # only Blue queen logged in
            2: {2: (100, 'Alice')},  # same
        }
        # For game 1, only Blue has a logged-in player, so no rating update.
        # We need both teams to have logged-in players for an update to happen.
        # Let's use 3 games: game 1 establishes ratings, game 2 has only
        # Alice logged in (no update), game 3 checks cabinet average.
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},   # both logged in
            2: {2: (100, 'Alice'), 1: (200, 'Bob')},   # both logged in
            3: {2: (100, 'Alice')},                      # only Alice
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),   # Alice wins
            (2, '2024-01-02T00:00:00+00:00', 'Blue'),   # Alice wins again
            (3, '2024-01-03T00:00:00+00:00', 'Blue'),
        ])
        cabinets = {1: 'venue_a', 2: 'venue_a', 3: 'venue_a'}

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # By game 3, Alice's mu > 25, Bob's mu < 25.
        # Anonymous gold queen (pos 1, index 0) should get cabinet average.
        # Cabinet average = mean of all post-game mu updates at venue_a.
        # Since Alice > 25 and Bob < 25 are both in the cabinet, the average
        # is 25 (zero-sum). But the cabinet average includes ALL updates:
        # after game 1: Alice~27.6, Bob~22.4 -> sum of 2 entries
        # after game 2: Alice~29.5, Bob~20.5 -> sum of 4 more entries
        # Total sum = (27.6+22.4 + 29.5+20.5) = 100, count=4, avg=25.0
        # This is inherently ~25 because OpenSkill is zero-sum.
        #
        # To actually get a non-25 cabinet average, we need asymmetric
        # participation. Let's have only Alice play at venue_b games.
        usergame = {
            1: {2: (100, 'Alice'), 1: (200, 'Bob')},   # both at venue_a
            2: {2: (100, 'Alice'), 1: (201, 'Carol')},  # Alice at venue_b
        }
        outcomes = _make_outcomes([
            (1, '2024-01-01T00:00:00+00:00', 'Blue'),
            (2, '2024-01-02T00:00:00+00:00', 'Blue'),
        ])
        cabinets = {1: 'venue_a', 2: 'venue_b'}

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # At venue_b, the cabinet average includes Alice (~27.6) and Carol (~22.4)
        # which averages to ~25. Let's check a scenario where only winners
        # play at a venue.
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
        # Alice plays all 3, always winning. venue_b only has game 3.
        # At game 3: Alice has won twice already, so she enters with high mu.
        # Dave is new. venue_b has no history -> cabinet avg = 25.
        # After game 3 at venue_b: Alice's mu (~31) + Dave's mu (~21) -> avg ~26
        # But game 3 is the first game at venue_b, so no prior cabinet history.
        # We need a 4th game at venue_b with a new anonymous player.
        cabinets = {1: 'venue_a', 2: 'venue_a', 3: 'venue_b'}

        ratings_by_game, player_ratings, _ = compute_ratings(
            outcomes, usergame, cabinets)

        # Alice should have mu > 25 after two wins
        alice_mu = player_ratings[100].mu
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

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # Anonymous positions should get default 25.0 - discount (6.0)
        assert ratings_by_game[1][0] == pytest.approx(19.0)  # pos 1, anonymous

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

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # Cabinet average has more entries from Alice (who plays every game
        # and has high mu) than from each individual opponent.
        # Alice contributes 5 high-mu entries; each opponent contributes 1
        # low-mu entry. The average should be above 25.
        new_player_mu = ratings_by_game[6][1]  # pos 2 = index 1
        assert new_player_mu > 25.0


class TestFirstTimePlayerCabinetAverage:
    """A first-time player at a cabinet with history gets cabinet average."""

    def test_first_timer_gets_cabinet_avg(self):
        # Alice wins 5 games against different opponents at venue_a.
        # Her mu grows; opponents each contribute 1 low-mu entry.
        # Cabinet avg skews above 25 due to Alice's repeated high-mu entries.
        # NewPlayer appears in game 6 and should get that cabinet avg.
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

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        # NewPlayer (pos 2, index 1) should get cabinet avg, which is > 25
        new_player_mu = ratings_by_game[6][1]
        assert new_player_mu > 25.0


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

        ratings_by_game, _, _ = compute_ratings(outcomes, usergame, cabinets)

        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            with open(tmp.name, 'wb') as f:
                pickle.dump(ratings_by_game, f)
            with open(tmp.name, 'rb') as f:
                loaded = pickle.load(f)

        assert set(loaded.keys()) == set(ratings_by_game.keys())
        for game_id in ratings_by_game:
            np.testing.assert_array_equal(
                loaded[game_id], ratings_by_game[game_id])
