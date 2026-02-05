"""Token vocabulary and event-to-token mapping for Killer Queen game sequences.

Each game event becomes 1-4 tokens. The vocabulary is small (~53 tokens)
to let the transformer compose event semantics from parts.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from constants import Team, VictoryCondition, PlayerCategory, MaidenType, Map


# --- Token definitions ---
# Special tokens
PAD = 0
BOS = 1
EOS = 2

# Map tokens (4)
MAP_DAY = 3
MAP_NIGHT = 4
MAP_DUSK = 5
MAP_TWILIGHT = 6

# Side assignment (2)
GOLD_LEFT = 7
GOLD_RIGHT = 8

# Event type tokens (14)
SPAWN = 9
CARRY_FOOD = 10
BERRY_DEPOSIT = 11
BERRY_KICK_IN = 12
PLAYER_KILL = 13
BLESS_MAIDEN = 14
USE_MAIDEN = 15
GET_ON_SNAIL = 16
GET_OFF_SNAIL = 17
SNAIL_EAT = 18
SNAIL_ESCAPE = 19

# Victory tokens — combined team + condition (6)
VICTORY_BLUE_MILITARY = 20
VICTORY_BLUE_ECONOMIC = 21
VICTORY_BLUE_SNAIL = 22
VICTORY_GOLD_MILITARY = 23
VICTORY_GOLD_ECONOMIC = 24
VICTORY_GOLD_SNAIL = 25

# Player/position IDs (10: positions 1-10)
PLAYER_1 = 26
PLAYER_2 = 27
PLAYER_3 = 28
PLAYER_4 = 29
PLAYER_5 = 30
PLAYER_6 = 31
PLAYER_7 = 32
PLAYER_8 = 33
PLAYER_9 = 34
PLAYER_10 = 35

# Maiden IDs (5)
MAIDEN_0 = 36
MAIDEN_1 = 37
MAIDEN_2 = 38
MAIDEN_3 = 39
MAIDEN_4 = 40

# Team indicators (2)
TEAM_BLUE = 41
TEAM_GOLD = 42

# Bot flag (2)
IS_BOT = 43
IS_HUMAN = 44

# Maiden type (2)
MAIDEN_SPEED = 45
MAIDEN_WINGS = 46

# berryKickIn direction (2)
OWN_TEAM_GOAL = 47
OPP_TEAM_GOAL = 48

# Kill target type (3)
KILLED_QUEEN = 49
KILLED_SOLDIER = 50
KILLED_WORKER = 51

# Snail position deciles (10)
SNAIL_POS_0 = 52  # leftmost decile of track
SNAIL_POS_1 = 53
SNAIL_POS_2 = 54
SNAIL_POS_3 = 55
SNAIL_POS_4 = 56
SNAIL_POS_5 = 57
SNAIL_POS_6 = 58
SNAIL_POS_7 = 59
SNAIL_POS_8 = 60
SNAIL_POS_9 = 61  # rightmost decile of track

VOCAB_SIZE = 62

# --- Lookup tables ---

_MAP_TOKEN = {
    Map.map_day: MAP_DAY,
    Map.map_night: MAP_NIGHT,
    Map.map_dusk: MAP_DUSK,
    Map.map_twilight: MAP_TWILIGHT,
}

_VICTORY_TOKEN = {
    (Team.BLUE, VictoryCondition.military): VICTORY_BLUE_MILITARY,
    (Team.BLUE, VictoryCondition.economic): VICTORY_BLUE_ECONOMIC,
    (Team.BLUE, VictoryCondition.snail): VICTORY_BLUE_SNAIL,
    (Team.GOLD, VictoryCondition.military): VICTORY_GOLD_MILITARY,
    (Team.GOLD, VictoryCondition.economic): VICTORY_GOLD_ECONOMIC,
    (Team.GOLD, VictoryCondition.snail): VICTORY_GOLD_SNAIL,
}

_KILLED_TYPE_TOKEN = {
    PlayerCategory.Queen: KILLED_QUEEN,
    PlayerCategory.Soldier: KILLED_SOLDIER,
    PlayerCategory.Worker: KILLED_WORKER,
}

_TEAM_TOKEN = {
    Team.BLUE: TEAM_BLUE,
    Team.GOLD: TEAM_GOLD,
}

# Token names for debugging/display
TOKEN_NAMES = [
    '<PAD>', '<BOS>', '<EOS>',
    'map_day', 'map_night', 'map_dusk', 'map_twilight',
    'gold_left', 'gold_right',
    'spawn', 'carryFood', 'berryDeposit', 'berryKickIn',
    'playerKill', 'blessMaiden', 'useMaiden',
    'getOnSnail', 'getOffSnail', 'snailEat', 'snailEscape',
    'victory_blue_military', 'victory_blue_economic', 'victory_blue_snail',
    'victory_gold_military', 'victory_gold_economic', 'victory_gold_snail',
    'player_1', 'player_2', 'player_3', 'player_4', 'player_5',
    'player_6', 'player_7', 'player_8', 'player_9', 'player_10',
    'maiden_0', 'maiden_1', 'maiden_2', 'maiden_3', 'maiden_4',
    'team_blue', 'team_gold',
    'is_bot', 'is_human',
    'maiden_speed', 'maiden_wings',
    'own_team_goal', 'opp_team_goal',
    'killed_queen', 'killed_soldier', 'killed_worker',
    'snail_p0', 'snail_p1', 'snail_p2', 'snail_p3', 'snail_p4',
    'snail_p5', 'snail_p6', 'snail_p7', 'snail_p8', 'snail_p9',
]

assert len(TOKEN_NAMES) == VOCAB_SIZE


def player_token(position_id: int) -> int:
    """Convert position_id (1-10) to player token."""
    assert 1 <= position_id <= 10, f"Invalid position_id: {position_id}"
    return PLAYER_1 + (position_id - 1)


def maiden_token(maiden_index: int) -> int:
    """Convert maiden index (0-4) to maiden token."""
    assert 0 <= maiden_index <= 4, f"Invalid maiden_index: {maiden_index}"
    return MAIDEN_0 + maiden_index


def token_name(token_id: int) -> str:
    """Get human-readable name for a token ID."""
    if 0 <= token_id < VOCAB_SIZE:
        return TOKEN_NAMES[token_id]
    return f'<UNK:{token_id}>'


def decode_tokens(token_ids):
    """Convert a sequence of token IDs to human-readable names."""
    return [token_name(t) for t in token_ids]


# --- Event to token conversion ---
# These functions take preprocess event objects and return lists of token IDs.

def tokenize_game_start(map_type, gold_on_left):
    """Tokenize game header: <BOS> map_type side_assignment."""
    return [BOS, _MAP_TOKEN[map_type], GOLD_LEFT if gold_on_left else GOLD_RIGHT]


def tokenize_spawn(position_id, is_bot):
    """spawn player_N is_bot|is_human"""
    return [SPAWN, player_token(position_id), IS_BOT if is_bot else IS_HUMAN]


def tokenize_carry_food(position_id):
    """carryFood player_N"""
    return [CARRY_FOOD, player_token(position_id)]


def tokenize_berry_deposit(position_id):
    """berryDeposit player_N"""
    return [BERRY_DEPOSIT, player_token(position_id)]


def tokenize_berry_kick_in(position_id, counts_for_own_team):
    """berryKickIn player_N own_team|opp_team"""
    direction = OWN_TEAM_GOAL if counts_for_own_team else OPP_TEAM_GOAL
    return [BERRY_KICK_IN, player_token(position_id), direction]


def tokenize_player_kill(killer_position_id, killed_position_id, killed_category):
    """playerKill player_N player_M killed_type"""
    return [
        PLAYER_KILL,
        player_token(killer_position_id),
        player_token(killed_position_id),
        _KILLED_TYPE_TOKEN[killed_category],
    ]


def tokenize_bless_maiden(maiden_index, team):
    """blessMaiden maiden_K team_X"""
    return [BLESS_MAIDEN, maiden_token(maiden_index), _TEAM_TOKEN[team]]


def tokenize_use_maiden(position_id, maiden_type):
    """useMaiden player_N maiden_speed|wings"""
    type_tok = MAIDEN_SPEED if maiden_type == MaidenType.maiden_speed else MAIDEN_WINGS
    return [USE_MAIDEN, player_token(position_id), type_tok]


def snail_position_token(snail_x, track_left, track_width):
    """Convert raw snail_x pixel coordinate to a decile token.

    The track runs from track_left to track_left + track_width.
    We normalize to [0, 1] (left-to-right on screen), clamp, and bucket
    into 10 deciles. The model already knows gold_left/gold_right from
    the BOS header, so it can interpret direction.
    """
    fraction = (snail_x - track_left) / track_width
    fraction = max(0.0, min(1.0, fraction))
    bucket = min(9, int(fraction * 10))
    return SNAIL_POS_0 + bucket


def tokenize_get_on_snail(position_id, snail_pos_tok):
    """getOnSnail player_N snail_pN"""
    return [GET_ON_SNAIL, player_token(position_id), snail_pos_tok]


def tokenize_get_off_snail(position_id, snail_pos_tok):
    """getOffSnail player_N snail_pN"""
    return [GET_OFF_SNAIL, player_token(position_id), snail_pos_tok]


def tokenize_snail_eat(rider_position_id, eaten_position_id, snail_pos_tok):
    """snailEat player_N player_M snail_pN"""
    return [SNAIL_EAT, player_token(rider_position_id),
            player_token(eaten_position_id), snail_pos_tok]


def tokenize_snail_escape(position_id, snail_pos_tok):
    """snailEscape player_N snail_pN"""
    return [SNAIL_ESCAPE, player_token(position_id), snail_pos_tok]


def tokenize_victory(winning_team, victory_condition):
    """victory_team_condition (single combined token)"""
    return [_VICTORY_TOKEN[(winning_team, victory_condition)]]


def tokenize_event(event):
    """Convert a preprocess GameEvent to a list of token IDs.

    Returns None for events that don't produce tokens (GameStartEvent, MapStartEvent).
    The game header is handled separately by tokenize_game_start().
    """
    from preprocess import (
        SpawnEvent, CarryFoodEvent, BerryDepositEvent, BerryKickInEvent,
        PlayerKillEvent, BlessMaidenEvent, UseMaidenEvent,
        GetOnSnailEvent, GetOffSnailEvent, SnailEatEvent, SnailEscapeEvent,
        VictoryEvent, GameStartEvent, MapStartEvent,
    )

    if isinstance(event, SpawnEvent):
        return tokenize_spawn(event.position_id, event.is_bot)
    elif isinstance(event, CarryFoodEvent):
        return tokenize_carry_food(event.position_id)
    elif isinstance(event, BerryDepositEvent):
        return tokenize_berry_deposit(event.position_id)
    elif isinstance(event, BerryKickInEvent):
        return tokenize_berry_kick_in(event.position_id, event.counts_for_own_team)
    elif isinstance(event, PlayerKillEvent):
        return tokenize_player_kill(
            event.killer_position_id, event.killed_position_id,
            event.killed_player_category)
    elif isinstance(event, BlessMaidenEvent):
        # Need maiden_index — we get it from the event's maiden coordinates.
        # But tokenize_games.py resolves the maiden_index before calling us.
        # This path is for convenience; callers that have maiden_index should
        # use tokenize_bless_maiden directly.
        return None  # Handled specially in tokenize_games.py
    elif isinstance(event, UseMaidenEvent):
        return tokenize_use_maiden(event.position_id, event.maiden_type)
    elif isinstance(event, (GetOnSnailEvent, GetOffSnailEvent,
                            SnailEatEvent, SnailEscapeEvent)):
        return None  # Handled specially in tokenize_games.py (needs map_info for snail pos)
    elif isinstance(event, VictoryEvent):
        return tokenize_victory(event.winning_team, event.victory_condition)
    elif isinstance(event, (GameStartEvent, MapStartEvent)):
        return None  # Handled by tokenize_game_start
    else:
        return None  # Unknown event type, skip


# --- What we drop from tokenization ---
# TODO: Add discretized spatial grid tokens (e.g. 8x6 buckets) for kill
#       locations — may help model learn map control patterns
# TODO: Add time-gap bucket tokens (e.g. <fast>/<medium>/<slow>/<very_slow>)
#       between events — may help model reason about game pace and urgency
# Snail position deciles are now included (SNAIL_POS_0..9) on all snail events.
# TODO: Add berry-slot tokens (0-11) to berryDeposit events —
#       lets model track how close a team is to economic victory
