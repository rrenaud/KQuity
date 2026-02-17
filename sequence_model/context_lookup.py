"""Load usergame.csv + game.csv and provide per-game player/scene/cabinet context.

Dense index mappings (0 = unknown/anonymous):
  user_id  -> 1..N_users   (0 = anonymous)
  scene    -> 1..N_scenes   (0 = unknown)
  cabinet  -> 1..N_cabinets (0 = unknown)

Context vector per game: shape (21,) dtype uint16
  [user_idx_pos1..pos10, scene_idx_pos1..pos10, cabinet_idx]
"""

import csv
import json
import os

import numpy as np

CONTEXT_DIM = 21  # 10 user_ids + 10 scene_ids + 1 cabinet_id


class ContextLookup:
    def __init__(self, usergame_csv, game_csv):
        # Build dense index mappings
        self._user_to_idx = {}  # user_id (int) -> dense index (1..N)
        self._scene_to_idx = {}  # scene (str) -> dense index (1..N)
        self._cabinet_to_idx = {}  # cabinet_id (int) -> dense index (1..N)

        # game_id -> {position_id: (user_idx, scene_idx)}
        self._game_users = {}
        # game_id -> cabinet_idx
        self._game_cabinet = {}

        self._load_usergame(usergame_csv)
        self._load_game(game_csv)

    def _load_usergame(self, path):
        """Load usergame.csv: id,game_id,position_id,user_id,name,scene"""
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_id = int(row['game_id'])
                position_id = int(row['position_id'])
                user_id = int(row['user_id'])
                scene = row['scene'].strip()

                # Assign dense indices
                if user_id not in self._user_to_idx:
                    self._user_to_idx[user_id] = len(self._user_to_idx) + 1
                if scene and scene not in self._scene_to_idx:
                    self._scene_to_idx[scene] = len(self._scene_to_idx) + 1

                user_idx = self._user_to_idx[user_id]
                scene_idx = self._scene_to_idx.get(scene, 0)

                if game_id not in self._game_users:
                    self._game_users[game_id] = {}
                self._game_users[game_id][position_id] = (user_idx, scene_idx)

    def _load_game(self, path):
        """Load game.csv: id,...,cabinet_id,..."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_id = int(row['id'])
                cabinet_id_str = row['cabinet_id'].strip()
                if cabinet_id_str:
                    cabinet_id = int(cabinet_id_str)
                    if cabinet_id not in self._cabinet_to_idx:
                        self._cabinet_to_idx[cabinet_id] = len(self._cabinet_to_idx) + 1
                    self._game_cabinet[game_id] = self._cabinet_to_idx[cabinet_id]

    def get_game_context(self, game_id):
        """Returns np.array shape (21,) dtype uint16.

        [user_idx_pos1..pos10, scene_idx_pos1..pos10, cabinet_idx]
        Positions are ordered 1-10. Missing positions get index 0.
        """
        ctx = np.zeros(CONTEXT_DIM, dtype=np.uint16)

        users = self._game_users.get(game_id, {})
        for pos_id in range(1, 11):
            if pos_id in users:
                user_idx, scene_idx = users[pos_id]
                ctx[pos_id - 1] = user_idx        # slots 0-9: user indices
                ctx[10 + pos_id - 1] = scene_idx   # slots 10-19: scene indices

        ctx[20] = self._game_cabinet.get(game_id, 0)
        return ctx

    @staticmethod
    def swap_context(context):
        """Swap positions for symmetry augmentation.

        Blue positions are even (2,4,6,8,10), Gold are odd (1,3,5,7,9).
        Symmetry swaps blue<->gold, so pairs: pos1<->pos2, pos3<->pos4, ..., pos9<->pos10.
        """
        swapped = context.copy()
        for i in range(0, 10, 2):  # user_ids
            swapped[i], swapped[i + 1] = context[i + 1], context[i]
        for i in range(10, 20, 2):  # scene_ids
            swapped[i], swapped[i + 1] = context[i + 1], context[i]
        # cabinet stays the same (slot 20)
        return swapped

    @property
    def n_users(self):
        return len(self._user_to_idx)

    @property
    def n_scenes(self):
        return len(self._scene_to_idx)

    @property
    def n_cabinets(self):
        return len(self._cabinet_to_idx)

    def save_meta(self, path):
        """Save index mapping sizes to JSON for model construction."""
        meta = {
            'n_users': self.n_users,
            'n_scenes': self.n_scenes,
            'n_cabinets': self.n_cabinets,
            'context_dim': CONTEXT_DIM,
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)
        return meta
