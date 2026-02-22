import collections
from typing import Any

import copy
from constants import SCREEN_WIDTH, MaidenType, Map
import json

from _types import Coord


class MapStructureInfo:

    def get_berry_index(self, berry_x: int, berry_y: int) -> int:
        berry_coord: Coord = (berry_x, berry_y)
        if berry_coord in self._gold_berries:
            return self._gold_berries[berry_coord]
        elif berry_coord in self._blue_berries:
            return self._blue_berries[berry_coord]
        raise ValueError('Berry not found: ({}, {})'.format(berry_x, berry_y))

    def get_type_and_maiden_index(self, maiden_x: int, maiden_y: int) -> tuple[MaidenType, int]:
        if not (maiden_x, maiden_y) in self._maidens:
            raise ValueError('Maiden not found: ({}, {})'.format(maiden_x, maiden_y))
        return self._maidens[(maiden_x, maiden_y)]

    def __init__(self, map_id: Map, raw_info: dict[str, Any]) -> None:
        self.map_id: Map = map_id
        self._gold_berries: dict[Coord, int] = {}
        self._blue_berries: dict[Coord, int] = {}
        self._maidens: dict[Coord, tuple[MaidenType, int]] = {}

        def index_coord_list(l: list[list[int]]) -> dict[Coord, int]:
            return {(value[0], value[1]): i for i, value in enumerate(l)}

        self.gold_on_left: bool = True
        self._gold_berries = index_coord_list(raw_info['left_berries'])
        self._blue_berries = index_coord_list(raw_info['right_berries'])

        for idx, maiden in enumerate(raw_info['maiden_info']):
            maiden_type, x, y = maiden
            maiden_type = MaidenType(maiden_type)
            self._maidens[(x, y)] = (maiden_type, idx)

        self.snail_track_width: float = raw_info['snail_track_width']
        self.total_berries: int = raw_info['total_berries']

    def flip_sides(self) -> 'MapStructureInfo':
        flipped = copy.deepcopy(self)
        flipped._gold_berries, flipped._blue_berries = flipped._blue_berries, flipped._gold_berries
        flipped._maidens = {(SCREEN_WIDTH - k[0], k[1]): v for k, v in flipped._maidens.items()}
        flipped.gold_on_left = not self.gold_on_left
        return flipped


class MapStructureInfos:

    def __init__(self) -> None:
        self.backing: dict[tuple[Map, bool], MapStructureInfo] = {}
        with open('map_structure_info.json', 'rb') as f:
            raw_info_dict: dict[str, Any] = json.load(f)
            for map_name, raw_info in raw_info_dict.items():
                original = MapStructureInfo(Map[map_name], raw_info)
                self.backing[(Map[map_name], True)] = original
                self.backing[(Map[map_name], False)] = original.flip_sides()

    def get_map_info(self, map: Map, gold_on_left: bool) -> MapStructureInfo:
        return self.backing[(map, gold_on_left)]
