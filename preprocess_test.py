import csv
import datetime
import io
from typing import Optional
import unittest

import preprocess


def parse_event_helper(line: str) -> Optional[preprocess.GameEvent]:
    line_with_header = 'id,timestamp,event_type,values,game_id\n' + line
    return preprocess.parse_event(next(csv.DictReader(io.StringIO(line_with_header))))


class GameStartEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14678452,2022-05-01 23:30:59.664+00,gamestart,"{map_day,False,0,False}",363946')
        self.assertEqual(event.map, preprocess.Map.map_day)
        self.assertEqual(event.timestamp, datetime.datetime(2022, 5, 1, 23, 30, 59, 664000,
                                                            tzinfo=datetime.timezone.utc))


class MapStartEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper(
            '34229734,2022-09-19 00:13:35.058+00,mapstart,"{map_night,False,0,False,17.26}",430536')
        self.assertEqual(event.map, preprocess.Map.map_night)
        self.assertEqual(event.game_version, '17.26')


class BerryDepositEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('35074841,2022-09-26 02:20:23.549+00,berryDeposit,"{1058,722,9}",434751')
        self.assertEqual(event.drone_x, 1058)
        self.assertEqual(event.drone_y, 722)
        self.assertEqual(event.position_id, 9)


class BerryKickInEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('35079762,2022-09-26 02:47:08.521+00,berryKickIn,"{1692,110,1,True}",434770')
        self.assertEqual(event.drone_x, 1692)
        self.assertEqual(event.drone_y, 110)
        self.assertEqual(event.position_id, 1)
        self.assertTrue(event.counts_for_own_team)


class BlessMaidenEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('35079630,2022-09-26 02:46:44.638+00,blessMaiden,"{960,700,Blue}",434770')
        self.assertEqual(event.gate_x, 960)
        self.assertEqual(event.gate_y, 700)
        self.assertEqual(event.gate_color, preprocess.GateState.BLUE)

        event = parse_event_helper('35079537,2022-09-26 02:46:13.12+00,blessMaiden,"{1750,740,Red}",434770')
        self.assertEqual(event.gate_color, preprocess.GateState.GOLD)
        self.assertEqual(event.timestamp, datetime.datetime(2022, 9, 26, 2, 46, 13, 120000,
                                                            tzinfo=datetime.timezone.utc))


class CarryFoodEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('35079652,2022-09-26 02:46:49.591+00,carryFood,{10},434770')
        self.assertEqual(event.position_id, 10)


class GetOffSnailEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14683854,2022-05-01 23:42:10.787+00,getOffSnail,"{1680,11,"""",9}",363957')
        self.assertEqual(event.snail_x, 1680)
        self.assertEqual(event.snail_y, 11)
        self.assertEqual(event.position_id, 9)


class GetOnSnailEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14683840,2022-05-01 23:42:08.362+00,getOnSnail,"{1669,11,9}",363957')
        self.assertEqual(event.snail_x, 1669)
        self.assertEqual(event.snail_y, 11)
        self.assertEqual(event.position_id, 9)


class GlanceEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14683945,2022-05-01 23:42:25.96+00,glance,"{91,1015,8,3}",363957')
        self.assertEqual(event.glance_x, 91)
        self.assertEqual(event.glance_y, 1015)
        self.assertEqual(event.position_ids, [8, 3])


class PlayerKillEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14683949,2022-05-01 23:42:26.433+00,playerKill,"{1822,998,3,2,Queen}",363957')
        self.assertEqual(event.killer_x, 1822)
        self.assertEqual(event.killer_y, 998)
        self.assertEqual(event.killer_position_id, 3)
        self.assertEqual(event.killed_position_id, 2)
        self.assertEqual(event.killed_player_category, preprocess.PlayerCategory.Queen)


class UseMaidenEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('14683930,2022-05-01 23:42:23.774+00,useMaiden,"{310,620,maiden_wings,3}",363957')
        self.assertEqual(event.maiden_x, 310)
        self.assertEqual(event.maiden_y, 620)
        self.assertEqual(event.maiden_type, preprocess.MaidenType.maiden_wings)
        self.assertEqual(event.position_id, 3)

        event = parse_event_helper('14682204,2022-05-01 23:39:06.502+00,useMaiden,"{340,140,maiden_speed,8}",363957')
        self.assertEqual(event.maiden_type, preprocess.MaidenType.maiden_speed)


class VictoryEventTest(unittest.TestCase):
    def test_parse(self):
        event = parse_event_helper('34245901,2022-09-19 02:53:46.448+00,victory,"{Blue,military}",430601')
        self.assertEqual(event.winning_team, preprocess.Team.Blue)
        self.assertEqual(event.victory_condition, preprocess.VictoryCondition.military)

        event = parse_event_helper('34258141,2022-09-19 04:02:28.067+00,victory,"{Gold,snail}",430647')
        self.assertEqual(event.winning_team, preprocess.Team.Gold)
        self.assertEqual(event.victory_condition, preprocess.VictoryCondition.snail)

        event = parse_event_helper('34258141,2022-09-19 04:02:28.067+00,victory,"{Gold,economic}",430647')
        self.assertEqual(event.winning_team, preprocess.Team.Gold)
        self.assertEqual(event.victory_condition, preprocess.VictoryCondition.economic)


if __name__ == '__main__':
    unittest.main()
