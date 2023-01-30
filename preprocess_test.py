import csv
import io
from typing import Optional
import unittest

import preprocess


def parse_event_helper(line: str) -> Optional[preprocess.GameEvent]:
    line_with_header = 'id,timestamp,event_type,values,game_id\n' + line
    return preprocess.parse_event(next(csv.DictReader(io.StringIO(line_with_header))))


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
        self.assertEqual()


if __name__ == '__main__':
    unittest.main()
