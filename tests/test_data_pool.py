"""Tests for data_pool.interleave_dedup and build_union_pool."""

import unittest
from data_pool import interleave_dedup, build_union_pool


class TestInterleaveDedup(unittest.TestCase):
    def test_two_sources_no_overlap(self):
        a = [(1, b'a'), (2, b'b')]
        b = [(3, b'c'), (4, b'd')]
        result = interleave_dedup(a, b)
        # Round-robin: a[0], b[0], a[1], b[1]
        self.assertEqual(result, [(1, b'a'), (3, b'c'), (2, b'b'), (4, b'd')])

    def test_two_sources_with_overlap(self):
        a = [(1, b'a1'), (2, b'a2')]
        b = [(2, b'b2'), (3, b'b3')]
        result = interleave_dedup(a, b)
        # a[0]=1, b[0]=2, a[1]=2(dup skip), b[1]=3
        self.assertEqual(result, [(1, b'a1'), (2, b'b2'), (3, b'b3')])

    def test_first_source_wins_on_tie(self):
        a = [(1, b'from_a')]
        b = [(1, b'from_b')]
        result = interleave_dedup(a, b)
        self.assertEqual(result, [(1, b'from_a')])

    def test_three_sources(self):
        a = [(1, b'a')]
        b = [(2, b'b')]
        c = [(3, b'c')]
        result = interleave_dedup(a, b, c)
        self.assertEqual(result, [(1, b'a'), (2, b'b'), (3, b'c')])

    def test_three_sources_all_duplicates(self):
        a = [(1, b'a')]
        b = [(1, b'b')]
        c = [(1, b'c')]
        result = interleave_dedup(a, b, c)
        self.assertEqual(result, [(1, b'a')])

    def test_empty_source(self):
        a = [(1, b'a'), (2, b'b')]
        result = interleave_dedup(a, [])
        self.assertEqual(result, [(1, b'a'), (2, b'b')])

    def test_all_empty(self):
        result = interleave_dedup([], [], [])
        self.assertEqual(result, [])

    def test_single_source(self):
        a = [(1, b'a'), (2, b'b')]
        result = interleave_dedup(a)
        self.assertEqual(result, [(1, b'a'), (2, b'b')])

    def test_unequal_lengths(self):
        a = [(1, b'a'), (2, b'b'), (3, b'c')]
        b = [(4, b'd')]
        result = interleave_dedup(a, b)
        # a[0]=1, b[0]=4, a[1]=2, (b exhausted), a[2]=3
        self.assertEqual(result, [(1, b'a'), (4, b'd'), (2, b'b'), (3, b'c')])


class TestBuildUnionPool(unittest.TestCase):
    def test_stats(self):
        qf = [(1, b'q1'), (2, b'q2'), (3, b'q3')]
        li = [(2, b'l2'), (4, b'l4')]
        pool, stats = build_union_pool(qf, li)
        self.assertEqual(stats['qf_total'], 3)
        self.assertEqual(stats['li_total'], 2)
        self.assertEqual(stats['overlap'], 1)  # game_id 2
        self.assertEqual(stats['union_total'], 4)
        self.assertEqual(len(pool), 4)

    def test_no_overlap(self):
        qf = [(1, b'q')]
        li = [(2, b'l')]
        pool, stats = build_union_pool(qf, li)
        self.assertEqual(stats['overlap'], 0)
        self.assertEqual(stats['union_total'], 2)


if __name__ == '__main__':
    unittest.main()
