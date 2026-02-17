"""Utilities for building deduplicated training pools from multiple data sources.

Each source provides a list of (game_id, encoded_bytes) tuples, pre-sorted
by quality ranking. Interleaving ensures top-N sampling gets a balanced mix.
"""


def interleave_dedup(*sources):
    """Round-robin interleave multiple entry lists, deduplicating by game_id.

    Args:
        *sources: Each source is a list of (game_id, encoded_bytes) tuples.

    Returns:
        List of (game_id, encoded_bytes) with duplicates removed.
        First occurrence (by round-robin order) wins.
    """
    seen = set()
    pool = []
    indices = [0] * len(sources)

    active = True
    while active:
        active = False
        for i, src in enumerate(sources):
            if indices[i] < len(src):
                active = True
                gid, data = src[indices[i]]
                indices[i] += 1
                if gid not in seen:
                    seen.add(gid)
                    pool.append((gid, data))

    return pool


def build_union_pool(qf_entries, li_entries):
    """Build deduplicated QF+LI union with overlap statistics.

    Returns:
        (pool, stats_dict) where stats_dict has keys:
        qf_total, li_total, overlap, union_total
    """
    pool = interleave_dedup(qf_entries, li_entries)
    qf_ids = {gid for gid, _ in qf_entries}
    li_ids = {gid for gid, _ in li_entries}
    stats = {
        'qf_total': len(qf_entries),
        'li_total': len(li_entries),
        'overlap': len(qf_ids & li_ids),
        'union_total': len(pool),
    }
    return pool, stats
