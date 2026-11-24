import unittest

from ccld.analyzer.tree_wait import TreeWaitAttribution, directional_scores, uniform_slowdown


def states():
    parents = (-1, 0, 0, 1)
    waits = (
        {1: 104, 2: 0},
        {0: 50, 3: 100},
        {0: 108},
        {1: 53},
    )
    return {
        rank: dict(tree_wait_probe=True, start_mono_ns=1_000_000 + rank * 1000, duration_ns=2_000_000,
                   channels=[dict(id=0, tree_parent=parents[rank],
                                  tree_wait_peers=[dict(rank=peer, data_spins=spins, credit_spins=0)
                                                   for peer, spins in waits[rank].items()])])
        for rank in range(4)
    }


class TreeWaitTests(unittest.TestCase):
    def test_propagated_wait_is_removed_and_leaf_is_attributed_after_repetition(self):
        rows = states()
        scores = directional_scores(rows)
        self.assertEqual(scores, {0: 0, 1: 7, 2: 0, 3: 100})
        stages = TreeWaitAttribution(.2)
        for window in range(3):
            event = stages.refine({'kind': 'undetermined'}, rows, window + 1, window)
        self.assertEqual((event['kind'], event['roots'], event['communication_roots']), ('S2', [3], [3]))
        rows[1]['channels'][0]['tree_wait_peers'].clear()
        event = stages.refine({'kind': 'undetermined'}, rows, 4, 3)
        self.assertEqual((event['kind'], event['roots']), ('undetermined', []))

    def test_attribution_gate_requires_shared_clock_and_every_rank_to_be_slow(self):
        rows, hosts = states(), dict.fromkeys(range(4), 'same')
        self.assertTrue(uniform_slowdown(rows, hosts, .5, 2))
        self.assertFalse(uniform_slowdown(rows, {**hosts, 2: 'other'}, .5, 2))
        rows[3]['start_mono_ns'] += 5_000_000
        self.assertFalse(uniform_slowdown(rows, hosts, .5, 2))
        rows[3]['start_mono_ns'] -= 5_000_000
        rows[0]['duration_ns'] = 500_000
        self.assertFalse(uniform_slowdown(rows, hosts, .5, 2))

    def test_bad_edge_or_saturated_counter_does_not_supply_an_attribution(self):
        broken = states()
        broken[2]['channels'][0]['tree_wait_peers'][0]['data_spins'] = 2**32 - 1
        with self.assertRaisesRegex(ValueError, 'saturated'):
            directional_scores(broken)
        broken[3]['channels'][0]['tree_parent'] = 2
        with self.assertRaisesRegex(ValueError, 'Tree .* evidence'):
            directional_scores(broken)

    def test_previous_completion_can_explain_inherited_entry_skew(self):
        rows, hosts = states(), dict.fromkeys(range(4), 'same')
        for row in rows.values():
            row['start_mono_ns'] += 100_000_000
        rows[3]['start_mono_ns'] += 3_000_000
        ends = {rank: row['start_mono_ns'] - 60_000_000 for rank, row in rows.items()}
        self.assertFalse(uniform_slowdown(rows, hosts, .5, 2))
        self.assertTrue(uniform_slowdown(rows, hosts, .5, 2, ends))
        rows[3]['start_mono_ns'] += 5_000_000
        self.assertFalse(uniform_slowdown(rows, hosts, .5, 2, ends))
