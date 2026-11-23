import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyzer'))
from tree_s1_vote import analyze_files, vote


def fixture():
    # Two comparable layers on each of two channels; ranks 0 and 7 have
    # singleton layers and are not eligible to receive a vote there.
    parents = [-1, 0, 0, 1, 1, 2, 2, 3]
    states = {}
    for rank,parent in enumerate(parents):
        states[rank] = {'event':'complete', 'comm_hash':'comm', 'rank':rank,
                        'world_size':8, 'mono_ns':1000, 'op_count':4, 'duration_ns':100,
                        'op':'AllReduce', 'count':256, 'datatype':'float32',
                        'algorithm':'TREE', 'protocol':'SIMPLE',
                        'channels':[{'id':cid,'tree_parent':parent} for cid in (0,1)]}
    decisions = []
    for channel in (0,1):
        decisions.extend([
            {'category':'slow','kind':'S1','roots':[1], 'comm_hash':'comm','op_count':4,
             'window':2,'group':f'tree:channel={channel}:depth=1'},
            {'category':'slow','kind':'S2','roots':[4], 'comm_hash':'comm','op_count':4,
             'window':2,'group':f'tree:channel={channel}:depth=2'},
        ])
    return states,decisions


class TreeS1VoteTests(unittest.TestCase):
    def test_s1_ignores_affected_s2_groups(self):
        states,decisions = fixture()
        result = vote(states,decisions)
        self.assertEqual((result['status'],result['roots']),('candidate',[1]))
        self.assertEqual(result['scores']['1'],{'s1_votes':2,'eligible_channels':2})
        self.assertEqual(result['scores']['4'],{'s1_votes':0,'eligible_channels':2})
        self.assertEqual(result['scores']['0']['eligible_channels'],0)

    def test_s2_only_does_not_force_an_s1_candidate(self):
        states,decisions = fixture()
        for item in decisions:
            item['kind'] = 'S2'
        result = vote(states,decisions)
        self.assertEqual((result['status'],result['roots']),('not_applicable',[]))

    def test_tied_unanimous_candidates_are_undetermined(self):
        states,decisions = fixture()
        for item in decisions:
            if item['kind'] == 'S1':
                item['roots'] = [1,2]
        result = vote(states,decisions)
        self.assertEqual((result['status'],result['roots']),('undetermined',[]))

    def test_missing_or_wrong_group_is_rejected(self):
        states,decisions = fixture()
        with self.assertRaisesRegex(ValueError,'incomplete'):
            vote(states,decisions[:-1])
        decisions[0]['roots'] = [7]
        with self.assertRaisesRegex(ValueError,'outside'):
            vote(states,decisions)

    def test_different_operation_signatures_are_rejected(self):
        states,decisions = fixture()
        states[3]['count'] = 255
        with self.assertRaisesRegex(ValueError,'one completed Tree operation'):
            vote(states,decisions)

    def test_invalid_operation_signature_is_rejected(self):
        states,decisions = fixture()
        states[3]['count'] = [256]
        with self.assertRaisesRegex(ValueError,'signature is incomplete'):
            vote(states,decisions)

    def test_offline_reader_refuses_missing_coverage(self):
        states,decisions = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank,state in states.items():
                init = {'event':'comm_init','comm_hash':'comm','rank':rank,'world_size':8,'mono_ns':0}
                events = [init,state]
                if rank == 1:
                    events.append({'event':'coverage_gap','comm_hash':'comm','rank':1,
                                   'world_size':8,'mono_ns':2000})
                (root/f'ccld-rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            (root/'decisions.ndjson').write_text(''.join(json.dumps(x)+'\n' for x in decisions))
            result = analyze_files(root,root/'decisions.ndjson')
            self.assertEqual((result[0]['status'],result[0]['roots']),('incomplete',[]))

    def test_offline_reader_does_not_treat_preliminary_detection_as_a_vote(self):
        states,decisions = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank,state in states.items():
                init = {'event':'comm_init','comm_hash':'comm','rank':rank,'world_size':8,'mono_ns':0}
                (root/f'ccld-rank{rank}.jsonl').write_text(json.dumps(init)+'\n'+json.dumps(state)+'\n')
            for item in decisions:
                item['kind'] = 'detected'
            (root/'decisions.ndjson').write_text(''.join(json.dumps(x)+'\n' for x in decisions))
            result = analyze_files(root,root/'decisions.ndjson')
            self.assertEqual((result[0]['status'],result[0]['roots']),('no_confirmed_tree_slow',[]))


if __name__ == '__main__':
    unittest.main()
