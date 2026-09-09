import json
import unittest

from agent.research import SourceStore
from agent.research_fast import FastResearchEngine, parse_object
from test_research import FakeClient, call, reply


class FastResearchTests(unittest.TestCase):
    def setUp(self):
        self.store = SourceStore()
        self.store.add('a', 'Ada founded North Lab in 2010.')
        self.store.add('b', 'North Lab is located in Oslo.')
        self.addCleanup(self.store.db.close)

    def report(self):
        return {'findings': [
            {'source_id': 'S1', 'fact': 'Ada founded North Lab.', 'quote': 'Ada founded North Lab in 2010.', 'evidence_id': 'ignored'},
            {'source_id': 'S2', 'fact': 'North Lab is in Oslo.', 'quote': 'North Lab is located in Oslo.'}], 'gaps': []}

    def test_three_call_path_and_extra_fields(self):
        client = FakeClient([call('research', query='Ada Oslo', source_ids=['S1']), reply(self.report()),
                             reply({'prediction': 'North Lab', 'support': 'S1 founder; S2 location', 'citations': ['S1', 'S2'], 'gaps': []})])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['trace'], ['main', 'sub', 'main'])
        self.assertEqual(len(result['evidence']), 2)
        # Initial call always covers both documents, despite model selecting only S1.
        self.assertIn('North Lab is located in Oslo', client.calls[1]['messages'][0]['content'])

    def test_invalid_quote_falls_back_without_repair_loop(self):
        bad = {'findings': [{'source_id': 'S1', 'fact': 'False', 'quote': 'Fabricated text'}]}
        client = FakeClient([call('research', query='Ada'), reply(bad),
                             reply({'prediction': 'North Lab', 'support': 'S1', 'citations': ['S1'], 'gaps': ['location uncertain']})])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['prediction'], 'North Lab')
        self.assertEqual(result['evidence'], [])
        self.assertIn('source_fallback', result['reports'][0])

    def test_duplicate_research_does_not_call_worker_again(self):
        client = FakeClient([call('research', query='Ada'), reply(self.report()), call('research', query='Ada'),
                             reply({'prediction': 'North Lab', 'citations': ['S1'], 'gaps': []})])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(len(result['reports']), 1)
        self.assertEqual(result['requests'], 4)

    def test_packet_budget_and_long_doc_retrieval(self):
        self.store.add('c', ('boilerplate ' * 1000) + 'uniqueplace Antarctica ' + ('other ' * 1000))
        engine = FastResearchEngine(FakeClient([]), self.store, packet_chars=6000)
        packet = engine.packet('uniqueplace')
        self.assertLessEqual(sum(len(x['text']) for p in packet for x in p['passages']), 6000)
        self.assertTrue(all(p['complete'] for p in packet[:2]))
        self.assertFalse(packet[2]['complete'])
        with self.assertRaises(ValueError):
            engine.packet('anything', ['other-sample'])

    def test_focused_followup_uses_one_more_sub_and_returns_to_main(self):
        client = FakeClient([call('research', query='Ada'), reply(self.report()),
                             reply({'prediction': 'North Lab', 'follow_up': {'query': 'Oslo', 'source_ids': ['S2']}}),
                             reply({'findings': [self.report()['findings'][1]], 'gaps': []}),
                             reply({'prediction': 'North Lab', 'citations': ['S1', 'S2'], 'gaps': []})])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(result['requests'], 5)
        self.assertEqual(result['trace'], ['main', 'sub', 'main', 'sub', 'main'])
        self.assertIsNone(client.calls[2]['tools'])
        self.assertEqual(len(result['reports']), 2)

    def test_fenced_json_and_no_research_cannot_answer(self):
        self.assertEqual(parse_object('```json\n{"prediction":"A"}\n```')['prediction'], 'A')
        client = FakeClient([reply({'prediction': 'North Lab'}) for _ in range(3)])
        result = FastResearchEngine(client, self.store, max_followups=0).run('Question')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['prediction'], '')


if __name__ == '__main__':
    unittest.main()
