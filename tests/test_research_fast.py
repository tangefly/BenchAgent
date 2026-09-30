import json
import unittest
import io
from contextlib import redirect_stdout

from agent.research import SourceStore
from agent.research_fast import FastResearchEngine, parse_object, tool_result_text
from test_research import FakeClient, call, reply


class FastResearchTests(unittest.TestCase):
    def test_log_levels_filter_console_without_changing_research(self):
        runs = {}
        for level in ('basic', 'full'):
            client = FakeClient([
                call('research', query='Founder', source_ids=['S1']),
                call('search_documents', queries=['Ada']),
                reply({'findings': [self.report()['findings'][0]], 'gaps': []}),
                call('research', query='Location', source_ids=['S2']),
                reply({'findings': [self.report()['findings'][1]], 'gaps': []}),
                reply(self.final()),
            ])
            output = io.StringIO()
            with redirect_stdout(output):
                result = FastResearchEngine(client, self.store, log_level=level).run('Which lab?')
            runs[level] = (result, client.calls, output.getvalue())
        basic, full = runs['basic'][2], runs['full'][2]
        for marker in ('[MAIN TOOL CALL]', '[SUB TOOL CALL]', '[SUB TOOL RESULT]', '[SUB RAW OUTPUT]', '[MAIN ANALYSIS]'):
            self.assertNotIn(marker, basic)
            self.assertIn(marker, full)
        for output in (basic, full):
            self.assertIn('[main -> sub #1]', output)
            self.assertIn('[SUB OUTPUT]', output)
            self.assertIn('[FINAL MAIN OUTPUT]', output)
            self.assertIn('Ada founded North Lab.', output)
        self.assertNotIn('"passages"', basic)
        self.assertIn('"passages"', full)
        self.assertEqual(runs['basic'][1], runs['full'][1])
        for key in ('answer', 'evidence', 'reports', 'trace', 'requests'):
            self.assertEqual(runs['basic'][0][key], runs['full'][0][key])
        for result, _, _ in runs.values():
            self.assertTrue(any(e['event'] == 'tool' and e.get('role') == 'researcher' for e in result['events']))

    def test_malformed_passages_return_tool_error_and_research_continues(self):
        invalid = ['[{"source_id":"S1"}, "location":"broken"}]',
                   '{"source_id":"S1"}', None, {}, [], ["S1"], [None],
                   [{"source_id": []}], [{"source_id": "S1", "start": "0"}]]
        for passages in invalid:
            with self.subTest(passages=passages):
                client = FakeClient([
                    call('read_passages', passages=passages),
                    reply({'findings': [self.report()['findings'][0]], 'gaps': []})])
                engine = FastResearchEngine(client, self.store, max_sub_tool_rounds=1)
                engine.active_source = 'S1'
                result = engine._extract(engine.packet('Ada', ['S1']), 'Founder')
                self.assertEqual(len(result['findings']), 1)
                event = next(e for e in engine.events if e['event'] == 'tool')
                self.assertIn('error', event['result'])
                self.assertEqual(engine.requests, 2)
                self.assertEqual(engine.trace[-1], 'main')

    def test_passage_validation_preserves_document_scope(self):
        engine = FastResearchEngine(FakeClient([]), self.store)
        engine.active_source = 'S1'
        with self.assertRaisesRegex(ValueError, 'assigned document'):
            engine._sub_tool('read_passages', {'passages': [{'source_id': 'S2'}]})
        result = engine._sub_tool('read_passages', {'passages': [{'source_id': 'S1'}]})
        self.assertEqual(result['passages'][0]['source_id'], 'S1')

    def test_invalid_log_level_rejected(self):
        with self.assertRaises(ValueError):
            FastResearchEngine(FakeClient([]), self.store, log_level='silent')

    def setUp(self):
        self.store = SourceStore()
        self.store.add('a', 'Ada founded North Lab in 2010.')
        self.store.add('b', 'North Lab is located in Oslo.')
        self.addCleanup(self.store.db.close)

    def report(self):
        return {'findings': [
            {'source_id': 'S1', 'fact': 'Ada founded North Lab.', 'quote': 'Ada founded North Lab in 2010.', 'evidence_id': 'ignored'},
            {'source_id': 'S2', 'fact': 'North Lab is in Oslo.', 'quote': 'North Lab is located in Oslo.'}], 'gaps': []}


    def test_packet_budget_and_long_doc_retrieval(self):
        self.store.add('c', ('boilerplate ' * 1000) + 'uniqueplace Antarctica ' + ('other ' * 1000))
        engine = FastResearchEngine(FakeClient([]), self.store, packet_chars=6000)
        packet = engine.packet('uniqueplace')
        self.assertLessEqual(sum(len(x['text']) for p in packet for x in p['passages']), 6000)
        self.assertTrue(all(p['complete'] for p in packet[:2]))
        self.assertFalse(packet[2]['complete'])
        with self.assertRaises(ValueError):
            engine.packet('anything', ['other-sample'])


    def test_sub_search_and_read_are_native_tools_and_expand_citable_sources(self):
        client = FakeClient([
            call('search_documents', queries=['Oslo'], source_ids=['S2']),
            call('read_passages', passages=[{'source_id': 'S2', 'start': 0, 'length': 100}]),
            reply({'findings': [self.report()['findings'][1]], 'gaps': []})])
        engine = FastResearchEngine(client, self.store, max_sub_tool_rounds=2)
        engine.query = 'Where is North Lab?'
        result = engine._extract(engine.packet('Ada', ['S1']), 'Find the location')
        self.assertEqual(result['findings'][0]['source_id'], 'S2')
        self.assertEqual(result['warnings'], [])
        self.assertEqual(len(result['coverage']), 2)
        self.assertEqual(engine.requests, 3)
        # _extract 不再自己管理 trace: 直接调用时停在进入前的 main
        self.assertTrue(all(c['trace'] == ['main'] for c in client.calls))
        self.assertIsNone(client.calls[-1]['tools'])
        self.assertEqual(client.calls[0]['tools'][0]['function']['name'], 'search_documents')
        self.assertEqual(client.calls[1]['messages'][-1]['role'], 'tool')
        self.assertEqual(engine.trace, ['main'])

    def test_sub_tools_scoped_without_text_truncation(self):
        engine = FastResearchEngine(FakeClient([]), self.store)
        self.assertEqual(engine._sub_tool('search_documents', {'queries': ['Oslo'], 'source_ids': ['S1']})['passages'], [])
        for name, args in [('search_documents', {'queries': ['Oslo'], 'source_ids': ['S99']}),
                           ('read_passages', {'passages': [{'source_id': '/etc/passwd'}]})]:
            with self.assertRaises((ValueError, KeyError)):
                engine._sub_tool(name, args)
        self.store.add('large', 'North Lab Oslo ' * 3000)
        rows = engine._sub_tool('read_passages', {'passages': [
            {'source_id': 'S3', 'start': i, 'length': 8000} for i in (0, 8000, 16000)]})['passages']
        self.assertEqual(sum(len(r['text']) for r in rows), 24000)

    def test_repeated_sub_call_uses_cache_without_forcing_final(self):
        client = FakeClient([call('search_documents', queries=['Oslo']),
                             call('search_documents', queries=['Oslo']), reply(self.report())])
        engine = FastResearchEngine(client, self.store, max_sub_tool_rounds=5)
        engine._extract(engine.packet('Ada'), 'Location')
        self.assertEqual(engine.requests, 3)
        self.assertIsNotNone(client.calls[-1]['tools'])
        tools = [e for e in engine.events if e['event'] == 'tool']
        self.assertTrue(tools[-1]['result']['cached'])

    def test_sub_executes_all_queries_and_tool_calls_with_explicit_unlimited_rounds(self):
        from unittest.mock import patch
        engine = FastResearchEngine(FakeClient([]), self.store)
        with patch.object(self.store, 'search', wraps=self.store.search) as search:
            engine._sub_tool('search_documents', {'queries': ['Oslo'] * 7, 'limit': 20})
            self.assertEqual(search.call_count, 7)
            self.assertEqual(search.call_args.kwargs['limit'], 20)
        batch = call('search_documents', queries=['Oslo'])
        second = call('read_passages', passages=[{'source_id': 'S1'}])['tool_calls'][0]
        second['id'] = 'call_2'
        batch['tool_calls'].append(second)
        client = FakeClient([batch, call('search_documents', queries=['Ada']),
                             call('search_documents', queries=['North']), reply(self.report())])
        engine = FastResearchEngine(client, self.store, max_sub_tool_rounds=None)
        engine._extract(engine.packet('Ada'), 'Research')
        self.assertEqual(engine.requests, 4)
        self.assertTrue(all(c['tools'] for c in client.calls))
        self.assertEqual(len([e for e in engine.events if e['event'] == 'tool']), 4)


    def test_global_request_budget_stops_worker_and_returns_control(self):
        client = FakeClient([call('research', query='Founder'), call('search_documents', queries=['Ada'])])
        result = FastResearchEngine(client, self.store, max_requests=2).run('Which lab?')
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['trace'][-1], 'main')
        self.assertIn('global model request budget exhausted', result['gaps'])

    def test_default_sub_tool_budget_returns_to_main(self):
        client = FakeClient([call('search_documents', queries=['Ada']),
                             call('search_documents', queries=['Oslo']), reply(self.report())])
        engine = FastResearchEngine(client, self.store)
        engine.query = 'Which lab?'
        engine._extract(engine.packet('Ada'), 'Founder')
        self.assertIsNone(client.calls[-1]['tools'])
        self.assertEqual(engine.trace[-1], 'main')

    def final(self):
        return {'prediction': 'North Lab', 'support': 'S1 founder; S2 location', 'citations': ['S1', 'S2'], 'gaps': []}

    def test_main_analyzes_between_single_document_workers(self):
        client = FakeClient([call('research', query='Extract founder evidence', source_ids=['S1']),
                             reply({'findings': [self.report()['findings'][0]], 'gaps': []}),
                             call('research', query='Extract location evidence', source_ids=['S2']),
                             reply({'findings': [self.report()['findings'][1]], 'gaps': []}), reply(self.final())])
        result = FastResearchEngine(client, self.store, max_followups=0).run('Which lab?')
        self.assertEqual(result['trace'], ['main', 'sub', 'main', 'sub', 'main'])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['requests'], 5)
        self.assertEqual([r['document_id'] for r in result['reports']], ['S1', 'S2'])
        self.assertEqual(result['remaining_sources'], [])
        self.assertNotIn('North Lab is located in Oslo.', client.calls[1]['messages'][0]['content'])
        self.assertNotIn('Ada founded North Lab in 2010.', client.calls[3]['messages'][0]['content'])
        self.assertIn('Ada founded North Lab', client.calls[2]['messages'][-2]['content'])
        self.assertIsNotNone(client.calls[2]['tools'])
        # tools 恒定传入: 收口轮不再藏工具, 保证 main prompt 的模板头部逐请求稳定
        self.assertIsNotNone(client.calls[-1]['tools'])
        events = [e['event'] for e in result['events'] if e['event'] in {'delegation', 'sub_result'}]
        self.assertEqual(events, ['delegation', 'sub_result', 'delegation', 'sub_result'])

    def test_early_final_rejected_until_every_document_processed(self):
        client = FakeClient([call('research', query='Extract relevant facts'), reply(self.report()), reply(self.final()),
                             call('research', query='Extract relevant facts'), reply(self.report()), reply(self.final())])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['requests'], 6)
        self.assertIn('Remaining documents: S2', client.calls[3]['messages'][-1]['content'])
        self.assertEqual([r['document_id'] for r in result['reports']], ['S1', 'S2'])

    def test_sub_cannot_search_or_read_other_documents(self):
        client = FakeClient([call('search_documents', queries=['Oslo']),
                             call('read_passages', passages=[{'source_id': 'S2'}]),
                             reply({'findings': [self.report()['findings'][0]], 'gaps': []})])
        engine = FastResearchEngine(client, self.store)
        engine.query = 'Which lab?'
        result = engine._delegate({'query': 'Extract founder facts', 'source_ids': ['S1']})
        events = [e for e in engine.events if e['event'] == 'tool']
        self.assertEqual(events[0]['result']['passages'], [])
        self.assertIn('assigned document', events[1]['result']['error'])
        self.assertEqual([r['source_id'] for r in result['coverage']], ['S1'])
        self.assertIsNone(engine.active_source)
        engine.active_source = 'S1'
        with self.assertRaises(ValueError):
            engine._sub_tool('search_documents', {'queries': ['Oslo'], 'source_ids': ['S2']})

    def test_rejects_multiple_documents_and_whole_task(self):
        engine = FastResearchEngine(FakeClient([]), self.store)
        engine.query = 'Which lab?'
        for args in ({'query': 'Which lab?'}, {'query': 'Facts', 'source_ids': ['S1', 'S2']},
                     {'query': 'Facts', 'source_ids': []}, {'query': 'Facts', 'source_ids': ['S99']}):
            with self.assertRaises(ValueError):
                engine._delegate(args)
        self.assertEqual(engine.requests, 0)

    def test_reinspection_after_initial_pass_allows_new_question_same_text(self):
        client = FakeClient([call('research', query='Extract facts', source_ids=['S1']), reply(self.report()),
                             call('research', query='Extract facts', source_ids=['S2']), reply(self.report()),
                             call('research', query='Verify founder date', source_ids=['S1']), reply(self.report()),
                             reply(self.final())])
        result = FastResearchEngine(client, self.store, max_followups=1).run('Which lab?')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual([r['document_id'] for r in result['reports']], ['S1', 'S2', 'S1'])
        self.assertIsNotNone(client.calls[-1]['tools'])

    def test_duplicate_reinspection_does_not_start_worker(self):
        engine = FastResearchEngine(FakeClient([reply(self.report()), reply(self.report())]), self.store)
        engine.query = 'Which lab?'
        engine._delegate({'query': 'Facts', 'source_ids': ['S1']})
        with self.assertRaises(ValueError):
            engine._delegate({'query': 'Verify date', 'source_ids': ['S1']})
        engine._delegate({'query': 'Facts', 'source_ids': ['S2']})
        result = engine._delegate({'query': 'Facts', 'source_ids': ['S1']})
        self.assertIn('already researched', result['notice'])
        self.assertEqual(engine.requests, 2)

    def test_only_one_document_delegated_per_main_turn(self):
        batch = call('research', query='Founder', source_ids=['S1'])
        extra = call('research', query='Location', source_ids=['S2'])['tool_calls'][0]
        extra['id'] = 'call_2'
        batch['tool_calls'].append(extra)
        client = FakeClient([batch, reply(self.report()), call('research', query='Location', source_ids=['S2']),
                             reply(self.report()), reply(self.final())])
        result = FastResearchEngine(client, self.store).run('Which lab?')
        self.assertEqual(result['requests'], 5)
        self.assertIn('Not executed', client.calls[2]['messages'][-2]['content'])

    def test_unfinished_document_pass_cannot_answer(self):
        client = FakeClient([call('research', query='Founder'), reply(self.report()), reply(self.final())])
        result = FastResearchEngine(client, self.store, max_main_turns=2).run('Which lab?')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['prediction'], '')
        self.assertEqual(result['remaining_sources'], ['S2'])

    def test_one_document_needs_only_one_worker(self):
        store = SourceStore()
        self.addCleanup(store.db.close)
        store.add('only', 'Ada founded North Lab in 2010.')
        client = FakeClient([call('research', query='Extract founder'), reply(self.report()),
                             reply({'prediction': 'North Lab', 'citations': ['S1'], 'gaps': []})])
        result = FastResearchEngine(client, store, max_followups=0).run('Which lab?')
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['status'], 'answered')

    def test_batch_delegation_runs_one_sub_per_document_in_order(self):
        client = FakeClient([call('research', query='Extract facts', source_ids=['S1', 'S2']),
                             reply({'findings': [self.report()['findings'][0]], 'gaps': []}),
                             reply({'findings': [self.report()['findings'][1]], 'gaps': []}),
                             reply(self.final())])
        result = FastResearchEngine(client, self.store, max_followups=0, subagents_per_turn=2).run('Which lab?')
        # 一次批量委派的两个 sub 共用 main 的同一次下潜: main -> sub sub -> main
        self.assertEqual(result['trace'], ['main', 'sub', 'sub', 'main'])
        self.assertEqual(client.calls[1]['trace'], ['main', 'sub'])
        self.assertEqual(client.calls[2]['trace'], ['main', 'sub', 'sub'])
        self.assertEqual(client.calls[3]['trace'], ['main', 'sub', 'sub', 'main'])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['requests'], 4)
        self.assertEqual([r['document_id'] for r in result['reports']], ['S1', 'S2'])
        self.assertEqual(result['subagents_per_turn'], 2)
        self.assertEqual([e['event'] for e in result['events'] if e['event'] in {'delegation', 'sub_result'}],
                         ['delegation', 'sub_result', 'delegation', 'sub_result'])
        # 批内每个 sub 仍只见自己的文档: S1 的 sub 提示词不含 S2 的内容, 反之亦然
        self.assertNotIn('located in Oslo', client.calls[1]['messages'][0]['content'])
        self.assertNotIn('Ada founded North Lab in 2010.', client.calls[2]['messages'][0]['content'])
        # 两个 sub 顺序执行, 结果合并进同一条 tool 消息一次返回 main
        tool_message = next(m for m in client.calls[3]['messages'] if m['role'] == 'tool')
        merged = json.loads(tool_message['content'])
        self.assertEqual([r['document_id'] for r in merged['results']], ['S1', 'S2'])
        # sub 原文逐字嵌入 report 字段(供 LMInfer 按位拼接输出 KV), 元数据在外层
        self.assertEqual(merged['results'][0]['report']['findings'][0]['fact'], 'Ada founded North Lab.')
        self.assertIn(json.dumps({'findings': [self.report()['findings'][0]], 'gaps': []}),
                      tool_message['content'])

    def test_batch_reinspection_truncates_to_remaining_research_budget(self):
        client = FakeClient([reply({'findings': [self.report()['findings'][0]], 'gaps': []}),
                             reply({'findings': [self.report()['findings'][1]], 'gaps': []}),
                             reply({'findings': [self.report()['findings'][0]], 'gaps': []})])
        engine = FastResearchEngine(client, self.store, max_followups=1, subagents_per_turn=2)
        engine.query = 'Which lab?'
        engine._delegate({'query': 'Facts', 'source_ids': ['S1']})
        engine._delegate({'query': 'Location', 'source_ids': ['S2']})
        # 总 sub 预算 = 2 文档 + 1 复查: 剩 1 个名额时批量复查只执行第一篇, 其余不产生 sub 请求
        merged = engine._delegate({'query': 'Verify both', 'source_ids': ['S1', 'S2']})
        self.assertEqual([r['document_id'] for r in merged['results'] if 'document_id' in r], ['S1'])
        self.assertIn('S2', merged['results'][-1]['notice'])
        self.assertEqual(engine.requests, 3)
        self.assertFalse(engine._stalled(merged))

    def test_research_tool_schema_and_duplicates_follow_subagents_per_turn(self):
        for per_turn in (1, 2):
            engine = FastResearchEngine(FakeClient([]), self.store, subagents_per_turn=per_turn)
            tool = engine.research_tools[0]['function']
            self.assertEqual(tool['parameters']['properties']['source_ids']['maxItems'], per_turn)
        client = FakeClient([reply({'findings': [self.report()['findings'][0]], 'gaps': []})])
        engine = FastResearchEngine(client, self.store, subagents_per_turn=2)
        engine.query = 'Which lab?'
        engine._delegate({'query': 'Facts', 'source_ids': ['S1', 'S1']})
        self.assertEqual(engine.requests, 1)

    def test_fenced_json(self):
        self.assertEqual(parse_object('```json\n{"prediction":"A"}\n```')['prediction'], 'A')

    def test_tool_result_text_embeds_raw_output_and_keeps_valid_json(self):
        raw = json.dumps({'findings': [{'source_id': 'S1', 'fact': 'F', 'quote': 'Q'}], 'gaps': ['g']})
        row = {'task_id': 1, 'document_id': 'S1', 'task': 'Facts', 'remaining_documents': ['S2'],
               'warnings': [], 'coverage': [{'source_id': 'S1', 'complete': True, 'ranges': [[0, 30]]}],
               'findings': json.loads(raw)['findings'], 'gaps': ['g'], '_raw': raw}
        text = tool_result_text(row)
        merged = json.loads(text)                       # 拼接后整体仍是合法 JSON
        self.assertEqual(merged['report'], json.loads(raw))
        self.assertIn(raw, text)                        # sub 输出逐字节出现
        # findings/gaps 只保留在原文里, 元数据部分不重复序列化
        self.assertNotIn('findings', text[:text.index('"report"')])
        # 非 sub 行(通知/报错)照常序列化; 批量结果逐行拼接
        self.assertEqual(tool_result_text({'notice': 'stop'}), '{"notice": "stop"}')
        batch = tool_result_text({'results': [{'document_id': 'S1', '_raw': raw}, {'notice': 'budget'}]})
        merged_batch = json.loads(batch)
        self.assertEqual(merged_batch['results'][0]['report'], json.loads(raw))
        self.assertEqual(merged_batch['results'][1], {'notice': 'budget'})


if __name__ == '__main__':
    unittest.main()
