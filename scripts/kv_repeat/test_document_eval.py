"""Regression checks for private document loading and paired task execution."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.kv_repeat import run_subagent_kv_repeat_eval as evaluator


class DocumentEvalTests(unittest.TestCase):
    def test_sentence_trimming(self):
        for original, expected in [
            ('One fact.\nUnfinished', 'One fact.'),
            ('One fact.\nAnother Dr. Smi', 'One fact.'),
            ('One fact. Second partial', 'One fact.'),
            ('完整句子。\n未完成', '完整句子。'),
            ('One fact.\nFinished!', 'One fact.\nFinished!'),
            ('Only a fragment', ''),
        ]:
            with self.subTest(original=original):
                actual, _ = evaluator.trim_incomplete_sentence(original, 'length')
                self.assertEqual(actual, expected)
        self.assertEqual(evaluator.trim_incomplete_sentence('No punctuation', 'stop')[0],
                         'No punctuation')

    def test_document_only_reaches_sub_and_task_is_bound(self):
        calls = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.last_usage = {}
                self.last_reused_tokens = 0
                self.last_finish_reason = 'stop'
                self.session_id = 'fake'

            def chat(self, messages, **kwargs):
                calls.append((copy.deepcopy(messages), kwargs))
                self.last_usage = {'completion_tokens': 64}
                if len(calls) == 1:
                    return {'tool_calls': [{'id': 'call', 'function': {
                        'name': 'call_subagent',
                        'arguments': json.dumps({'task': 'Changed task'})}}]}
                if len(calls) == 2:
                    self.last_finish_reason = 'length'
                    return {'content': 'Extracted fact.\nUnfinished private suffix'}
                return {'content': 'Extracted fact.'}

            def release_kv(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = 'Private document content that main must never receive.'
            (root / 'doc.txt').write_text(text)
            row = dict(document_path='doc.txt', document_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       subagent_prompt='Extract facts.', target_repeat_tokens=64)
            args = SimpleNamespace(dataset=root/'questions.jsonl', base_url='unused',
                                   api_key='unused', model='fake', timeout=1,
                                   enable_thinking=False, temperature=0, max_tokens=2048,
                                   sub_max_tokens=None, sub_temperature=0, include_text=True,
                                   release_kv=True, repeat_instruction='strict',
                                   trim_incomplete_sentence=True)
            with patch.object(evaluator, 'LLMClient', FakeClient):
                result = evaluator.run_one(row, 0, args)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[2][0][-1], {
                'role': 'user', 'content': evaluator.STRICT_REPEAT_INSTRUCTION})
            self.assertIn(text, calls[1][0][0]['content'])
            self.assertNotIn(text, json.dumps(calls[0][0]+calls[2][0]))
            self.assertTrue(calls[1][0][0]['content'].startswith('Extract facts.'))
            self.assertEqual(calls[1][1]['max_tokens'], 64)
            self.assertEqual(calls[1][1]['trace'], ['main', 'sub'])
            self.assertEqual(result['metrics']['exact_match'], 1.0)
            self.assertNotIn('Unfinished private suffix', json.dumps(calls[2][0]))
            self.assertTrue(result['sentence_trimming']['applied'])
            self.assertEqual(result['sub_output'], 'Extracted fact.')
            self.assertIn('Unfinished private suffix', result['sub_output_raw'])
            row['document_sha256'] = 'incorrect'
            with patch.object(evaluator, 'LLMClient') as client:
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    evaluator.run_one(row, 0, args)
                client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
