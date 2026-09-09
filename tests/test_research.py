import copy
import json
from pathlib import Path
import tempfile
import unittest

from agent.research import ResearchEngine, SourceStore
from scripts.research.run_research import sample_paths


def call(name, **args):
    return {"content": "Next research step.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def reply(value):
    return {"content": json.dumps(value)}


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.last_usage = {"prompt_tokens": 10}
        self.last_reused_tokens = 4

    def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy({"messages": messages, **kwargs}))
        return next(self.replies)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.store = SourceStore()
        self.store.add("doc1.txt", "Ada founded North Lab in 2010.\nThe lab studied ice.")
        self.store.add("doc2.txt", "North Lab is located in Oslo.")
        self.addCleanup(self.store.db.close)
        self.requirements = [{"id": "C1", "text": "Founded by Ada"}, {"id": "C2", "text": "Located in Oslo"}]

    def evidence(self, sid="S1", **overrides):
        return {"source_id": sid, "quote": "Ada founded North Lab in 2010.", "claim": "Ada founded North Lab",
                "entity": "North Lab", "requirement_ids": ["C1"], "stance": "supports", **overrides}

    def engine_with_evidence(self, replies=()):
        engine = ResearchEngine(FakeClient(replies), self.store)
        engine._dispatch("set_plan", {"requirements": self.requirements[:1]})
        windows = []
        engine._read_dispatch("read", {"source_id": "S1"}, windows)
        engine._read_dispatch("record_evidence", self.evidence(), windows)
        return engine

    def test_search_ranking_paging_and_literal_query(self):
        results = self.store.search('Oslo OR " ; DROP TABLE chunks;')
        self.assertEqual(results[0]["source_id"], "S2")
        first = self.store.read("S1", length=8)
        second = self.store.read("S1", start=first["next_start"])
        self.assertEqual(first["text"] + second["text"], self.store.sources["S1"]["text"])
        with self.assertRaises(ValueError):
            self.store.read("S1", start=-1)

    def test_quotes_require_worker_read_exact_source_and_valid_requirement(self):
        engine = ResearchEngine(FakeClient([]), self.store)
        engine._dispatch("set_plan", {"requirements": self.requirements})
        windows = []
        engine._read_dispatch("search", {"query": "Ada"}, windows)
        with self.assertRaises(ValueError):
            engine._read_dispatch("record_evidence", self.evidence(), windows)
        engine._read_dispatch("read", {"source_id": "S1"}, windows)
        evidence = engine._read_dispatch("record_evidence", self.evidence(), windows)
        self.assertEqual(evidence["evidence_id"], "E1")
        self.assertEqual(engine._read_dispatch("record_evidence", self.evidence(), windows)["evidence_id"], "E1")
        for args in [self.evidence(quote="Ada founded South Lab in 2010."), self.evidence(requirement_ids=["C9"])]:
            with self.assertRaises(ValueError):
                engine._read_dispatch("record_evidence", args, windows)
        with self.assertRaises(ValueError):
            engine._read_dispatch("record_evidence", self.evidence(), [])

    def test_end_to_end_multihop_review_and_trace(self):
        proposal = {"answer": "North Lab", "synthesis": "E1 establishes founder Ada; E2 locates the same lab in Oslo.",
                    "evidence_ids": ["E1", "E2"]}
        client = FakeClient([
            call("set_plan", requirements=self.requirements),
            call("delegate", task="Find Ada's lab and its location across documents."),
            call("search", query="Ada"), call("read", source_id="S1"),
            call("record_evidence", **self.evidence()),
            call("search", query="North Lab Oslo"), call("read", source_id="S2"),
            call("record_evidence", **self.evidence("S2", quote="North Lab is located in Oslo.",
                                                   claim="North Lab is in Oslo", requirement_ids=["C2"])),
            reply({"evidence_ids": ["E1", "E2"], "gaps": []}),
            call("review", **proposal),
            call("read", source_id="S1"), call("read", source_id="S2"),
            reply({"approved": True, "checked_evidence_ids": ["E1", "E2"], "gaps": [], "rationale": "Both links verified."}),
            call("finish", **proposal, status="answered", gaps=[]),
        ])
        result = ResearchEngine(client, self.store).run("Which lab founded by Ada is located in Oslo?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["trace"], ["main", "sub", "main", "sub", "main"])
        self.assertEqual(client.calls[9]["trace"], ["main", "sub", "main"])
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["requests"], 14)

    def test_review_cannot_approve_unread_citations_or_finish_without_review(self):
        engine = self.engine_with_evidence([reply({"approved": True, "checked_evidence_ids": ["E1"], "gaps": []})])
        proposal = {"answer": "North Lab", "synthesis": "Founded by Ada", "evidence_ids": ["E1"]}
        with self.assertRaises(ValueError):
            engine._dispatch("finish", dict(proposal, status="answered", gaps=[]))
        self.assertFalse(engine._dispatch("review", proposal)["approved"])
        with self.assertRaises(ValueError):
            engine._dispatch("finish", dict(proposal, status="answered", gaps=[]))

    def test_new_evidence_invalidates_review(self):
        engine = self.engine_with_evidence()
        engine.review = {"approved": True}
        windows = []
        engine._read_dispatch("read", {"source_id": "S1"}, windows)
        engine._read_dispatch("record_evidence", self.evidence(claim="The founding year was 2010"), windows)
        self.assertIsNone(engine.review)
        with self.assertRaises(ValueError):
            engine._dispatch("set_plan", {"requirements": self.requirements})

    def test_missing_coverage_blocks_review(self):
        engine = self.engine_with_evidence()
        engine.requirements["C2"] = "Located in Oslo"
        with self.assertRaises(ValueError):
            engine._dispatch("review", {"answer": "North Lab", "synthesis": "Only founder known", "evidence_ids": ["E1"]})
        self.assertEqual(engine.requests, 0)

    def test_budget_exhaustion_and_batch_calls(self):
        response = call("set_plan", requirements=self.requirements)
        response["tool_calls"] += call("delegate", task="must not execute")["tool_calls"]
        response["tool_calls"][1]["id"] = "call_2"
        result = ResearchEngine(FakeClient([response]), self.store, max_requests=1).run("Question")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["trace"], ["main"])
        self.assertIn("Not executed", result["events"][-1]["result"]["error"])

    def test_sample_isolation_and_no_gold_in_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("alphaonly")
            (root / "b.txt").write_text("betaonly")
            metadata = root / "metadata.json"
            metadata.write_text('{"answer": "secretgold"}')
            store = SourceStore(sample_paths({"evidence_docs": ["a.txt"]}, metadata))
            self.addCleanup(store.db.close)
            self.assertTrue(store.search("alphaonly"))
            self.assertEqual(store.search("betaonly secretgold"), [])
            with self.assertRaises(KeyError):
                store.read(str(root / "b.txt"))
            self.assertEqual(len(store.catalog()), 1)


if __name__ == "__main__":
    unittest.main()
