from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent_memory_demo import (
    Candidate, MemoryConflict, MemoryStore, ProviderUnavailable, Source,
    decision, digest, jev_advisory, retrieve, run_demo,
)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.sources = [Source("cedar", "policy", "Escalate critical support incidents"),
                        Source("maple", "policy", "SECRET synthetic Maple policy")]

    def test_stale_index_uses_current_text(self):
        result = retrieve("cedar", "support", self.sources, [Candidate("cedar", "policy", digest("old"))])
        self.assertEqual(result["rejected"][0]["reason"], "stale_hash")
        self.assertEqual(result["hits"][0]["route"], "lexical_fallback")
        self.assertEqual(result["hits"][0]["text"], self.sources[0].text)

    def test_same_id_wrong_project_cannot_leak(self):
        candidate = Candidate("maple", "policy", digest(self.sources[1].text))
        result = retrieve("cedar", "SECRET", self.sources, [candidate])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["rejected"][0]["reason"], "wrong_project")
        self.assertTrue(result["requires_review"])

    def test_current_hash_and_duplicates(self):
        candidate = Candidate("cedar", "policy", digest(self.sources[0].text))
        result = retrieve("cedar", "support", self.sources, [candidate, candidate])
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["route"], "verified_candidate")

    def test_missing_source_rejected(self):
        result = retrieve("cedar", "unknown", self.sources, [Candidate("cedar", "missing", digest("x"))])
        self.assertEqual(result["rejected"][0]["reason"], "source_missing")
        self.assertTrue(result["requires_review"])

    def test_empty_candidate_list_still_falls_back(self):
        self.assertEqual(retrieve("cedar", "support", self.sources, [])["hits"][0]["route"], "lexical_fallback")

    def test_duplicate_canonical_source_is_ambiguous(self):
        with self.assertRaises(ValueError):
            retrieve("cedar", "support", self.sources + [self.sources[0]], [])

    def test_invalid_inputs(self):
        for project, query, limit in [("", "x", 1), ("cedar", "", 1), ("cedar", "x", 0)]:
            with self.subTest(project=project, query=query, limit=limit), self.assertRaises(ValueError):
                retrieve(project, query, self.sources, [], limit)


class DecisionTests(unittest.TestCase):
    def test_confidence_boundaries(self):
        expected = [(0.2, "propose_deny"), (0.2001, "review"), (0.7999, "review"), (0.8, "propose_allow")]
        for probability, route in expected:
            with self.subTest(probability=probability):
                self.assertEqual(decision(probability, "fixture", True)["route"], route)

    def test_missing_evidence_requires_review(self):
        self.assertEqual(decision(0.99, "live", False)["route"], "review")

    def test_invalid_probability_and_mode(self):
        for probability in [float("nan"), float("inf"), -0.1, 1.1, "0.9", True]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                decision(probability, "fixture", True)
        with self.assertRaises(ValueError):
            decision(0.9, "mock", True)

    def test_fixture_never_grants_execution(self):
        self.assertEqual(decision(1.0, "fixture", True)["execution_authority"], "none")


class ProviderTests(unittest.TestCase):
    def response(self, data, returncode=0):
        return subprocess.CompletedProcess([], returncode, json.dumps(data), "synthetic private stderr")

    def valid(self):
        return {"mode": "live", "model": {"resolved": "synthetic-test-provider/model"},
                "decisions": {"decision": {"probabilityTrue": 0.9, "value": True, "verdict": "act"}},
                "review": {"required": False}}

    def test_live_response_and_argv(self):
        with patch("agent_memory_demo.subprocess.run", return_value=self.response(self.valid())) as run:
            result = jev_advisory("Question $(not shell)")
            self.assertEqual(result["mode"], "live")
            self.assertEqual(result["execution_authority"], "none")
            self.assertEqual(run.call_args.args[0], ["jev", "--json", "ask",
                "Public synthetic portfolio demo; advisory only.", "Question $(not shell)", "--id", "decision"])

    def test_provider_review_is_preserved(self):
        data = self.valid()
        data["review"]["required"] = True
        with patch("agent_memory_demo.subprocess.run", return_value=self.response(data)):
            self.assertEqual(jev_advisory("test")["route"], "review")

    def test_mock_or_missing_model_is_rejected(self):
        for data in [dict(self.valid(), mode="mock"), dict(self.valid(), model={})]:
            with patch("agent_memory_demo.subprocess.run", return_value=self.response(data)), self.assertRaises(ProviderUnavailable):
                jev_advisory("test")

    def test_failed_cli_does_not_expose_stderr_or_fall_back(self):
        with patch("agent_memory_demo.subprocess.run", return_value=self.response(self.valid(), 1)):
            with self.assertRaises(ProviderUnavailable) as caught:
                jev_advisory("test")
            self.assertNotIn("private stderr", str(caught.exception))

    def test_timeout_and_malformed_contract_fail_closed(self):
        for exc in [subprocess.TimeoutExpired("jev", 1), FileNotFoundError()]:
            with patch("agent_memory_demo.subprocess.run", side_effect=exc), self.assertRaises(ProviderUnavailable):
                jev_advisory("test")
        for data in [{}, dict(self.valid(), decisions={"decision": {"probabilityTrue": "high"}})]:
            with patch("agent_memory_demo.subprocess.run", return_value=self.response(data)), self.assertRaises(ProviderUnavailable):
                jev_advisory("test")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(Path(self.directory.name) / "memory.sqlite")

    def test_duplicate_and_exact_readback(self):
        self.assertTrue(self.store.retain("cedar", "key", {"b": 2, "a": 1}, "MARKER")["created"])
        self.assertFalse(self.store.retain("cedar", "key", {"a": 1, "b": 2}, "MARKER")["created"])
        self.assertEqual(self.store.readback("cedar", "key"), {"payload": {"a": 1, "b": 2}, "marker": "MARKER", "status": "processed"})

    def test_conflicts_do_not_overwrite(self):
        self.store.retain("cedar", "key", {"v": 1}, "MARKER")
        for payload, marker in [({"v": 2}, "MARKER"), ({"v": 1}, "OTHER")]:
            with self.assertRaises(MemoryConflict):
                self.store.retain("cedar", "key", payload, marker)
        self.assertEqual(self.store.readback("cedar", "key")["payload"], {"v": 1})

    def test_project_and_exact_marker_isolation(self):
        self.store.retain("cedar", "key", {"v": 1}, "MARKER")
        self.store.retain("maple", "key", {"v": 2}, "MARKER")
        self.assertEqual(len(self.store.recall_marker("cedar", "MARKER")), 1)
        self.assertEqual(self.store.recall_marker("cedar", "MARK"), [])
        self.assertIsNone(self.store.readback("unknown", "key"))
        self.assertEqual(self.store.recall_marker("cedar", "MARKER")[0]["payload"], {"v": 1})

    def test_concurrent_retries_create_one_row(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.store.retain("cedar", "key", {"v": 1}, "MARKER"), range(24)))
        self.assertEqual(sum(r["created"] for r in results), 1)
        self.assertEqual(len(self.store.recall_marker("cedar", "MARKER")), 1)

    def test_invalid_payload_cannot_commit(self):
        for payload in [{"v": float("nan")}, [1, 2]]:
            with self.assertRaises(ValueError):
                self.store.retain("cedar", "key", payload, "MARKER")
        self.assertIsNone(self.store.readback("cedar", "key"))


class DemoTests(unittest.TestCase):
    def test_fixture_contracts_and_optional_provider_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            with patch("agent_memory_demo.jev_advisory", side_effect=ProviderUnavailable("offline")):
                result = run_demo(path, "test")
            self.assertEqual(result["passed"], result["total"])
            self.assertEqual(result["total"], 8)
            self.assertEqual(result["live_advisory"]["route"], "review")
            self.assertEqual(json.loads(path.read_text())["data"], "synthetic")


if __name__ == "__main__":
    unittest.main()
