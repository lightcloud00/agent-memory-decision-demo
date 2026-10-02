"""Small, dependency-free agent-memory reference implementation.

All bundled inputs are synthetic. Fixture decisions never authorize actions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import time
from typing import Any, Iterator


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class Source:
    project: str
    source_id: str
    text: str


@dataclass(frozen=True)
class Candidate:
    project: str
    source_id: str
    indexed_sha256: str


def retrieve(
    project: str,
    query: str,
    sources: list[Source],
    candidates: list[Candidate],
    limit: int = 3,
) -> dict[str, Any]:
    """Verify supplied index candidates against current sources; fall back locally.

    This is a candidate-validation contract, not an implementation of vector search.
    Only the requested project's current text can appear in a returned hit.
    """
    if not project.strip() or not query.strip() or limit < 1:
        raise ValueError("project, query, and a positive limit are required")
    current: dict[tuple[str, str], Source] = {}
    for source in sources:
        key = (source.project, source.source_id)
        if key in current:
            raise ValueError("duplicate canonical source identifier")
        current[key] = source
    hits, rejected, seen = [], [], set()
    query_terms = terms(query)
    for candidate in candidates:
        key = (candidate.project, candidate.source_id)
        if candidate.project != project:
            rejected.append({"source_id": candidate.source_id, "reason": "wrong_project"})
            continue
        source = current.get(key)
        if source is None:
            rejected.append({"source_id": candidate.source_id, "reason": "source_missing"})
        elif candidate.indexed_sha256 != digest(source.text):
            rejected.append({"source_id": candidate.source_id, "reason": "stale_hash"})
        elif key not in seen:
            hits.append({"source_id": source.source_id, "text": source.text,
                         "sha256": digest(source.text), "route": "verified_candidate"})
            seen.add(key)
    # Add bounded lexical hits if candidate validation leaves room. Tie order is stable.
    fallback = sorted(
        ((len(query_terms & terms(s.text)), s.source_id, s)
         for s in sources if s.project == project and (s.project, s.source_id) not in seen),
        key=lambda item: (-item[0], item[1]),
    )
    for overlap, _, source in fallback:
        if len(hits) >= limit or overlap == 0:
            break
        hits.append({"source_id": source.source_id, "text": source.text,
                     "sha256": digest(source.text), "route": "lexical_fallback"})
    return {"project": project, "hits": hits[:limit], "rejected": rejected,
            "requires_review": not bool(hits)}


def decision(probability_true: float, mode: str, evidence_available: bool) -> dict[str, Any]:
    """Return a proposed boolean route, never permission to execute a tool."""
    if (isinstance(probability_true, bool) or not isinstance(probability_true, (int, float))
            or not math.isfinite(probability_true) or not 0 <= probability_true <= 1):
        raise ValueError("probability must be a finite number between zero and one")
    if mode not in {"fixture", "live"}:
        raise ValueError("explicit fixture or live mode required")
    route = "review"
    if evidence_available:
        if probability_true >= 0.8:
            route = "propose_allow"
        elif probability_true <= 0.2:
            route = "propose_deny"
    return {"probability_true": probability_true, "mode": mode, "route": route,
            "execution_authority": "none"}


class ProviderUnavailable(RuntimeError):
    """A live provider failed; the caller must surface review, not a fixture answer."""


def jev_advisory(question: str, cli: str = "jev", timeout: float = 15.0) -> dict[str, Any]:
    """Optional adapter for the operator-managed `jev --json ask` CLI.

    No credentials are read. This adapter executes no suggested action. The CLI
    must return a successful, explicitly live result with resolved model metadata.
    """
    try:
        process = subprocess.run(
            [cli, "--json", "ask", "Public synthetic portfolio demo; advisory only.",
             question, "--id", "decision"], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        if process.returncode != 0:
            raise ProviderUnavailable("Jev CLI failed; provider stderr is not exposed")
        data = json.loads(process.stdout)
        if data.get("mode") != "live" or not data.get("model", {}).get("resolved"):
            raise ProviderUnavailable("live mode and resolved model evidence required")
        result = data["decisions"]["decision"]
        if type(result.get("value")) is not bool or result.get("verdict") not in {"act", "review"}:
            raise ProviderUnavailable("typed boolean value and verdict required")
        proposal = decision(result["probabilityTrue"], "live", True)
        if result["verdict"] == "review" or data.get("review", {}).get("required"):
            proposal["route"] = "review"
        proposal["provider_model"] = data["model"]["resolved"]
        return proposal
    except ProviderUnavailable:
        raise
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError,
            KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ProviderUnavailable("Jev unavailable or response contract invalid") from exc


class MemoryConflict(ValueError):
    """Reusing an idempotency key with different data is a conflict."""


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    project TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    marker TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status = 'processed'),
                    PRIMARY KEY(project, idempotency_key)
                )
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def retain(self, project: str, key: str, payload: dict[str, Any], marker: str) -> dict[str, Any]:
        if not project.strip() or not key.strip() or not marker.strip():
            raise ValueError("project, idempotency key, and exact marker are required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            # Serializes competing writes; uniqueness remains enforced by the database.
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM memory WHERE project=? AND idempotency_key=?", (project, key),
            ).fetchone()
            if existing:
                if existing["payload_json"] != serialized or existing["marker"] != marker:
                    raise MemoryConflict("idempotency key already belongs to a different payload")
                return {"created": False, "status": "processed", "project": project, "key": key}
            connection.execute("INSERT INTO memory VALUES (?, ?, ?, ?, 'processed')",
                               (project, key, serialized, marker))
        # Return processed only after the context manager successfully commits.
        return {"created": True, "status": "processed", "project": project, "key": key}

    def readback(self, project: str, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory WHERE project=? AND idempotency_key=?", (project, key),
            ).fetchone()
        return None if row is None else {"payload": json.loads(row["payload_json"]),
                                         "marker": row["marker"], "status": row["status"]}

    def recall_marker(self, project: str, marker: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT idempotency_key, payload_json FROM memory WHERE project=? AND marker=? "
                "ORDER BY idempotency_key", (project, marker),
            ).fetchall()
        return [{"key": row["idempotency_key"], "payload": json.loads(row["payload_json"])} for row in rows]


def run_demo(output: Path, live_question: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    fixture = json.loads((Path(__file__).parent / "fixtures" / "sources.json").read_text())
    sources = [Source(**source) for source in fixture["sources"]]
    candidates = [Candidate(**candidate) for candidate in fixture["candidates"]]
    retrieval = retrieve("cedar", "support incident escalation", sources, candidates)
    policy = [decision(p, "fixture", True) for p in (0.93, 0.5, 0.07)]
    cases = []
    def check(name: str, passed: bool) -> None:
        cases.append({"case": name, "passed": bool(passed)})
    check("current_source_verified", any(h["route"] == "verified_candidate" for h in retrieval["hits"]))
    check("stale_index_rejected", any(r["reason"] == "stale_hash" for r in retrieval["rejected"]))
    check("fresh_lexical_fallback", any(h["route"] == "lexical_fallback" for h in retrieval["hits"]))
    check("cross_project_rejected", any(r["reason"] == "wrong_project" for r in retrieval["rejected"]))
    check("uncertain_decision_reviews", policy[1]["route"] == "review")
    check("fixtures_cannot_execute", all(p["execution_authority"] == "none" for p in policy))
    # Each run uses an isolated temporary database; no user notes are ingested.
    with tempfile.TemporaryDirectory(prefix="agent-memory-demo-") as directory:
        memory = MemoryStore(Path(directory) / "memory.sqlite")
        payload = {"summary": "Synthetic Cedar support escalation policy", "source_id": "support-policy"}
        first = memory.retain("cedar", "support-policy:v2", payload, "DEMO-CEDAR-001")
        second = memory.retain("cedar", "support-policy:v2", payload, "DEMO-CEDAR-001")
        readback = memory.readback("cedar", "support-policy:v2")
        recall = memory.recall_marker("cedar", "DEMO-CEDAR-001")
        check("duplicate_write_is_idempotent", first["created"] and not second["created"])
        check("processed_readback_exact_recall", readback is not None and readback["payload"] == payload
              and readback["status"] == "processed" and len(recall) == 1 and recall[0]["payload"] == payload)
    live: dict[str, Any] = {"attempted": False, "mode": "fixture"}
    if live_question is not None:
        try:
            live = {"attempted": True, "result": jev_advisory(live_question)}
        except ProviderUnavailable as exc:
            live = {"attempted": True, "status": "unavailable", "route": "review", "reason": str(exc)}
    report = {"data": "synthetic", "mode": "fixture", "execution_authority": "none",
              "retrieval": retrieval, "decisions": policy,
              "memory": {"first": first, "duplicate": second, "readback": readback, "recall": recall},
              "contracts": cases, "passed": sum(c["passed"] for c in cases), "total": len(cases),
              "live_advisory": live, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
              "measurement_scope": "one local fixture run; not model accuracy or production latency"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/demo.json"))
    parser.add_argument("--live-question", help="Optional operator-managed Jev advisory; failures require review")
    args = parser.parse_args()
    report = run_demo(args.output, args.live_question)
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "mode": report["mode"], "report": str(args.output),
                      "live_advisory": report["live_advisory"]}, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
