# Hands-on workshop: agent memory and decision contracts

Time: 40 minutes. Level: basic Python. Everything runs offline. No AWS account, no cloud calls, no packages to install.

The snippets in each step build on each other. Paste all the snippets of one step into the same `python3` session (or one file), in order.

## What you need

- Python 3 (checked here on 3.15.2; the code uses only the standard library)
- git
- The repo on disk (Step 0)

## What you will learn

The repo has four contracts. Each step below covers one:

1. **Source freshness** — a candidate is accepted only when its hash matches the current source text.
2. **Project isolation** — a hit never leaves the project that was requested.
3. **Typed review** — a decision returns a proposal, never permission to act.
4. **Idempotent retention** — the same key with the same payload is stored once, and a reused key with different data is a conflict.

All data in the repo is synthetic.

---

## Step 0 — clone and run the tests (3 min)

```bash
git clone https://github.com/lightcloud00/agent-memory-decision-demo.git
cd agent-memory-decision-demo
python3 -m unittest discover -s tests -v
```

Expected, at the end of the run (the time varies):

```
Ran 39 tests in 0.3s

OK (skipped=1)
```

The skipped test is the opt-in live Bedrock test. Leave it skipped. It needs a real provider and may cost money.

Now run the demo:

```bash
python3 agent_memory_demo.py
```

Expected: a JSON summary with `"passed": 8`, `"total": 8`, `"mode": "fixture"` and `"live_advisory": {"attempted": false, ...}`. It also rewrites `results/demo.json`, which is tracked in git; Step 1 restores it.

This workshop uses `python3` everywhere. On some machines plain `python` does not exist.

---

## Step 1 — source freshness (7 min)

**The idea.** A candidate carries the hash of the text that was indexed, so a stale index entry is detected by hashing the current source text and comparing. When the hash does not match, the candidate is rejected and the current text is served by a local lexical fallback instead.

Open a REPL from the repo root (`python3`) and paste:

```python
from agent_memory_demo import Source, Candidate, retrieve, digest

new_text = "Support incidents with customer data loss require escalation and human review."
old_text = "Support incidents with customer data loss require escalation."
sources = [Source("cedar", "support-policy", new_text)]
stale = Candidate("cedar", "support-policy", digest(old_text))

result = retrieve("cedar", "vacuum telemetry", sources, [stale])
assert result["hits"] == []
assert result["requires_review"] is True
assert any(r.get("reason") == "stale_hash" for r in result["rejected"])
```

Expected: all asserts pass. The query shares no terms with the source, so there is no hit, `requires_review` is `True`, and the rejection reason is `stale_hash`.

Now the same stale candidate, with a query that overlaps the new text:

```python
fallback = retrieve("cedar", "customer data loss", sources, [stale])
assert fallback["hits"], "expected a lexical fallback hit"
hit = fallback["hits"][0]
assert hit["route"] == "lexical_fallback"
assert hit["text"] == new_text
```

Expected: the asserts pass. The stale candidate is still rejected, and the hit comes back with route `lexical_fallback` and the current text.

### Break it on purpose

Change the fixture so the indexed hash no longer matches the stored text. Line 7 of `fixtures/sources.json` holds the Cedar support policy text:

```bash
python3 -c "from pathlib import Path; p=Path('fixtures/sources.json'); p.write_text(p.read_text().replace('customer data loss','customer data deletion'))"
```

Then:

```bash
python3 -m unittest discover -s tests -v
python3 agent_memory_demo.py
```

Expected:

- Exactly one test fails: `test_fixture_contracts_and_optional_provider_failure`. Of the other 38, 37 pass and 1 is skipped, and the run ends in `FAILED (failures=1, skipped=1)`.
- The demo summary reports `"passed": 7` and `"total": 8`.

Restore the file:

```bash
git checkout -- .
python3 -m unittest discover -s tests -v
```

Expected: back to `Ran 39 tests` and `OK (skipped=1)`.

### Try this

Give the candidate the correct hash and run it again:

```python
fresh = Candidate("cedar", "support-policy", digest(new_text))
ok = retrieve("cedar", "customer data loss", sources, [fresh])
assert not ok["rejected"]
assert ok["hits"]
assert ok["requires_review"] is False
```

Expected: nothing is rejected, there is a hit, and `requires_review` is `False`.

Now change the source text without touching the candidate:

```python
changed = [Source("cedar", "support-policy", new_text + " Changed.")]
r = retrieve("cedar", "customer data loss", changed, [fresh])
print(r["rejected"], [h["route"] for h in r["hits"]], r["requires_review"])
r = retrieve("cedar", "vacuum telemetry", changed, [fresh])
print(r["rejected"], r["hits"], r["requires_review"])
```

Expected: the candidate is rejected with `stale_hash` in both calls. The first still finds the current text through `lexical_fallback`, so `requires_review` is `False`. The second has no overlapping terms, so there are no hits and `requires_review` is `True`.

---

## Step 2 — project isolation (7 min)

**The idea.** The project is part of the query, not a filter you can forget. A candidate from another project is rejected by reason, and its text can never appear in a returned hit.

```python
from agent_memory_demo import Source, Candidate, retrieve, digest

cedar_text = "Support incidents with customer data loss require escalation and human review."
maple_text = "Synthetic Maple account policy; never disclose across project boundaries."
sources = [Source("cedar", "support-policy", cedar_text),
           Source("maple", "support-policy", maple_text)]
maple_candidate = Candidate("maple", "support-policy", digest(maple_text))

result = retrieve("cedar", "customer data loss", sources, [maple_candidate])
assert any(r.get("reason") == "wrong_project" for r in result["rejected"])
assert all("Synthetic Maple" not in str(hit) for hit in result["hits"])
assert all(hit.get("text") == cedar_text for hit in result["hits"])
```

Expected: all asserts pass. The Maple candidate is rejected with reason `wrong_project`, and no hit contains Maple text.

### Try this

Ask for the Cedar project with a query that only matches Maple text:

```python
result2 = retrieve("cedar", "account policy", sources, [maple_candidate])
assert result2["requires_review"] is True
assert all("Synthetic Maple" not in str(hit) for hit in result2["hits"])
```

Expected: still no Maple text, and `requires_review` is `True`, because nothing was verified for Cedar.

Then ask for the Maple project instead, with the same candidate:

```python
result3 = retrieve("maple", "account policy", sources, [maple_candidate])
assert result3["hits"]
assert result3["hits"][0]["text"] == maple_text
```

Expected: the Maple text comes back, because the project in the request matches the project in the candidate.

---

## Step 3 — typed review (7 min)

**The idea.** `decision` returns a route that a human or a later step can act on, and the returned dict always carries `execution_authority: "none"`. It never grants permission to run a tool, and bad input is rejected with `ValueError` instead of a guess.

```python
from agent_memory_demo import decision

results = [decision(p, m, e) for p, m, e in
           [(0.8, "fixture", True), (0.7999, "fixture", True),
            (0.2, "fixture", True), (0.99, "live", False)]]
assert [r["route"] for r in results] == ["propose_allow", "review", "propose_deny", "review"]
assert all(r["execution_authority"] == "none" for r in results)
```

Expected: the routes are `propose_allow`, `review`, `propose_deny`, `review`, in that order. The last case has no evidence, so it stays in review even at 0.99. Every result has `execution_authority` equal to `none`.

The three invalid calls:

```python
cases = [((True, "fixture", True), "probability must be a finite number between zero and one"),
         ((1.1, "fixture", True), "probability must be a finite number between zero and one"),
         ((0.5, "mock", True), "explicit fixture or live mode required")]
for args, message in cases:
    try:
        decision(*args)
    except ValueError as exc:
        assert str(exc) == message, str(exc)
    else:
        raise AssertionError("expected ValueError for %r" % (args,))
```

Expected: each call raises `ValueError` with the message shown. `True` is rejected even though it is an `int`, because a boolean is not a probability here.

### Try this

Remove the evidence and keep a high probability:

```python
assert decision(0.8, "fixture", False)["route"] == "review"
assert decision(0.2, "fixture", False)["route"] == "review"
```

Expected: both stay in `review`. Without evidence, probability alone never produces a proposal to allow or deny.

---

## Step 4 — idempotent retention (7 min)

**The idea.** A write is keyed by project and idempotency key, so the same payload written twice produces one row. Reusing a key with different data raises `MemoryConflict`, and a payload that cannot survive a JSON round trip is rejected before it is stored.

```python
import os, tempfile
from agent_memory_demo import MemoryStore, MemoryConflict

path = os.path.join(tempfile.mkdtemp(), "memory.sqlite3")
store = MemoryStore(path)
payload = {"summary": "customer data loss escalation"}

first = store.retain("cedar", "key-1", payload, "marker-1")
second = store.retain("cedar", "key-1", payload, "marker-1")
assert first["created"] is True
assert second["created"] is False
assert second["status"] == "processed"
```

Expected: the first call reports `created` `True`, the second reports `created` `False` and status `processed`.

Same key, different payload:

```python
try:
    store.retain("cedar", "key-1", {"summary": "different"}, "marker-1")
except MemoryConflict as exc:
    assert str(exc) == "idempotency key already belongs to a different payload"
else:
    raise AssertionError("expected MemoryConflict")
```

Expected: `MemoryConflict` with that message.

A payload that does not round-trip through JSON:

```python
try:
    store.retain("cedar", "key-2", {"tags": ("a", "b")}, "marker-2")
except ValueError as exc:
    assert str(exc) == "payload must round-trip through JSON unchanged"
else:
    raise AssertionError("expected ValueError")
```

Expected: `ValueError` with that message. A tuple would come back from JSON as a list, so the store refuses it instead of changing the data.

Read it back:

```python
row = store.readback("cedar", "key-1")
assert row == {"payload": payload, "marker": "marker-1", "status": "processed"}
assert store.readback("cedar", "key-2") is None
```

Expected: the first readback is the payload, the marker and the status `processed`. The second is `None`, because the rejected write never reached the database.

### Try this

Reuse the key with the same payload but a different marker, then list by marker:

```python
try:
    store.retain("cedar", "key-1", payload, "marker-2")
except MemoryConflict:
    print("conflict: same key, different marker")

rows = store.recall_marker("cedar", "marker-1")
assert [r["key"] for r in rows] == ["key-1"]
```

Expected: a `MemoryConflict` is raised, because the marker is part of the identity check. `recall_marker` returns the one row written under `marker-1`.

---

## Step 5 — the Bedrock adapter with a fake client (5 min, optional)

**The idea.** The live path asks a Bedrock model for one advisory and expects a `record_decision` tool use back. With a fake client in place, you can see the success shape and the fail-closed path without an AWS account, a network call, or a bill.

The live path has not been run for this workshop. It needs an AWS account, credentials, and the `boto3` package. A real call may cost money.

Open `bedrock_advisory.py` first. The adapter reads three fields from the tool input: `value` (a boolean), `probabilityTrue` (0 to 1) and `verdict` (`act` or `review`). The fake below returns exactly those.

```python
from bedrock_advisory import bedrock_advisory
from agent_memory_demo import ProviderUnavailable

class FakeClient:
    calls = []
    def converse(self, **kwargs):
        self.calls.append(kwargs)
        tool_input = {"value": True, "probabilityTrue": 0.9, "verdict": "act"}
        return {"stopReason": "tool_use", "output": {"message": {"content": [
            {"toolUse": {"name": "record_decision", "input": tool_input}}]}}}
```

```python
fake = FakeClient()
result = bedrock_advisory("Should this be escalated?", "example.synthetic-model-v1", client=fake)
assert result["execution_authority"] == "none"
assert result["route"] == "propose_allow"
print(result)
print(list(fake.calls[0]))
```

Expected: the asserts pass. The first print is the proposal:

```
{'probability_true': 0.9, 'mode': 'live', 'route': 'propose_allow', 'execution_authority': 'none', 'provider': 'amazon-bedrock', 'provider_model': 'example.synthetic-model-v1'}
```

The second print lists the arguments the adapter passed to `converse`: `['modelId', 'system', 'messages', 'toolConfig', 'inferenceConfig']`. The request forces the `record_decision` tool, and the fake never touches AWS or `boto3`. The authority is still `none`; a live model does not get to grant permission. If the tool input says `verdict` is `review`, the route becomes `review` even at high probability.

Now the fail-closed case. A response that ends the turn instead of using the tool:

```python
class EndTurnClient:
    def converse(self, **kwargs):
        return {"stopReason": "end_turn", "output": {"message": {"content": []}}}

try:
    bedrock_advisory("Should this be escalated?", "example.synthetic-model-v1", client=EndTurnClient())
except ProviderUnavailable as exc:
    print("ProviderUnavailable:", exc)
else:
    raise AssertionError("expected ProviderUnavailable")
```

Expected: it prints `ProviderUnavailable: Bedrock unavailable or response contract invalid`. The caller must surface review, and the provider's own error text is not copied into the message.

### Try this

Return a tool use with the wrong tool name:

```python
class WrongToolClient:
    def converse(self, **kwargs):
        return {"stopReason": "tool_use",
                "output": {"message": {"content": [
                    {"toolUse": {"name": "other_tool", "input": {}}}]}}}

try:
    bedrock_advisory("Should this be escalated?", "example.synthetic-model-v1", client=WrongToolClient())
except ProviderUnavailable as exc:
    print("ProviderUnavailable:", exc)
else:
    raise AssertionError("expected ProviderUnavailable")
```

Expected: the same `ProviderUnavailable` message again. A broken response contract is treated the same way as an unavailable provider.

---

## Wrap-up

Each contract blocks one specific failure:

- **Source freshness** stops a stale index entry from being served as if it were the current text. Without it, the answer looks current but was hashed against older content.
- **Project isolation** stops text from one project from reaching another project's answer. Without it, a request for Cedar can return Maple content.
- **Typed review** stops a confidence number from becoming an action. Without it, `0.8` quietly turns into "run it". The proposal still needs a human or an outer step, and `execution_authority` stays `none`.
- **Idempotent retention** stops duplicate writes and silent overwrites. Without it, a retry creates a second row, or a reused key replaces data that was already recorded.

The data in `fixtures/sources.json` is synthetic. No accuracy, quality or coverage claim is made about it.

If something in this workshop did not match the code, open an issue:
https://github.com/lightcloud00/agent-memory-decision-demo/issues/new
