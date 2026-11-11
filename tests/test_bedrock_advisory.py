"""Tests for the optional Amazon Bedrock adapter.

No network access and no boto3 install are required. The live test is opt-in and is
skipped unless AMD_LIVE_BEDROCK=1 and BEDROCK_MODEL_ID are both set.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory_demo import ProviderUnavailable
from bedrock_advisory import bedrock_advisory

MODEL_ID = "example.synthetic-model-v1"  # never called; FakeClient only


def tool_use_response(value=True, probability=0.93, verdict="act", stop_reason="tool_use"):
    """Build a synthetic Converse response that used the tool."""
    return {
        "stopReason": stop_reason,
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Recording one proposal."},
                    {
                        "toolUse": {
                            "name": "record_decision",
                            "input": {
                                "value": value,
                                "probabilityTrue": probability,
                                "verdict": verdict,
                            },
                        }
                    },
                ],
            }
        },
        "usage": {"inputTokens": 41, "outputTokens": 12},
    }


class FakeClient:
    """Records the kwargs of converse() and returns a canned response or raises."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def kwargs(self):
        return self.calls[-1]


class BedrockAdvisoryTests(unittest.TestCase):
    def test_act_proposal_carries_provider_fields(self):
        client = FakeClient(tool_use_response())
        result = bedrock_advisory("Should the synthetic demo act?", MODEL_ID, client=client)
        self.assertEqual(result["route"], "propose_allow")
        self.assertEqual(result["execution_authority"], "none")
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["probability_true"], 0.93)
        self.assertEqual(result["provider"], "amazon-bedrock")
        self.assertEqual(result["provider_model"], MODEL_ID)
        self.assertEqual(result["usage"], {"inputTokens": 41, "outputTokens": 12})

    def test_review_verdict_overrides_high_probability(self):
        client = FakeClient(tool_use_response(probability=0.95, verdict="review"))
        result = bedrock_advisory("q", MODEL_ID, client=client)
        self.assertEqual(result["route"], "review")
        self.assertEqual(result["execution_authority"], "none")

    def test_request_shape_forces_the_record_decision_tool(self):
        client = FakeClient(tool_use_response())
        bedrock_advisory("q", MODEL_ID, client=client)
        kwargs = client.kwargs
        self.assertEqual(kwargs["modelId"], MODEL_ID)
        self.assertEqual(kwargs["inferenceConfig"], {"maxTokens": 300, "temperature": 0})
        self.assertEqual(
            kwargs["toolConfig"]["toolChoice"], {"tool": {"name": "record_decision"}}
        )
        tool_spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        self.assertEqual(tool_spec["name"], "record_decision")
        schema = tool_spec["inputSchema"]["json"]
        self.assertEqual(schema["required"], ["value", "probabilityTrue", "verdict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["value"]["type"], "boolean")
        self.assertEqual(schema["properties"]["probabilityTrue"]["type"], "number")
        self.assertEqual(schema["properties"]["verdict"]["enum"], ["act", "review"])
        self.assertEqual(
            kwargs["messages"], [{"role": "user", "content": [{"text": "q"}]}]
        )
        self.assertEqual(len(kwargs["system"]), 1)
        self.assertIn("text", kwargs["system"][0])

    def test_no_key_like_argument_names_in_the_request(self):
        client = FakeClient(tool_use_response())
        bedrock_advisory("q", MODEL_ID, client=client)
        forbidden = ("key", "secret", "token", "password", "credential", "auth")
        for name in client.kwargs:
            for word in forbidden:
                self.assertNotIn(word, name.lower(), name)

    def test_signature_accepts_no_credential_argument(self):
        names = list(inspect.signature(bedrock_advisory).parameters)
        self.assertEqual(names, ["question", "model_id", "client", "region", "max_tokens"])
        # "max_tokens" is a size limit, so match credential names, not the bare word "token".
        forbidden = ("access_key", "api_key", "secret", "session_token", "password", "credential")
        for name in names:
            for word in forbidden:
                self.assertNotIn(word, name.lower(), name)

    def test_wrong_stop_reason_is_rejected(self):
        client = FakeClient(tool_use_response(stop_reason="end_turn"))
        with self.assertRaises(ProviderUnavailable):
            bedrock_advisory("q", MODEL_ID, client=client)

    def test_missing_tool_use_is_rejected(self):
        response = tool_use_response()
        response["output"]["message"]["content"] = [{"text": "No tool call here."}]
        with self.assertRaises(ProviderUnavailable):
            bedrock_advisory("q", MODEL_ID, client=FakeClient(response))

    def test_string_value_is_rejected(self):
        with self.assertRaises(ProviderUnavailable):
            bedrock_advisory("q", MODEL_ID, client=FakeClient(tool_use_response(value="true")))

    def test_boolean_probability_is_rejected(self):
        with self.assertRaises(ProviderUnavailable):
            bedrock_advisory(
                "q", MODEL_ID, client=FakeClient(tool_use_response(probability=True))
            )

    def test_out_of_range_probability_is_rejected(self):
        with self.assertRaises(ProviderUnavailable):
            bedrock_advisory(
                "q", MODEL_ID, client=FakeClient(tool_use_response(probability=1.5))
            )

    def test_provider_error_text_is_not_exposed(self):
        client = FakeClient(error=RuntimeError("secret detail"))
        with self.assertRaises(ProviderUnavailable) as caught:
            bedrock_advisory("q", MODEL_ID, client=client)
        self.assertNotIn("secret detail", str(caught.exception))
        self.assertEqual(
            str(caught.exception), "Bedrock unavailable or response contract invalid"
        )

    def test_missing_boto3_is_reported(self):
        with patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaises(ProviderUnavailable) as caught:
                bedrock_advisory("q", MODEL_ID)
        self.assertIn("boto3", str(caught.exception))


@unittest.skipUnless(
    os.environ.get("AMD_LIVE_BEDROCK") == "1" and bool(os.environ.get("BEDROCK_MODEL_ID")),
    "live Bedrock test is opt-in and off by default",
)
class LiveBedrockTests(unittest.TestCase):
    def test_live_call_returns_a_review_only_proposal(self):
        model_id = os.environ["BEDROCK_MODEL_ID"]
        result = bedrock_advisory(
            "Synthetic advisory question for the agent-memory demo. Advisory only.",
            model_id,
        )
        self.assertEqual(result["execution_authority"], "none")
        self.assertEqual(result["provider"], "amazon-bedrock")
        self.assertIn(result["route"], {"propose_allow", "propose_deny", "review"})


if __name__ == "__main__":
    unittest.main()
