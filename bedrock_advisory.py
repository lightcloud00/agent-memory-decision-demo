"""Optional Amazon Bedrock adapter for the agent-memory decision demo.

The adapter asks a Bedrock model for one structured advisory proposal and returns the
same review-only shape as the other adapters in this demo. A proposal is never permission
to execute anything: `execution_authority` stays "none" and the route may only be
proposed, not performed.

Boto3 is imported lazily, so this module imports and its tests run without boto3 and
without a network. No credential is read, printed, logged or accepted as an argument;
AWS credentials come from the standard AWS credential chain inside boto3.

The live service has not been called from this adapter yet.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from agent_memory_demo import ProviderUnavailable, decision

SYSTEM_TEXT = (
    "You are an advisory decision service for a synthetic demo. Answer only by calling "
    "the record_decision tool exactly once. The tool records a proposal, never an action. "
    "Report a probability between zero and one that the proposal is correct, and use the "
    "verdict review whenever the evidence is weak or ambiguous."
)

TOOL_DESCRIPTION = (
    "Record one advisory decision proposal. Advisory only: it never grants execution "
    "authority and never runs a tool."
)

RECORD_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {
            "type": "boolean",
            "description": "Proposed boolean answer to the advisory question.",
        },
        "probabilityTrue": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Probability between zero and one that the proposal is correct.",
        },
        "verdict": {
            "type": "string",
            "enum": ["act", "review"],
            "description": "act when the proposal is clear, review when a human should look.",
        },
    },
    "required": ["value", "probabilityTrue", "verdict"],
    "additionalProperties": False,
}


def _make_client(region: str | None) -> Any:
    """Create a bedrock-runtime client. Credentials come from the AWS chain, not from us."""
    try:
        import boto3  # lazy import: the module must work without boto3 installed
    except ImportError as exc:
        raise ProviderUnavailable("boto3 not installed") from exc
    return boto3.client("bedrock-runtime", region_name=region)


def _extract_tool_input(response: Any) -> dict[str, Any]:
    """Return the record_decision tool input, or raise on a broken response contract."""
    if not isinstance(response, dict):
        raise TypeError("response must be a mapping")
    if response.get("stopReason") != "tool_use":
        raise ValueError("stopReason must be tool_use")
    content = response["output"]["message"]["content"]
    if not isinstance(content, list):
        raise TypeError("content must be a list")
    for item in content:
        if not isinstance(item, dict):
            continue
        tool_use = item.get("toolUse")
        if isinstance(tool_use, dict) and tool_use.get("name") == "record_decision":
            tool_input = tool_use.get("input")
            if not isinstance(tool_input, dict):
                raise TypeError("tool input must be a mapping")
            return tool_input
    raise ValueError("record_decision tool use not found")


def bedrock_advisory(
    question: str,
    model_id: str,
    client: Any = None,
    region: str | None = None,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Ask a Bedrock model for one advisory proposal.

    Returns the shared proposal dict. Raises ProviderUnavailable when the provider is
    missing or the response contract is invalid. The provider's own error text is never
    placed in the message.
    """
    try:
        if client is None:
            client = _make_client(region)
        response = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_TEXT}],
            messages=[{"role": "user", "content": [{"text": question}]}],
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "record_decision",
                            "description": TOOL_DESCRIPTION,
                            "inputSchema": {"json": RECORD_DECISION_SCHEMA},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": "record_decision"}},
            },
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        tool_input = _extract_tool_input(response)
        value = tool_input["value"]
        if type(value) is not bool:
            raise ValueError("value must be a boolean")
        verdict = tool_input["verdict"]
        if verdict not in {"act", "review"}:
            raise ValueError("verdict must be act or review")
        # decision() rejects bad probabilities, including booleans, with ValueError.
        proposal = decision(tool_input["probabilityTrue"], "live", True)
        if verdict == "review":
            proposal["route"] = "review"
        proposal["provider"] = "amazon-bedrock"
        proposal["provider_model"] = model_id
        usage = response.get("usage")
        if isinstance(usage, dict):
            copied = {
                key: usage[key]
                for key in ("inputTokens", "outputTokens")
                if key in usage
            }
            if copied:
                proposal["usage"] = copied
        return proposal
    except ProviderUnavailable:
        raise
    except Exception as exc:
        raise ProviderUnavailable(
            "Bedrock unavailable or response contract invalid"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask a Bedrock model for one advisory proposal."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Bedrock model ID or inference profile ID that you have access to in your region",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region for bedrock-runtime; the standard region chain is used when omitted",
    )
    parser.add_argument("question", help="Advisory question")
    args = parser.parse_args(argv)
    try:
        proposal = bedrock_advisory(args.question, args.model_id, region=args.region)
    except ProviderUnavailable:
        print(json.dumps({"status": "unavailable", "route": "review"}))
        return 0
    print(json.dumps(proposal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
