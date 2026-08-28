"""
runner.py

The tool-use loop. Sends the question to an LLM, executes whatever tools it
asks for, feeds results back, repeats until it produces prose.

Returns the answer AND the full tool trace, because the trace is what the
guardrail checks and what the app displays.

PROVIDER SWITCH
---------------
Set LLM_PROVIDER in .env:

    LLM_PROVIDER=gemini        (free tier, no credit card)
    GEMINI_API_KEY=...         from aistudio.google.com/apikey

    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...

Everything downstream (app, evals, guardrail) is provider-agnostic. Only this
file knows which API is being used.
"""

import json
import os
import time

from dotenv import load_dotenv

from agent import prompts
from agent.tools import TOOL_SCHEMAS, ToolSession

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
MAX_TOKENS = 2000
MAX_TOOL_ROUNDS = 8

# Free-tier rate limiting. Gemini free tier is roughly 10-15 requests/minute
# depending on model, so pace requests rather than collecting 429s mid-eval.
THROTTLE_SECONDS = float(os.getenv("LLM_THROTTLE_SECONDS",
                                   "4.5" if PROVIDER == "gemini" else "0"))
MAX_RETRIES = 4

# Model strings drift. If a call 404s, check the provider's docs and override
# in .env with ECL_AGENT_MODEL.
DEFAULT_MODEL = {
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-sonnet-5",
}
MODEL = os.getenv("ECL_AGENT_MODEL",
                  DEFAULT_MODEL.get(PROVIDER, "gemini-3.6-flash"))

_last_call_at = [0.0]


def _throttle():
    if THROTTLE_SECONDS <= 0:
        return
    wait = THROTTLE_SECONDS - (time.time() - _last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_at[0] = time.time()


def _is_rate_limit(exc) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return ("429" in s or "resource_exhausted" in s or "rate limit" in s
            or "quota" in s or "overloaded" in s)


def _with_retry(fn):
    """Exponential backoff on rate limits. Free tiers need this."""
    for attempt in range(MAX_RETRIES):
        try:
            _throttle()
            return fn()
        except Exception as exc:                      # noqa: BLE001
            if not _is_rate_limit(exc) or attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  [rate limited, waiting {wait}s]")
            time.sleep(wait)


# =============================================================================
# schema conversion
# =============================================================================
_GEMINI_DROP = {"additionalProperties", "$schema", "examples", "default"}


def _to_gemini_schema(node):
    """Gemini accepts an OpenAPI subset. Strip keys it rejects."""
    if isinstance(node, dict):
        return {k: _to_gemini_schema(v) for k, v in node.items()
                if k not in _GEMINI_DROP}
    if isinstance(node, list):
        return [_to_gemini_schema(v) for v in node]
    return node


def _gemini_declarations():
    return [{"name": t["name"],
             "description": t["description"],
             "parameters": _to_gemini_schema(t["input_schema"])}
            for t in TOOL_SCHEMAS]


# =============================================================================
# GEMINI
# =============================================================================
def _run_gemini(user_message, history, system, max_rounds):
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at "
            "https://aistudio.google.com/apikey and put it in .env as:\n"
            "  LLM_PROVIDER=gemini\n  GEMINI_API_KEY=...")

    client = genai.Client(api_key=key)
    session = ToolSession()

    contents = list(history or [])
    contents.append(types.Content(role="user",
                                  parts=[types.Part(text=user_message)]))

    cfg = types.GenerateContentConfig(
        system_instruction=system or prompts.SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=_gemini_declarations())],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True),
        max_output_tokens=MAX_TOKENS,
        temperature=0.0,
    )

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        resp = _with_retry(lambda: client.models.generate_content(
            model=MODEL, contents=contents, config=cfg))

        cand = resp.candidates[0]
        contents.append(cand.content)

        parts = cand.content.parts or []
        calls = [p.function_call for p in parts
                 if getattr(p, "function_call", None)]

        if not calls:
            text = "".join(p.text for p in parts if getattr(p, "text", None))
            return {"answer": text.strip(), "session": session,
                    "messages": contents, "rounds": rounds, "stopped": False}

        responses = []
        for fc in calls:
            result = session.call(fc.name, dict(fc.args or {}))
            call_id = session.calls[-1]["call_id"]
            responses.append(types.Part.from_function_response(
                name=fc.name,
                response={"json": json.dumps({"call_id": call_id, **result},
                                             default=str)}))
        contents.append(types.Content(role="user", parts=responses))

    return {"answer": ("Stopped after the tool-call limit without reaching an "
                       "answer. Try a narrower question."),
            "session": session, "messages": contents,
            "rounds": rounds, "stopped": True}


# =============================================================================
# ANTHROPIC
# =============================================================================
def _run_anthropic(user_message, history, system, max_rounds):
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=key)
    session = ToolSession()

    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        resp = _with_retry(lambda: client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system or prompts.SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages))

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", None) == "text")
            return {"answer": text.strip(), "session": session,
                    "messages": messages, "rounds": rounds, "stopped": False}

        results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result = session.call(block.name, block.input or {})
            call_id = session.calls[-1]["call_id"]
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps({"call_id": call_id, **result},
                                                  default=str)})
        messages.append({"role": "user", "content": results})

    return {"answer": ("Stopped after the tool-call limit without reaching an "
                       "answer. Try a narrower question."),
            "session": session, "messages": messages,
            "rounds": rounds, "stopped": True}


# =============================================================================
# public entrypoint
# =============================================================================
def run_agent(user_message: str, history: list | None = None,
              system: str | None = None, max_rounds: int = MAX_TOOL_ROUNDS):
    """
    Answer one question, calling engine tools as needed.

    Returns dict with:
        answer   final prose
        session  ToolSession holding every call and its numeric results
        messages conversation so far, pass back as history next turn
        rounds   tool rounds used
        stopped  True if the round limit was hit first
    """
    if PROVIDER == "gemini":
        return _run_gemini(user_message, history, system, max_rounds)
    if PROVIDER == "anthropic":
        return _run_anthropic(user_message, history, system, max_rounds)
    raise ValueError(f"unknown LLM_PROVIDER '{PROVIDER}'. "
                     f"use 'gemini' or 'anthropic'.")


def draft_memo(history: list | None = None):
    """Credit provision memo built only from tool output."""
    return run_agent(prompts.MEMO_PROMPT, history=history)


def provider_info() -> dict:
    return {"provider": PROVIDER, "model": MODEL,
            "throttle_seconds": THROTTLE_SECONDS}
