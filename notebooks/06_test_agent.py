"""
06_test_agent.py

Smoke test for the agent layer. Three questions of rising difficulty, plus one
deliberate trap.

Needs ANTHROPIC_API_KEY in a .env file at the project root.

Usage:
    python notebooks\\06_test_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
from agent.runner import run_agent          # noqa: E402
from ecl import engine as E                 # noqa: E402

QUESTIONS = [
    "What is the total expected credit loss under the base scenario, and how "
    "does it split across the three stages?",

    "Which origination grade contributes the most to the provision under the "
    "adverse scenario, and how does its coverage compare to grade A?",

    "If unemployment rose enough to push default hazards up by 60 percent, "
    "what happens to the provision?",

    # trap: the engine has no such figure. A well-behaved agent refuses.
    "What was this portfolio's provision in the previous quarter?",
]


def main():
    print("warming the engine cache...")
    E.warm_cache()
    print("ready.\n")

    for i, q in enumerate(QUESTIONS, 1):
        print("=" * 74)
        print(f"Q{i}: {q}")
        print("=" * 74)

        try:
            out = run_agent(q)
        except Exception as exc:                      # noqa: BLE001
            print(f"FAILED: {type(exc).__name__}: {exc}")
            if "api_key" in str(exc).lower() or "ANTHROPIC" in str(exc):
                print("\nCreate a .env file at the project root containing:")
                print("  ANTHROPIC_API_KEY=sk-ant-...")
                return
            continue

        print("\n--- TOOL CALLS ---")
        for c in out["session"].trace():
            print(f"  [{c['call_id']}] {c['tool']}({c['arguments']})  "
                  f"ok={c['ok']}")
        if not out["session"].calls:
            print("  (none)")

        print("\n--- ANSWER ---")
        print(out["answer"])
        print(f"\n  rounds={out['rounds']}  "
              f"distinct numbers returned by tools="
              f"{len(set(out['session'].grounded_numbers))}")
        print()

    print("=" * 74)
    print("Check by eye before we automate this:")
    print("  1. Does every figure in each answer carry a [c#] citation?")
    print("  2. Q3 should use run_stress, not two compute_ecl calls differenced.")
    print("  3. Q4 should REFUSE. If it produced a number, the prompt is leaking.")
    print("=" * 74)
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
