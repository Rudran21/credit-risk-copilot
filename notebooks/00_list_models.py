"""
00_list_models.py

Lists the Gemini models this API key can actually reach, and probes each
candidate with a one-token request to see which ones respond.

Free-tier quotas differ enormously by model. gemini-3.6-flash turned out to be
20 requests per DAY, which cannot run a 15-question eval. Flash-Lite variants
are normally far more generous. This script finds out rather than guessing.

Usage:
    python notebooks\\00_list_models.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set in .env")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=key)

    print("--- MODELS VISIBLE TO THIS KEY ---")
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        name = m.name.replace("models/", "")
        names.append(name)
        print(f"  {name}")

    print(f"\n  {len(names)} models support generateContent")

    # Probe the cheapest-looking candidates with a trivial request.
    print("\n--- PROBE (1 token each, cheapest candidates first) ---")
    prefer = [n for n in names if "lite" in n.lower()]
    prefer += [n for n in names if "flash" in n.lower() and n not in prefer]
    prefer = prefer[:8]

    if not prefer:
        print("  no flash/lite models found; probing first 5")
        prefer = names[:5]

    working = []
    for name in prefer:
        try:
            client.models.generate_content(
                model=name, contents="hi",
                config={"max_output_tokens": 1})
            print(f"  OK        {name}")
            working.append(name)
        except Exception as exc:                       # noqa: BLE001
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                quota = ""
                if "quotaValue" in msg:
                    import re
                    hit = re.search(r"'quotaValue':\s*'(\d+)'", msg)
                    if hit:
                        quota = f"  (quota {hit.group(1)})"
                print(f"  QUOTA     {name}{quota}")
            elif "404" in msg:
                print(f"  NOT FOUND {name}")
            elif "PERMISSION" in msg or "403" in msg:
                print(f"  NO ACCESS {name}")
            else:
                print(f"  ERROR     {name}: {msg[:80]}")

    print("\n--- RECOMMENDATION ---")
    if working:
        print(f"  Set this in .env:\n    ECL_AGENT_MODEL={working[0]}")
        if len(working) > 1:
            print(f"  Fallbacks: {', '.join(working[1:])}")
    else:
        print("  Nothing responded. Either the daily quota is exhausted across")
        print("  all models (wait for reset, usually midnight Pacific), or")
        print("  switch LLM_PROVIDER=anthropic in .env.")

    print("\n  Live per-model quotas: https://ai.dev/rate-limit")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
