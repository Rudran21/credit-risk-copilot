"""
run_eval.py

Runs the golden questions, checks three things per question, and writes
evals/results.md for the README.

CHECKED
  1. Numeric grounding  every figure in the prose traces to a tool call
  2. Answer accuracy    the truth value appears, at any sensible scaling
  3. Tool selection     an expected tool was actually called

Truth values come from calling the engine directly at eval time, so the eval
never drifts out of date when a parameter changes.

CHECKPOINTING
Results are saved after EVERY question. A re-run skips questions already
answered, so a free-tier daily quota can be spread across several sessions
until all 15 are done. Use --fresh to discard and start over.

If a per-DAY quota error is hit, the run aborts immediately rather than
burning backoff time against a quota that will not reset for hours.

Usage:
    python evals\\run_eval.py
    python evals\\run_eval.py --only g01,g14      run a subset
    python evals\\run_eval.py --limit 5           stop after 5 new questions
    python evals\\run_eval.py --fresh             ignore the checkpoint
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.abspath("."))
from agent import guardrail                     # noqa: E402
from agent.runner import provider_info, run_agent  # noqa: E402
from ecl import engine as E                     # noqa: E402

QUESTIONS_PATH = "evals/golden_questions.yaml"
RESULTS_PATH = "evals/results.md"

_FNS = {
    "compute_ecl": E.compute_ecl,
    "get_portfolio_slice": E.get_portfolio_slice,
    "run_stress": E.run_stress,
    "explain_stage_assignment": E.explain_stage_assignment,
}


def resolve_truth(spec: dict):
    """Call the engine to get the expected answer. Never hardcoded."""
    fn = _FNS[spec["fn"]]
    result = fn(**spec.get("args", {}))
    if "error" in result:
        raise RuntimeError(
            f"engine returned an error for the truth call "
            f"{spec['fn']}({spec.get('args')}): {result['error']}")
    node = result
    for part in spec["field"].split("."):
        if part not in node:
            raise RuntimeError(
                f"field '{spec['field']}' not in {spec['fn']} result. "
                f"available at this level: {sorted(node)}")
        node = node[part]
    return float(node)


def answer_contains(answer: str, truth: float) -> bool:
    """Does the prose state this value, at any sensible scaling or rounding?"""
    for fig in guardrail.extract_figures(answer):
        for cand, tol in fig["candidates"]:
            if abs(cand - truth) <= max(tol, abs(truth) * 1e-6):
                return True
    return False


CHECKPOINT_PATH = "evals/.checkpoint.json"


def _is_daily_quota(msg: str) -> bool:
    """A per-day quota will not reset with backoff. Stop instead of retrying."""
    m = msg.lower()
    return ("perday" in m.replace("_", "").replace("-", "")
            or "requestsperday" in m.replace("_", "").replace("-", "")
            or ("429" in m and "day" in m))


def load_checkpoint() -> dict:
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f) if not r.get("error")}
    except Exception:                                   # noqa: BLE001
        return {}


def save_checkpoint(rows) -> None:
    os.makedirs("evals", exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated question ids")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW questions this session")
    ap.add_argument("--fresh", action="store_true",
                    help="discard the checkpoint and re-run everything")
    args = ap.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in keep]

    done = {} if args.fresh else load_checkpoint()
    if done:
        print(f"checkpoint: {len(done)} question(s) already answered "
              f"({', '.join(sorted(done))})")

    info = provider_info()
    print(f"provider={info['provider']}  model={info['model']}  "
          f"throttle={info['throttle_seconds']}s")
    print("warming engine cache...")
    E.warm_cache()
    print(f"running {len(questions)} questions\n")

    rows = []
    t0 = time.time()
    new_this_session = 0

    for q in questions:
        if q["id"] in done:
            rows.append(done[q["id"]])
            continue

        if args.limit is not None and new_this_session >= args.limit:
            print(f"[{q['id']}] skipped, --limit {args.limit} reached")
            continue

        print(f"[{q['id']}] {q['question'][:64]}...")
        new_this_session += 1

        row = {"id": q["id"], "type": q["type"], "question": q["question"].strip()}

        try:
            out = run_agent(q["question"])
        except Exception as exc:                          # noqa: BLE001
            msg = str(exc)
            print(f"    ERROR {type(exc).__name__}: {msg[:120]}")
            if _is_daily_quota(msg):
                print("\n" + "!" * 66)
                print("  DAILY quota exhausted for this model.")
                print("  Progress so far is checkpointed. Options:")
                print("    - wait for reset (usually midnight Pacific), re-run")
                print("    - python notebooks\\00_list_models.py  to find a")
                print("      model with a higher daily limit")
                print("    - set LLM_PROVIDER=anthropic in .env")
                print("!" * 66 + "\n")
                break
            row.update({"error": f"{type(exc).__name__}: {msg[:300]}",
                        "grounded": False, "correct": False, "tool_ok": False})
            rows.append(row)
            save_checkpoint(rows)
            continue

        answer, session = out["answer"], out["session"]
        g = guardrail.check(answer, session)
        tools_called = [c["tool"] for c in session.calls]

        row.update({
            "answer": answer,
            "tools_called": tools_called,
            "rounds": out["rounds"],
            "grounded": g["passed"],
            "n_figures": g["n_figures"],
            "n_ungrounded": g["n_ungrounded"],
            "ungrounded": g["ungrounded"],
            "citation_coverage": g["citation_coverage"],
        })

        if q["type"] == "refusal":
            row["correct"] = g["n_figures"] == 0 or g["n_ungrounded"] == 0
            row["strict_refusal"] = g["n_figures"] == 0
            row["tool_ok"] = True
            row["truth"] = None

        elif q["type"] == "tool":
            row["correct"] = any(t in q["expect_tools"] for t in tools_called)
            row["tool_ok"] = row["correct"]
            row["truth"] = None

        else:
            try:
                truth = resolve_truth(q["truth"])
            except Exception as exc:                      # noqa: BLE001
                print(f"    TRUTH ERROR: {exc}")
                row.update({"error": f"truth resolution failed: {exc}",
                            "correct": False, "tool_ok": False})
                rows.append(row)
                save_checkpoint(rows)
                continue
            row["truth"] = truth
            row["correct"] = answer_contains(answer, truth)
            row["tool_ok"] = any(t in q["expect_tools"] for t in tools_called)

        flag = "ok " if (row["correct"] and row["grounded"]) else "FAIL"
        print(f"    {flag}  correct={row['correct']}  grounded={row['grounded']}"
              f"  tools={tools_called}")
        for u in row.get("ungrounded", []):
            print(f"      ungrounded: {u['text']}")

        rows.append(row)
        save_checkpoint(rows)

    elapsed = time.time() - t0
    write_results([r for r in rows if not r.get("error")], info, elapsed)
    summarise(rows, elapsed, total_questions=len(questions))


def summarise(rows, elapsed, total_questions=None):
    attempted = [r for r in rows if not r.get("error")]
    n = len(attempted)
    rows = attempted
    correct = sum(1 for r in rows if r.get("correct"))
    grounded = sum(1 for r in rows if r.get("grounded"))
    tool_ok = sum(1 for r in rows if r.get("tool_ok"))
    figs = sum(r.get("n_figures", 0) for r in rows)
    ung = sum(r.get("n_ungrounded", 0) for r in rows)

    print("\n" + "=" * 66)
    if total_questions and n < total_questions:
        print(f"  PARTIAL RUN: {n} of {total_questions} questions answered.")
        print(f"  Re-run to continue from the checkpoint.")
        print("-" * 66)
    if n == 0:
        print("  no questions completed")
        print("=" * 66)
        return
    print(f"  answer accuracy      {correct}/{n}  ({correct/n:.0%})")
    print(f"  numeric grounding    {grounded}/{n}  ({grounded/n:.0%})")
    print(f"  tool selection       {tool_ok}/{n}  ({tool_ok/n:.0%})")
    print(f"  figures checked      {figs}, ungrounded {ung}")
    print(f"  runtime              {elapsed:.0f}s")
    print("=" * 66)
    print(f"  written to {RESULTS_PATH}")
    print("\n=== DONE - paste this whole output into the chat ===")


def write_results(rows, info, elapsed):
    n = len(rows)
    if n == 0:
        return
    correct = sum(1 for r in rows if r.get("correct"))
    grounded = sum(1 for r in rows if r.get("grounded"))
    tool_ok = sum(1 for r in rows if r.get("tool_ok"))
    figs = sum(r.get("n_figures", 0) for r in rows)
    ung = sum(r.get("n_ungrounded", 0) for r in rows)

    L = []
    L.append("# Evaluation results\n")
    L.append(f"Run {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
             f"on `{info['model']}` via {info['provider']}. "
             f"{n} golden questions, {elapsed:.0f}s.\n")
    L.append("Truth values are computed by calling the engine at eval time, "
             "not hardcoded, so the eval cannot drift out of date when a "
             "model parameter changes.\n")

    L.append("## Headline\n")
    L.append("| metric | result |")
    L.append("|---|---|")
    L.append(f"| Answer accuracy | {correct}/{n} ({correct/n:.0%}) |")
    L.append(f"| Numeric grounding | {grounded}/{n} ({grounded/n:.0%}) |")
    L.append(f"| Tool selection | {tool_ok}/{n} ({tool_ok/n:.0%}) |")
    L.append(f"| Figures checked | {figs} |")
    L.append(f"| Ungrounded figures | {ung} |\n")

    L.append("## Per question\n")
    L.append("| id | type | correct | grounded | figures | tools called |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['id']} | {r['type']} | "
                 f"{'yes' if r.get('correct') else 'NO'} | "
                 f"{'yes' if r.get('grounded') else 'NO'} | "
                 f"{r.get('n_figures', 0)} | "
                 f"{', '.join(r.get('tools_called', [])) or '-'} |")

    fails = [r for r in rows if not (r.get("correct") and r.get("grounded"))]
    if fails:
        L.append("\n## Failures\n")
        for r in fails:
            L.append(f"### {r['id']}\n")
            L.append(f"**Question.** {r['question']}\n")
            if r.get("error"):
                L.append(f"**Error.** `{r['error']}`\n")
                continue
            if r.get("truth") is not None:
                L.append(f"**Expected value.** `{r['truth']:,.2f}`\n")
            for u in r.get("ungrounded", []):
                L.append(f"**Ungrounded figure.** `{u['text']}` "
                         f"in: {u['context']}\n")
            L.append(f"**Answer.**\n\n> {r.get('answer','')[:600]}\n")
    else:
        L.append("\nNo failures on this run.\n")

    L.append("\n## What this eval does not test\n")
    L.append("The grounding check verifies **figures**, not **mechanisms**. "
             "During development the agent produced a fully cited answer that "
             "described exposure migrating between IFRS 9 stages under stress. "
             "Stage assignment in this model is computed once and is "
             "scenario-invariant, so no migration occurs. Every number was "
             "grounded; the causal claim was fabricated. A numeric guardrail "
             "cannot catch that class of error.\n")
    L.append("The check also allowlists structurally unavoidable integers "
             "(stage numbers, month horizons, \"IFRS 9\"). A fabricated figure "
             "equal to one of those would pass.\n")

    os.makedirs("evals", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open("evals/results_raw.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


if __name__ == "__main__":
    main()
