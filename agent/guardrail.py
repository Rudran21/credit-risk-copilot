"""
guardrail.py

NUMERIC GROUNDING CHECK.

Extracts every figure from the agent's prose and verifies each one traces back
to a value some tool actually returned. Anything that does not is flagged.

WHAT IT HANDLES
  display scaling      "$771.4m"  <-  771423886.04
  percent conversion   "91.8%"    <-  0.918
  thousands separators "744,779"  <-  744779
  rounding             tolerance scales with the precision written

WHAT IT DOES NOT CATCH, and this is the important part
  Fabricated MECHANISMS. The agent can cite every number correctly and still
  describe machinery that does not exist. Observed in testing: the agent
  claimed exposure "migrates to Stages 2 and 3 under stress". Stage assignment
  in this model is computed once and is scenario-invariant, so nothing
  migrates. Every number in that sentence was grounded. The claim was false.

  Numeric grounding verifies figures, not causal stories. Treat it as a floor,
  not a guarantee. See docs/model_validation_note.md.

  It also allowlists structurally unavoidable integers (stage numbers, month
  counts, "IFRS 9"), so a fabricated figure that happens to equal one of those
  will pass. Documented limitation.
"""

import re

from ecl import config as C
from ecl import schema as S

# Integers that appear structurally in any IFRS 9 discussion and cannot be
# meaningfully checked: stage labels, "IFRS 9", horizons, contractual terms.
_STRUCTURAL = {0.0, 1.0, 2.0, 3.0, 9.0, 12.0, 36.0, 60.0}


def _declared_constants() -> set:
    """
    Every numeric constant declared in ecl/config.py, plus the DPD band edges
    from the schema.

    THE RULE THIS ENFORCES: a figure must trace to a tool call OR to a model
    parameter declared in config. Both are auditable. The system prompt
    legitimately cites thresholds and DPD bands when explaining a stage
    assignment, and those are model constants rather than portfolio figures.

    Harvesting programmatically means the allowlist follows config
    automatically and cannot drift out of sync with it.
    """
    out = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            out.add(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    for name in dir(C):
        if name.startswith("_"):
            continue
        walk(getattr(C, name))
    walk(S.DPD_BAND)
    return out


ALLOWLIST = _STRUCTURAL | _declared_constants()

CITATION_RE = re.compile(r"\[c\d+\]")
NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(\$?\s?-?\d[\d,]*(?:\.\d+)?)"
    r"\s*"
    r"(%|bn|billion|mn|m\b|million|k\b|thousand)?",
    re.IGNORECASE,
)

SUFFIX_SCALE = {
    "bn": 1e9, "billion": 1e9,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "k": 1e3, "thousand": 1e3,
}


def _decimals(numeral: str) -> int:
    return len(numeral.split(".")[1]) if "." in numeral else 0


def extract_figures(text: str) -> list:
    """Every numeric mention, with the candidate real values it could denote."""
    stripped = CITATION_RE.sub(" ", text)
    stripped = stripped.replace("*", " ").replace("`", " ")

    figures = []
    for match in NUMBER_RE.finditer(stripped):
        raw, suffix = match.group(1), (match.group(2) or "").lower()
        numeral = raw.replace("$", "").replace(",", "").strip()
        if not numeral or numeral in {"-", "."}:
            continue
        try:
            value = float(numeral)
        except ValueError:
            continue

        dp = _decimals(numeral)
        candidates = []

        if suffix == "%":
            # a percent may be stored raw (8.11) or as a fraction (0.0811)
            candidates.append((value, 0.5 * 10 ** -dp))
            candidates.append((value / 100.0, 0.5 * 10 ** -(dp + 2)))
        elif suffix in SUFFIX_SCALE:
            s = SUFFIX_SCALE[suffix]
            candidates.append((value * s, 0.5 * 10 ** -dp * s))
        else:
            candidates.append((value, 0.5 * 10 ** -dp))
            # a bare number may still be a percent the model wrote without a sign
            candidates.append((value / 100.0, 0.5 * 10 ** -(dp + 2)))

        figures.append({
            "text": match.group(0).strip(),
            "value": value,
            "suffix": suffix,
            "candidates": candidates,
            "position": match.start(),
        })
    return figures


_ALLOW_PCT = {round(v * 100, 6) for v in ALLOWLIST if 0 < abs(v) < 10}


def _grounded(figure: dict, grounded_values) -> bool:
    if figure["value"] in ALLOWLIST or figure["value"] in _ALLOW_PCT:
        return True
    if 1900 <= figure["value"] <= 2100 and figure["value"].is_integer():
        return True                                   # a year, not a quantity
    for cand, tol in figure["candidates"]:
        tol = max(tol, abs(cand) * 1e-9)
        for g in grounded_values:
            if abs(g - cand) <= tol:
                return True
    return False


def check(answer: str, session) -> dict:
    """
    Verify an agent answer against the tool calls that produced it.

    Returns
    -------
    dict with:
        passed              no ungrounded figures
        n_figures           figures found in the prose
        n_ungrounded        how many failed
        ungrounded          the offending figures with context
        citation_coverage   share of figures carrying a [c#] tag nearby
        n_tool_calls        calls made
    """
    grounded_values = list(session.grounded_numbers)
    figures = extract_figures(answer)

    ungrounded = []
    for f in figures:
        if _grounded(f, grounded_values):
            continue
        lo = max(0, f["position"] - 60)
        ungrounded.append({
            "text": f["text"],
            "value": f["value"],
            "context": "..." + answer[lo:f["position"] + 40].replace("\n", " ") + "...",
        })

    # citation coverage: is there a [c#] within 40 characters after the figure?
    cited = 0
    for f in figures:
        window = answer[f["position"]:f["position"] + 60]
        if CITATION_RE.search(window):
            cited += 1

    return {
        "passed": len(ungrounded) == 0,
        "n_figures": len(figures),
        "n_ungrounded": len(ungrounded),
        "ungrounded": ungrounded,
        "citation_coverage": round(cited / len(figures), 4) if figures else 1.0,
        "n_tool_calls": len(session.calls),
        "grounded_pool_size": len(set(grounded_values)),
    }


def format_report(result: dict) -> str:
    if result["passed"]:
        return (f"GROUNDED  {result['n_figures']} figures, all traced to "
                f"{result['n_tool_calls']} tool calls, "
                f"citations on {result['citation_coverage']:.0%}")
    lines = [f"UNGROUNDED  {result['n_ungrounded']} of {result['n_figures']} "
             f"figures could not be traced:"]
    for u in result["ungrounded"]:
        lines.append(f"    {u['text']}   in: {u['context']}")
    return "\n".join(lines)
