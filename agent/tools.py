"""
tools.py

The bridge between the LLM and the deterministic engine.

Two jobs:
  1. Expose the four engine functions as Anthropic tool schemas.
  2. Record EVERY numeric value returned by every call, so the guardrail can
     later verify that each figure in the agent's prose traces back to one.

This module does no arithmetic. It calls the engine and passes results through
unchanged. If you find yourself computing a ratio or a percentage in here,
that belongs in ecl/engine.py instead.
"""

import re
import uuid

from ecl import engine as E

# =============================================================================
# tool schemas
# =============================================================================
_FILTER_SCHEMA = {
    "type": "object",
    "description": ("Optional slice of the book. Omit for the whole portfolio. "
                    "Every value is a string or list of strings."),
    "properties": {
        "grade": {"type": "array", "items": {"type": "string"},
                  "description": "Origination grade letters, A through G."},
        "stage": {"type": "array", "items": {"type": "string"},
                  "description": "IFRS 9 stage: '1', '2' or '3'."},
        "term": {"type": "array", "items": {"type": "string"},
                 "description": "Contractual term in months: '36' or '60'."},
        "purpose": {"type": "array", "items": {"type": "string"},
                    "description": "Loan purpose, e.g. debt_consolidation."},
        "state": {"type": "array", "items": {"type": "string"},
                  "description": "US state code, e.g. CA."},
        "status": {"type": "array", "items": {"type": "string"},
                   "description": "Loan status string, e.g. 'Current'."},
        "vintage_year": {"type": "array", "items": {"type": "string"},
                         "description": "Origination year, e.g. '2018'."},
    },
    "additionalProperties": False,
}

TOOL_SCHEMAS = [
    {
        "name": "compute_ecl",
        "description": (
            "Expected credit loss for a slice of the portfolio under one "
            "macroeconomic scenario. Returns the provision in absolute "
            "currency, broken down by IFRS 9 stage, plus EAD, coverage ratio, "
            "and the LGD and PD scalar used. This is the only way to obtain a "
            "provision figure."),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": _FILTER_SCHEMA,
                "scenario": {
                    "type": "string",
                    "enum": ["base", "adverse", "severe"],
                    "description": "Macroeconomic scenario. Defaults to base.",
                },
            },
        },
    },
    {
        "name": "get_portfolio_slice",
        "description": (
            "Summary statistics for a slice of the book: loan count, exposure, "
            "weighted average interest rate and remaining life, mean 12-month "
            "and lifetime PD, mean FICO drift, and exposure split by stage and "
            "grade. No provision figure. Set top_n_by_exposure to also get "
            "individual loan ids, largest exposure first, which you then pass "
            "to explain_stage_assignment."),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": _FILTER_SCHEMA,
                "top_n_by_exposure": {
                    "type": "integer",
                    "description": ("Return this many individual loan ids, "
                                    "largest exposure first. Max 20. Use when "
                                    "asked about a specific or extreme loan."),
                },
            },
        },
    },
    {
        "name": "compare_segments",
        "description": (
            "Compare two or more slices side by side and get the RATIOS and "
            "DIFFERENCES between them, already computed. Use this whenever a "
            "question asks how one segment compares to another, which segment "
            "is largest, or what share of the total a segment represents. "
            "Never divide two figures yourself; call this instead."),
        "input_schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": _FILTER_SCHEMA,
                    "description": ("Two to eight filter objects, e.g. "
                                    "[{\"grade\": [\"A\"]}, {\"grade\": [\"G\"]}]"),
                },
                "scenario": {
                    "type": "string",
                    "enum": ["base", "adverse", "severe"],
                },
            },
            "required": ["segments"],
        },
    },
    {
        "name": "run_stress",
        "description": (
            "Apply an ad-hoc shock outside the three standard scenarios. "
            "Scales the monthly default hazard by pd_scalar and optionally "
            "adds lgd_addon to loss given default. Returns base provision, "
            "stressed provision, and the delta in both currency and percent."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pd_scalar": {
                    "type": "number",
                    "description": ("Multiplier on the default hazard. 1.0 is "
                                    "no shock, 1.5 raises hazards by half."),
                },
                "lgd_addon": {
                    "type": "number",
                    "description": ("Absolute addition to LGD, e.g. 0.02 for "
                                    "two percentage points. Capped at 1.0."),
                },
                "filters": _FILTER_SCHEMA,
            },
            "required": ["pd_scalar"],
        },
    },
    {
        "name": "explain_stage_assignment",
        "description": (
            "Why one specific loan sits in its IFRS 9 stage. Returns the stage, "
            "which SICR limbs fired with their threshold values, and the "
            "underlying facts: status, days past due, grade, FICO at "
            "origination and now, PD multiplier, and PD at origination versus "
            "today. Use whenever asked about an individual loan."),
        "input_schema": {
            "type": "object",
            "properties": {
                "loan_id": {
                    "type": "string",
                    "description": "The loan id from the live book.",
                },
            },
            "required": ["loan_id"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]


# =============================================================================
# grounded-value extraction
# =============================================================================
_STR_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(obj) -> list:
    """
    Every number anywhere in a tool result, including inside strings.

    Strings matter. Loan ids come back as text ("145178766"), and
    explain_stage_assignment reports the PD ratio inside a formatted sentence
    ("lifetime PD ratio 5.06x >= threshold 2.0x"). Both are legitimately
    quotable by the agent. An earlier version walked only numeric leaves and
    flagged both as ungrounded, which was a false positive in the checker, not
    a fabrication by the model.

    Anything present in a tool result is grounded by definition, whatever its
    type. Booleans are not numbers.
    """
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        for m in _STR_NUM_RE.findall(obj):
            try:
                out.append(float(m))
            except ValueError:
                pass
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(numbers_in(v))
    return out


# =============================================================================
# dispatch
# =============================================================================
def _coerce_filters(f):
    """Tool inputs arrive as lists of strings. The engine wants that too, but
    be liberal about a bare string sneaking through."""
    if not f:
        return None
    out = {}
    for k, v in f.items():
        out[k] = v if isinstance(v, (list, tuple)) else [v]
    return out


def dispatch(name: str, args: dict) -> dict:
    """Route one tool call to the engine. Errors are returned, not raised,
    so the agent can see what went wrong and correct itself."""
    try:
        if name == "compute_ecl":
            return E.compute_ecl(
                filters=_coerce_filters(args.get("filters")),
                scenario=args.get("scenario", "base"))

        if name == "get_portfolio_slice":
            return E.get_portfolio_slice(
                filters=_coerce_filters(args.get("filters")),
                top_n_by_exposure=int(args.get("top_n_by_exposure", 0) or 0))

        if name == "compare_segments":
            return E.compare_segments(
                segments=[_coerce_filters(s) for s in args.get("segments", [])],
                scenario=args.get("scenario", "base"))

        if name == "run_stress":
            return E.run_stress(
                pd_scalar=float(args["pd_scalar"]),
                lgd_addon=float(args.get("lgd_addon", 0.0)),
                filters=_coerce_filters(args.get("filters")))

        if name == "explain_stage_assignment":
            return E.explain_stage_assignment(args["loan_id"])

        return {"error": f"unknown tool '{name}'", "valid_tools": TOOL_NAMES}

    except Exception as exc:                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "tool": name,
                "arguments": args}


# =============================================================================
# session: dispatch plus an auditable call log
# =============================================================================
class ToolSession:
    """Runs tool calls and keeps the trace the guardrail needs."""

    def __init__(self):
        self.calls = []

    def call(self, name: str, args: dict) -> dict:
        call_id = f"c{len(self.calls) + 1}"
        result = dispatch(name, args)
        self.calls.append({
            "call_id": call_id,
            "tool": name,
            "arguments": args,
            "result": result,
            "numbers": numbers_in(result),
            "ok": "error" not in result,
            "uuid": str(uuid.uuid4())[:8],
        })
        return result

    @property
    def grounded_numbers(self) -> list:
        out = []
        for c in self.calls:
            out.extend(c["numbers"])
        return out

    def trace(self) -> list:
        """Compact log for display in the app."""
        return [{"call_id": c["call_id"], "tool": c["tool"],
                 "arguments": c["arguments"], "ok": c["ok"]}
                for c in self.calls]

    def reset(self):
        self.calls = []
