"""
engine.py

THE DETERMINISTIC ECL ENGINE.

Every number this project reports comes from here. The LLM agent calls these
functions and explains what comes back. It never computes anything itself.
That separation is the whole point of the project.

CORE FORMULA, per loan:

    ECL = sum over t of   mPD_t * LGD * EAD_t * DF_t

    mPD_t  marginal default probability in month t, conditional on the loan
           having survived to the reporting date. Taken from the competing
           risks term structure, divided by survival to current months-on-book,
           then multiplied by the loan's FICO-drift multiplier.
    EAD_t  contractual amortising balance at month t (annuity formula).
    DF_t   (1 + monthly EIR) ^ -t
    T      12 months for Stage 1, remaining contractual life for Stage 2.

    Stage 3 is credit-impaired: PD = 1 by definition, so ECL = LGD * EAD.
    The lifetime PD carried on those rows is ignored, deliberately.

Scenarios scale the monthly default HAZARD and recompute survival. They are
not applied to cumulative PD directly, which would be an approximation.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import schema as S

# Prefer the trimmed deployment artifact when it exists. It holds every loan
# and produces identical figures, but with unused columns dropped and dtypes
# downcast so the app fits a free-tier memory limit. Falls back to the full
# staged file for local work.
_DEPLOY_PATH = "data/processed/live_book_deploy.parquet"
_FULL_PATH = "data/processed/live_book_staged.parquet"
STAGED_PATH = _DEPLOY_PATH if os.path.exists(_DEPLOY_PATH) else _FULL_PATH
PD_CURVES_PATH = "data/processed/pd_curves.parquet"

MAX_TERM = 60
_state = None


# =============================================================================
# state
# =============================================================================
class _State:
    def __init__(self, book: pd.DataFrame, seg_index, mpd, surv, hazard):
        self.book = book
        self.scenario_cols = {}         # scenario -> per-loan ECL column name
        self.seg_index = seg_index      # (grade, term) -> row in the matrices
        self.mpd = mpd                  # (n_seg, MAX_TERM+1) marginal PD
        self.surv = surv                # (n_seg, MAX_TERM+1) survival
        self.hazard = hazard            # (n_seg, MAX_TERM+1) default hazard


def load_state(force: bool = False) -> _State:
    global _state
    if _state is not None and not force:
        return _state

    for p in (STAGED_PATH, PD_CURVES_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing. Run notebooks/01..04 first.")

    book = pd.read_parquet(STAGED_PATH)
    curves = pd.read_parquet(PD_CURVES_PATH)

    segs = sorted({(str(g), int(t)) for g, t in
                   zip(curves["grade"], curves["term_m"])})
    seg_index = {s: i for i, s in enumerate(segs)}

    n = len(segs)
    mpd = np.zeros((n, MAX_TERM + 1))
    surv = np.ones((n, MAX_TERM + 1))
    haz = np.zeros((n, MAX_TERM + 1))

    for (g, t), sub in curves.groupby(["grade", "term_m"]):
        i = seg_index[(str(g), int(t))]
        sub = sub.sort_values("month")
        m = sub["month"].to_numpy().astype(int)
        mpd[i, m] = sub["marginal_pd"].to_numpy()
        surv[i, m] = sub["survival"].to_numpy()
        haz[i, m] = sub["hazard_default"].to_numpy()
        if m.max() < MAX_TERM:                      # pad past maturity
            surv[i, m.max() + 1:] = surv[i, m.max()]

    book["_seg"] = [seg_index.get((g, int(t)), -1) for g, t in
                    zip(book["grade_str"], book["term_m"])]
    if (book["_seg"] < 0).any():
        raise ValueError("live book contains a (grade, term) with no PD curve")

    book["_horizon"] = np.where(
        book["stage"] == 1, C.STAGE1_HORIZON_M,
        book["remaining_m"].clip(lower=1).astype(int))
    book["_monthly_rate"] = (1.0 + book["eir_annual"]) ** (1 / 12.0) - 1.0

    _state = _State(book, seg_index, mpd, surv, haz)
    return _state


def _scenario_curves(scenario: str, pd_scalar_override=None):
    """Scale the default hazard, recompute survival and marginal PD."""
    st = load_state()
    if pd_scalar_override is not None:
        k = float(pd_scalar_override)
    else:
        if scenario not in C.SCENARIOS:
            raise ValueError(f"unknown scenario '{scenario}'. "
                             f"choose from {list(C.SCENARIOS)}")
        k = C.SCENARIOS[scenario]["pd_scalar"]

    if abs(k - 1.0) < 1e-12:
        return st.mpd, st.surv

    h = np.clip(st.hazard * k, 0.0, 0.99)
    # implied prepayment hazard from the base curves, held fixed
    with np.errstate(divide="ignore", invalid="ignore"):
        s_prev = np.concatenate([np.ones((st.surv.shape[0], 1)),
                                 st.surv[:, :-1]], axis=1)
        total_exit = np.where(s_prev > 1e-12, 1.0 - st.surv / s_prev, 0.0)
    h_prepay = np.clip(total_exit - st.hazard, 0.0, 1.0)

    tot = np.clip(h + h_prepay, 0.0, 0.99)
    surv = np.cumprod(1.0 - tot, axis=1)
    s_prev2 = np.concatenate([np.ones((surv.shape[0], 1)), surv[:, :-1]], axis=1)
    mpd = s_prev2 * h
    surv[:, 0] = 1.0
    mpd[:, 0] = 0.0
    return mpd, surv


def _amortising_balance(b0, monthly_rate, remaining, t):
    """Contractual balance after t payments on an annuity with `remaining` left."""
    i = np.where(monthly_rate <= 0, 1e-9, monthly_rate)
    n = np.maximum(remaining, 1)
    tt = np.minimum(t, n)
    num = (1 + i) ** n - (1 + i) ** tt
    den = (1 + i) ** n - 1
    frac = np.where(den > 0, num / den, np.maximum(0.0, 1 - tt / n))
    return b0 * np.clip(frac, 0.0, 1.0)


# =============================================================================
# filtering
# =============================================================================
_FILTER_FIELDS = {
    "grade": "grade_str", "stage": "stage", "term": "term_m",
    "purpose": S.PURPOSE, "state": S.STATE, "status": S.STATUS,
    "vintage_year": "vintage_year",
}


def _apply_filters(book: pd.DataFrame, filters: dict | None):
    """
    Filter the book. Values arrive as strings from the tool layer, but some
    columns are numeric (term_m and vintage_year are floats), so "60" would
    never match "60.0" under naive string comparison. Compare numerically
    where the column is numeric.
    """
    if not filters:
        return book, {}
    used = {}
    out = book
    for k, v in filters.items():
        col = _FILTER_FIELDS.get(k)
        if col is None:
            raise ValueError(f"unknown filter '{k}'. "
                             f"valid: {sorted(_FILTER_FIELDS)}")
        vals = v if isinstance(v, (list, tuple, set)) else [v]
        series = out[col]

        if pd.api.types.is_numeric_dtype(series):
            nums = []
            for x in vals:
                try:
                    nums.append(float(x))
                except (TypeError, ValueError):
                    pass
            mask = (series.astype(float).isin(nums) if nums
                    else pd.Series(False, index=series.index))
        else:
            mask = series.astype(str).isin([str(x) for x in vals])

        out = out[mask]
        used[k] = list(vals)
    return out, used


# =============================================================================
# per-loan ECL vector  (the expensive part, computed once per scenario)
# =============================================================================
def _ecl_vector(book: pd.DataFrame, mpd_m, surv_m, lgd: float):
    """Per-loan ECL and cumulative PD. Loops months, vectorised over loans."""
    seg = book["_seg"].to_numpy()
    mob = np.clip(book["mob"].to_numpy().astype(int), 0, MAX_TERM)
    term = book["term_m"].to_numpy().astype(int)
    horizon = book["_horizon"].to_numpy().astype(int)
    ead0 = book[S.OUT_PRNCP].to_numpy()
    rate = book["_monthly_rate"].to_numpy()
    mult = book["pd_multiplier"].to_numpy()
    stage = book["stage"].to_numpy()
    remaining = book["remaining_m"].to_numpy().astype(int)

    s_at_mob = np.where(mob == 0, 1.0, surv_m[seg, mob])
    s_at_mob = np.where(s_at_mob < 1e-9, 1e-9, s_at_mob)

    ecl = np.zeros(len(book))
    cum_pd = np.zeros(len(book))

    for t in range(1, MAX_TERM + 1):
        active = (t <= horizon) & (mob + t <= term) & (stage != 3)
        if not active.any():
            continue
        idx = np.where(active)[0]
        month_abs = mob[idx] + t

        m_pd = mpd_m[seg[idx], month_abs] / s_at_mob[idx] * mult[idx]
        m_pd = np.minimum(m_pd, np.clip(1.0 - cum_pd[idx], 0.0, 1.0))
        cum_pd[idx] += m_pd

        bal = _amortising_balance(ead0[idx], rate[idx], remaining[idx], t)
        df = (1.0 + rate[idx]) ** (-t)
        ecl[idx] += m_pd * lgd * bal * df

    s3 = stage == 3                      # credit-impaired: PD = 1 by definition
    ecl[s3] = lgd * ead0[s3]
    cum_pd[s3] = 1.0
    return ecl, cum_pd


def _ensure_scenario_column(scenario: str) -> str:
    """Compute and cache the full-book ECL vector for a named scenario.

    Per-loan ECL does not depend on filters, so this runs once per scenario
    and every later filtered query becomes a groupby sum.
    """
    st = load_state()
    col = st.scenario_cols.get(scenario)
    if col is not None and col in st.book.columns:
        return col

    cfg = C.SCENARIOS[scenario]
    lgd = min(C.LGD_PORTFOLIO + cfg["lgd_addon"], C.LGD_CAP)
    mpd_m, surv_m = _scenario_curves(scenario)
    ecl, _ = _ecl_vector(st.book, mpd_m, surv_m, lgd)

    col = f"_ecl_{scenario}"
    st.book[col] = ecl
    st.scenario_cols[scenario] = col
    return col


def warm_cache():
    """Precompute every named scenario. Call once at app startup."""
    for s in C.SCENARIOS:
        _ensure_scenario_column(s)


# =============================================================================
# TOOL 1: compute_ecl
# =============================================================================
def compute_ecl(filters: dict | None = None, scenario: str = "base",
                pd_scalar_override: float | None = None,
                lgd_override: float | None = None) -> dict:
    """
    Expected credit loss for a slice of the book under one scenario.
    Returns absolute currency amounts plus the inputs that produced them.
    """
    st = load_state()
    book, used = _apply_filters(st.book, filters)

    if len(book) == 0:
        return {"error": "no loans match those filters", "filters": used,
                "n_loans": 0, "ead": 0.0, "ecl_total": 0.0}

    custom = (pd_scalar_override is not None) or (lgd_override is not None)

    if not custom:
        if scenario not in C.SCENARIOS:
            raise ValueError(f"unknown scenario '{scenario}'. "
                             f"choose from {list(C.SCENARIOS)}")
        col = _ensure_scenario_column(scenario)
        ecl = book[col].to_numpy()
        lgd_addon = C.SCENARIOS[scenario]["lgd_addon"]
        lgd = min(C.LGD_PORTFOLIO + lgd_addon, C.LGD_CAP)
        pd_scalar = C.SCENARIOS[scenario]["pd_scalar"]
        scen_label = scenario
    else:
        lgd_addon = 0.0 if pd_scalar_override is not None \
            else C.SCENARIOS[scenario]["lgd_addon"]
        lgd = C.LGD_PORTFOLIO if lgd_override is None else float(lgd_override)
        lgd = min(lgd + lgd_addon, C.LGD_CAP)
        pd_scalar = float(pd_scalar_override) if pd_scalar_override is not None \
            else C.SCENARIOS[scenario]["pd_scalar"]
        mpd_m, surv_m = _scenario_curves(scenario, pd_scalar_override)
        ecl, _ = _ecl_vector(book, mpd_m, surv_m, lgd)
        scen_label = f"custom(pd_scalar={pd_scalar}, lgd={round(lgd,4)})"

    book = book.assign(_ecl=ecl)
    total_ead = float(book[S.OUT_PRNCP].sum())
    total_ecl = float(ecl.sum())

    by_stage = {}
    for stg, sub in book.groupby("stage"):
        e = float(sub[S.OUT_PRNCP].sum())
        se = float(sub["_ecl"].sum())
        by_stage[f"stage_{int(stg)}"] = {
            "n_loans": int(len(sub)),
            "ead": round(e, 2),
            "ecl": round(se, 2),
            "coverage_pct": round(se / e * 100, 4) if e else 0.0,
            # shares are returned so the agent never has to divide
            "share_of_slice_ecl_pct": round(se / total_ecl * 100, 4)
            if total_ecl else 0.0,
            "share_of_slice_ead_pct": round(e / total_ead * 100, 4)
            if total_ead else 0.0,
        }

    # if this is a filtered slice, say what share of the whole book it is
    share_block = None
    if used:
        whole = _ensure_scenario_column(scenario) if not custom else None
        if whole is not None:
            book_ecl = float(st.book[whole].sum())
            book_ead = float(st.book[S.OUT_PRNCP].sum())
            share_block = {
                "portfolio_ecl": round(book_ecl, 2),
                "portfolio_ead": round(book_ead, 2),
                "slice_share_of_portfolio_ecl_pct":
                    round(total_ecl / book_ecl * 100, 4) if book_ecl else 0.0,
                "slice_share_of_portfolio_ead_pct":
                    round(total_ead / book_ead * 100, 4) if book_ead else 0.0,
            }

    return {
        "scenario": scen_label,
        "filters": used,
        "n_loans": int(len(book)),
        "ead": round(total_ead, 2),
        "ecl_total": round(total_ecl, 2),
        "coverage_pct": round(total_ecl / total_ead * 100, 4) if total_ead else 0.0,
        "by_stage": by_stage,
        "portfolio_context": share_block,
        "inputs": {
            "lgd_used": round(lgd, 4),
            "pd_scalar": round(pd_scalar, 4),
            "lgd_addon": round(lgd_addon, 4),
            "stage1_horizon_months": C.STAGE1_HORIZON_M,
        },
        "currency": C.CURRENCY,
        "reporting_date": C.REPORTING_DATE,
    }


# =============================================================================
# TOOL 2: get_portfolio_slice
# =============================================================================
def get_portfolio_slice(filters: dict | None = None,
                        top_n_by_exposure: int = 0) -> dict:
    """
    Summary statistics for a slice. No ECL, no scenario. Just the book.

    top_n_by_exposure returns that many individual loan ids, largest exposure
    first, so a question about a specific loan can be chained into
    explain_stage_assignment. Without this there is no route from an aggregate
    question to a loan id.
    """
    st = load_state()
    book, used = _apply_filters(st.book, filters)
    if len(book) == 0:
        return {"error": "no loans match those filters", "filters": used,
                "n_loans": 0}

    top = None
    if top_n_by_exposure and top_n_by_exposure > 0:
        n = min(int(top_n_by_exposure), 20)
        head = book.nlargest(n, S.OUT_PRNCP)
        top = [{"loan_id": str(r[S.ID]),
                "outstanding_principal": round(float(r[S.OUT_PRNCP]), 2),
                "stage": int(r["stage"]),
                "grade": str(r["grade_str"]),
                "term_months": int(r["term_m"])}
               for _, r in head.iterrows()]

    ead = book[S.OUT_PRNCP]
    return {
        "top_loans_by_exposure": top,
        "filters": used,
        "n_loans": int(len(book)),
        "ead": round(float(ead.sum()), 2),
        "mean_exposure": round(float(ead.mean()), 2),
        "share_of_book_by_ead": round(float(ead.sum() / st.book[S.OUT_PRNCP].sum()) * 100, 4),
        "weighted_avg_interest_rate_pct": round(
            float(np.average(book[S.INT_RATE], weights=ead)), 3),
        "weighted_avg_remaining_months": round(
            float(np.average(book["remaining_m"], weights=ead)), 1),
        "mean_pd_12m": round(float(np.average(book["pd_12m"], weights=ead)), 5),
        "mean_pd_lifetime": round(float(np.average(book["pd_lifetime_curr"], weights=ead)), 5),
        "mean_fico_drift": round(float(np.average(book["fico_delta_used"], weights=ead)), 2),
        "ead_by_stage": {f"stage_{int(k)}": round(float(v), 2)
                         for k, v in book.groupby("stage")[S.OUT_PRNCP].sum().items()},
        "ead_by_grade": {str(k): round(float(v), 2)
                         for k, v in book.groupby("grade_str", observed=True)[S.OUT_PRNCP].sum().items()},
        "currency": C.CURRENCY,
    }


# =============================================================================
# TOOL 3: run_stress
# =============================================================================
def run_stress(pd_scalar: float, lgd_addon: float = 0.0,
               filters: dict | None = None) -> dict:
    """Ad-hoc shock. Returns the stressed provision and the delta versus base."""
    base = compute_ecl(filters=filters, scenario="base")
    lgd = min(C.LGD_PORTFOLIO + float(lgd_addon), C.LGD_CAP)
    shocked = compute_ecl(filters=filters, pd_scalar_override=float(pd_scalar),
                          lgd_override=lgd)
    delta = shocked["ecl_total"] - base["ecl_total"]
    return {
        "shock": {"pd_scalar": float(pd_scalar), "lgd_addon": float(lgd_addon),
                  "lgd_used": round(lgd, 4)},
        "filters": base.get("filters", {}),
        "base_ecl": base["ecl_total"],
        "stressed_ecl": shocked["ecl_total"],
        "delta_ecl": round(delta, 2),
        "delta_pct": round(delta / base["ecl_total"] * 100, 3)
        if base["ecl_total"] else 0.0,
        "ead": base["ead"],
        "base_coverage_pct": base["coverage_pct"],
        "stressed_coverage_pct": shocked["coverage_pct"],
        "currency": C.CURRENCY,
    }


# =============================================================================
# TOOL 4: explain_stage_assignment
# =============================================================================
def explain_stage_assignment(loan_id) -> dict:
    """Why one specific loan sits in the stage it does. Facts only."""
    st = load_state()
    key = str(loan_id).strip()
    ids = st.book[S.ID].astype(str)
    row = st.book[ids == key]
    if len(row) == 0 and key.endswith(".0"):
        row = st.book[ids == key[:-2]]
    if len(row) == 0:
        return {"error": f"loan_id {loan_id} not in the live book",
                "hint": "the live book holds only loans open at 2018-12-31"}
    r = row.iloc[0]

    triggers = []
    if bool(r["sicr_relative"]):
        triggers.append(
            f"relative: lifetime PD ratio {r['pd_ratio']:.2f}x "
            f">= threshold {C.SICR_PD_RATIO}x")
    if bool(r["sicr_absolute"]):
        triggers.append(
            f"absolute: lifetime PD up {r['pd_delta']*100:.2f}pp "
            f">= threshold {C.SICR_PD_ABS*100:.0f}pp")
    if bool(r["sicr_dpd"]):
        triggers.append(
            f"backstop: {r['dpd']:.0f} days past due "
            f">= threshold {C.DPD_BACKSTOP}")

    stage = int(r["stage"])
    if stage == 3:
        reason = (f"Stage 3, credit-impaired. loan_status is "
                  f"'{r[S.STATUS]}'. PD is set to 1.0 by definition and "
                  f"ECL equals LGD times EAD.")
    elif stage == 2:
        reason = ("Stage 2, significant increase in credit risk. "
                  "Triggered by: " + "; ".join(triggers) +
                  ". ECL is measured over remaining lifetime.")
    else:
        reason = ("Stage 1, no significant increase in credit risk. "
                  "No SICR limb triggered. ECL is measured over 12 months.")

    return {
        "loan_id": str(r[S.ID]),
        "stage": stage,
        "reason": reason,
        "triggers_fired": triggers,
        "facts": {
            "loan_status": str(r[S.STATUS]),
            "days_past_due_band": float(r["dpd"]),
            "origination_grade": str(r["grade_str"]),
            "behaves_like_grade": str(r["grade_narrative"]),
            "term_months": int(r["term_m"]),
            "months_on_book": int(r["mob"]),
            "remaining_months": int(r["remaining_m"]),
            "outstanding_principal": round(float(r[S.OUT_PRNCP]), 2),
            "interest_rate_pct": float(r[S.INT_RATE]),
            "fico_at_origination": float(r["fico_orig"]),
            "fico_current": float(r["fico_curr"]),
            "fico_drift": float(r["fico_delta_used"]),
            "pd_multiplier_from_fico": round(float(r["pd_multiplier"]), 4),
            "pd_lifetime_at_origination": round(float(r["pd_lifetime_orig"]), 5),
            "pd_lifetime_current": round(float(r["pd_lifetime_curr"]), 5),
            "pd_12m_current": round(float(r["pd_12m"]), 5),
            # returned as numbers, not only inside the trigger sentences,
            # so the agent can quote them without the checker guessing
            "pd_ratio_current_over_origination": round(float(r["pd_ratio"]), 4),
            "pd_increase_pp": round(float(r["pd_delta"]) * 100, 4),
        },
        "thresholds": {
            "sicr_pd_ratio": C.SICR_PD_RATIO,
            "sicr_pd_absolute": C.SICR_PD_ABS,
            "dpd_backstop": C.DPD_BACKSTOP,
        },
    }


# =============================================================================
# weighted headline
# =============================================================================
def compute_weighted_ecl(filters: dict | None = None) -> dict:
    """Probability-weighted ECL across all scenarios. The headline number."""
    per = {}
    total = 0.0
    for name, cfg in C.SCENARIOS.items():
        r = compute_ecl(filters=filters, scenario=name)
        per[name] = {"weight": cfg["weight"], "ecl": r["ecl_total"],
                     "coverage_pct": r["coverage_pct"]}
        total += cfg["weight"] * r["ecl_total"]
    ead = compute_ecl(filters=filters, scenario="base")["ead"]
    return {
        "weighted_ecl": round(total, 2),
        "ead": ead,
        "weighted_coverage_pct": round(total / ead * 100, 4) if ead else 0.0,
        "by_scenario": per,
        "currency": C.CURRENCY,
        "reporting_date": C.REPORTING_DATE,
    }


# =============================================================================
# TOOL 5: compare_segments
# =============================================================================
def compare_segments(segments: list, scenario: str = "base") -> dict:
    """
    Compare several slices side by side under one scenario, and return the
    ratios between them.

    Added because the evaluation showed the agent computing ratios itself when
    the engine did not supply them: share of total ECL, and one segment's
    coverage divided by another's. Both were correct arithmetic on grounded
    inputs, but the results were ungrounded figures. The fix is to return the
    comparison rather than to ask the model not to do it.

    segments: list of filter dicts, e.g. [{"grade": ["A"]}, {"grade": ["G"]}]
    """
    if not segments or len(segments) < 2:
        return {"error": "provide at least two segments to compare"}
    if len(segments) > 8:
        return {"error": "at most 8 segments per comparison"}

    rows = []
    for i, f in enumerate(segments):
        r = compute_ecl(filters=f, scenario=scenario)
        if r.get("n_loans", 0) == 0:
            rows.append({"label": f"segment_{i+1}", "filters": f,
                         "error": "no loans match"})
            continue
        rows.append({
            "label": f"segment_{i+1}",
            "filters": r["filters"],
            "n_loans": r["n_loans"],
            "ead": r["ead"],
            "ecl": r["ecl_total"],
            "coverage_pct": r["coverage_pct"],
        })

    valid = [r for r in rows if "error" not in r]
    comparisons = []
    for i in range(len(valid)):
        for j in range(len(valid)):
            if i >= j:
                continue
            a, b = valid[i], valid[j]
            comparisons.append({
                "a": a["label"], "b": b["label"],
                "ecl_ratio_a_over_b": round(a["ecl"] / b["ecl"], 4)
                if b["ecl"] else None,
                "coverage_ratio_a_over_b":
                    round(a["coverage_pct"] / b["coverage_pct"], 4)
                    if b["coverage_pct"] else None,
                "ecl_difference_a_minus_b": round(a["ecl"] - b["ecl"], 2),
                "coverage_difference_pp_a_minus_b":
                    round(a["coverage_pct"] - b["coverage_pct"], 4),
            })

    largest = max(valid, key=lambda r: r["ecl"]) if valid else None
    highest_cov = max(valid, key=lambda r: r["coverage_pct"]) if valid else None

    return {
        "scenario": scenario,
        "segments": rows,
        "comparisons": comparisons,
        "largest_ecl_segment": largest["label"] if largest else None,
        "highest_coverage_segment": highest_cov["label"] if highest_cov else None,
        "currency": C.CURRENCY,
    }
