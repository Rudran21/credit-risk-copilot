"""
test_engine.py

Regression tests for the deterministic engine and the grounding check.

No pytest needed. Run directly:
    python tests\\test_engine.py

These lock the properties that must never silently change: the headline
provision, the identities that IFRS 9 forces (Stage 3 coverage equals LGD),
monotonicity across stages and grades, and the fact that the guardrail still
catches a fabricated figure.

Tolerance on the headline is 0.01% rather than exact, because the deployment
artefact downcasts selected columns to float32. That introduces about $13 on a
$905m provision, which is disclosed in the validation note.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecl import config as C  # noqa: E402
from ecl import engine as E  # noqa: E402

HEADLINE_ECL = 905_350_344.0
HEADLINE_EAD = 9_510_384_856.44
TOL = 1e-4                       # 0.01% relative

_results = []


def check(name):
    def wrap(fn):
        _results.append((name, fn))
        return fn
    return wrap


def close(a, b, tol=TOL):
    return abs(a - b) <= abs(b) * tol


# =============================================================================
# headline
# =============================================================================
@check("weighted ECL matches the published headline")
def t_headline():
    w = E.compute_weighted_ecl()
    assert close(w["weighted_ecl"], HEADLINE_ECL), \
        f"got {w['weighted_ecl']:,.0f}, expected {HEADLINE_ECL:,.0f}"


@check("total EAD is unchanged")
def t_ead():
    r = E.compute_ecl(scenario="base")
    assert close(r["ead"], HEADLINE_EAD), f"got {r['ead']:,.2f}"


@check("live book holds 912,609 loans")
def t_count():
    r = E.compute_ecl(scenario="base")
    assert r["n_loans"] == 912_609, f"got {r['n_loans']:,}"


# =============================================================================
# IFRS 9 identities
# =============================================================================
@check("Stage 3 coverage equals LGD exactly (PD is 1 by definition)")
def t_stage3_identity():
    r = E.compute_ecl(scenario="base")
    cov = r["by_stage"]["stage_3"]["coverage_pct"] / 100.0
    assert close(cov, C.LGD_PORTFOLIO, 1e-3), \
        f"Stage 3 coverage {cov:.6f} != LGD {C.LGD_PORTFOLIO}"


@check("coverage rises across stages 1 -> 2 -> 3")
def t_stage_monotone():
    r = E.compute_ecl(scenario="base")
    c = [r["by_stage"][f"stage_{i}"]["coverage_pct"] for i in (1, 2, 3)]
    assert c[0] < c[1] < c[2], f"not monotone: {c}"


@check("ECL by stage sums to the total")
def t_stage_sums():
    r = E.compute_ecl(scenario="base")
    parts = sum(v["ecl"] for v in r["by_stage"].values())
    assert close(parts, r["ecl_total"], 1e-6), \
        f"parts {parts:,.2f} vs total {r['ecl_total']:,.2f}"


@check("scenarios are ordered base < adverse < severe")
def t_scenario_order():
    v = [E.compute_ecl(scenario=s)["ecl_total"]
         for s in ("base", "adverse", "severe")]
    assert v[0] < v[1] < v[2], f"not ordered: {v}"


@check("coverage rises monotonically from grade A to G")
def t_grade_monotone():
    cov = [E.compute_ecl(filters={"grade": [g]}, scenario="base")["coverage_pct"]
           for g in "ABCDEFG"]
    assert all(cov[i] < cov[i + 1] for i in range(len(cov) - 1)), \
        f"not monotone: {[round(c, 2) for c in cov]}"


@check("grade slices sum to the whole book")
def t_grade_sums():
    total = sum(E.compute_ecl(filters={"grade": [g]}, scenario="base")["ecl_total"]
                for g in "ABCDEFG")
    whole = E.compute_ecl(scenario="base")["ecl_total"]
    assert close(total, whole, 1e-6), f"{total:,.2f} vs {whole:,.2f}"


# =============================================================================
# filters
# =============================================================================
@check("numeric filters work on float columns (the term/vintage bug)")
def t_numeric_filters():
    for f in ({"term": ["60"]}, {"term": ["36"]},
              {"vintage_year": ["2018"]}, {"stage": ["2"]}):
        r = E.get_portfolio_slice(filters=f)
        assert r.get("n_loans", 0) > 0, f"{f} returned nothing"


@check("term slices partition the book")
def t_term_partition():
    a = E.get_portfolio_slice(filters={"term": ["36"]})["n_loans"]
    b = E.get_portfolio_slice(filters={"term": ["60"]})["n_loans"]
    assert a + b == 912_609, f"{a:,} + {b:,} != 912,609"


@check("an impossible filter returns an error, not a wrong number")
def t_empty_filter():
    r = E.compute_ecl(filters={"grade": ["Z"]}, scenario="base")
    assert "error" in r and r["n_loans"] == 0


# =============================================================================
# tools
# =============================================================================
@check("run_stress delta reconciles with base")
def t_stress():
    r = E.run_stress(pd_scalar=1.5)
    assert close(r["stressed_ecl"] - r["base_ecl"], r["delta_ecl"], 1e-6)
    assert r["stressed_ecl"] > r["base_ecl"]


@check("compare_segments returns ratios so the agent never divides")
def t_compare():
    r = E.compare_segments([{"grade": ["A"]}, {"grade": ["G"]}], scenario="base")
    assert len(r["comparisons"]) == 1
    c = r["comparisons"][0]
    assert c["coverage_ratio_a_over_b"] is not None
    assert c["coverage_difference_pp_a_minus_b"] < 0, "A should be below G"


@check("explain_stage_assignment returns numeric PD ratio, not only prose")
def t_explain():
    st = E.load_state()
    s2 = st.book[st.book["stage"] == 2]
    assert len(s2) > 0
    r = E.explain_stage_assignment(s2.iloc[0]["id"])
    assert r["stage"] == 2
    assert r["triggers_fired"]
    assert "pd_ratio_current_over_origination" in r["facts"]


@check("an unknown loan id errors rather than guessing")
def t_bad_loan():
    r = E.explain_stage_assignment("not-a-loan")
    assert "error" in r


# =============================================================================
# guardrail
# =============================================================================
@check("guardrail passes grounded figures including scaled and percent forms")
def t_guard_clean():
    from agent import guardrail as G

    class S:
        calls = [{"call_id": "c1"}]
        grounded_numbers = [771423886.04, 9510384856.44, 8.1113, 0.918, 744779]

    txt = ("ECL is $771.4m [c1] on EAD of $9,510,384,856.44 [c1], "
           "coverage 8.11% [c1], LGD 91.8% [c1], 744,779 loans [c1].")
    r = G.check(txt, S())
    assert r["passed"], r["ungrounded"]


@check("guardrail catches a fabricated figure")
def t_guard_catch():
    from agent import guardrail as G

    class S:
        calls = [{"call_id": "c1"}]
        grounded_numbers = [771423886.04]

    r = G.check("ECL is $771.4m [c1]. Last quarter it was $812.7m.", S())
    assert not r["passed"] and r["n_ungrounded"] == 1


@check("guardrail allowlists constants declared in config")
def t_guard_declared():
    from agent import guardrail as G
    for v in (C.SICR_PD_RATIO, float(C.DPD_BACKSTOP),
              float(C.IFRS9_DEFAULT_PRESUMPTION_DPD)):
        assert v in G.ALLOWLIST, f"{v} not allowlisted"


# =============================================================================
def main():
    print(f"engine source: {E.STAGED_PATH}\n")
    passed = failed = 0
    for name, fn in _results:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}\n          {exc}")
            failed += 1
        except Exception:                                  # noqa: BLE001
            print(f"  ERROR {name}")
            traceback.print_exc(limit=2)
            failed += 1

    print(f"\n  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
