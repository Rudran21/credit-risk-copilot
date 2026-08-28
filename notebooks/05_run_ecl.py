"""
05_run_ecl.py

Runs the engine and prints the headline deliverable: ECL by stage x scenario.

Usage:
    python notebooks\\05_run_ecl.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from ecl import config as C          # noqa: E402
from ecl import engine as E          # noqa: E402


def money(x):
    return f"${x/1e6:,.1f}m"


def main():
    t0 = time.time()
    st = E.load_state()
    print(f"loaded {len(st.book):,} live loans in {time.time()-t0:.1f}s")

    # ------------------------------------------------ ECL by stage x scenario
    print("\n" + "=" * 74)
    print("ECL BY STAGE x SCENARIO   (reporting date "
          f"{C.REPORTING_DATE}, {C.CURRENCY})")
    print("=" * 74)

    results = {s: E.compute_ecl(scenario=s) for s in C.SCENARIOS}
    base = results["base"]

    hdr = f"{'':<10} {'EAD':>14} " + "".join(
        f"{s:>14}" for s in C.SCENARIOS)
    print(hdr)
    print("-" * 74)

    for stg in [1, 2, 3]:
        key = f"stage_{stg}"
        if key not in base["by_stage"]:
            continue
        ead = base["by_stage"][key]["ead"]
        row = f"Stage {stg:<4} {money(ead):>14} "
        for s in C.SCENARIOS:
            row += f"{money(results[s]['by_stage'][key]['ecl']):>14}"
        print(row)

    print("-" * 74)
    tot = f"{'TOTAL':<10} {money(base['ead']):>14} "
    for s in C.SCENARIOS:
        tot += f"{money(results[s]['ecl_total']):>14}"
    print(tot)

    cov = f"{'coverage':<10} {'':>14} "
    for s in C.SCENARIOS:
        cov += f"{results[s]['coverage_pct']:>13.2f}%"
    print(cov)

    wt = f"{'weight':<10} {'':>14} "
    for s in C.SCENARIOS:
        wt += f"{C.SCENARIOS[s]['weight']:>14.0%}"
    print(wt)

    # ------------------------------------------------------- headline number
    w = E.compute_weighted_ecl()
    print("\n" + "=" * 74)
    print(f"  PROBABILITY-WEIGHTED ECL : {money(w['weighted_ecl'])}"
          f"   ({w['weighted_ecl']:,.0f} {C.CURRENCY})")
    print(f"  ON EAD OF                : {money(w['ead'])}")
    print(f"  COVERAGE RATIO           : {w['weighted_coverage_pct']:.2f}%")
    print("=" * 74)

    # ------------------------------------------------------- stage coverage
    print("\n--- COVERAGE BY STAGE (base scenario) ---")
    for k, v in base["by_stage"].items():
        print(f"  {k}: EAD {money(v['ead']):>12}  ECL {money(v['ecl']):>10}  "
              f"coverage {v['coverage_pct']:>7.2f}%")
    print("  sanity: coverage must rise sharply 1 -> 2 -> 3")

    # ------------------------------------------------------------ by grade
    print("\n--- BASE ECL BY ORIGINATION GRADE ---")
    print(f"  {'grade':<7} {'EAD':>12} {'ECL':>11} {'coverage':>10}")
    for g in ["A", "B", "C", "D", "E", "F", "G"]:
        r = E.compute_ecl(filters={"grade": g}, scenario="base")
        if r.get("n_loans", 0) == 0:
            continue
        print(f"  {g:<7} {money(r['ead']):>12} {money(r['ecl_total']):>11} "
              f"{r['coverage_pct']:>9.2f}%")

    # --------------------------------------------------------- tool smoke test
    print("\n--- TOOL SMOKE TEST ---")

    sl = E.get_portfolio_slice({"grade": ["F", "G"]})
    print(f"  get_portfolio_slice(grade=F,G): n={sl['n_loans']:,}  "
          f"EAD={money(sl['ead'])}  wavg rate={sl['weighted_avg_interest_rate_pct']}%")

    ss = E.run_stress(pd_scalar=1.5, lgd_addon=0.02)
    print(f"  run_stress(pd x1.5, lgd +2pp): base={money(ss['base_ecl'])} -> "
          f"{money(ss['stressed_ecl'])}  ({ss['delta_pct']:+.1f}%)")

    s2 = st.book[st.book["stage"] == 2]
    if len(s2):
        ex = E.explain_stage_assignment(s2.iloc[0][ "id" ])
        print(f"  explain_stage_assignment({ex['loan_id']}): stage {ex['stage']}")
        print(f"    triggers: {ex['triggers_fired']}")

    print(f"\n  total runtime {time.time()-t0:.1f}s")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
