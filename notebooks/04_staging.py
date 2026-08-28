"""
04_staging.py   (v3)

Assigns IFRS 9 Stage 1 / 2 / 3 and precomputes the PD horizons the engine needs.

WHY v3 -- v2's dual SICR test still left grade G at 0.0% Stage 2.

  ROOT CAUSE: v1 and v2 both routed the deterioration signal through an
  IMPLIED GRADE. Grade G is the worst grade that exists, so a G loan's
  implied grade can only be G or better. pd_delta <= 0 for every G loan,
  and both SICR limbs test for an increase. Grade F was nearly as bad:
  F-60m lifetime PD is 35.6% and G-60m is 36.4%, a 0.9pp gap, so even a
  genuine F -> G migration moved PD almost not at all. The grade scale
  compresses at the bottom and caps deterioration by construction.
  No threshold choice fixes this. See the v2 sensitivity table: every
  combination of relative and absolute cut-offs left G at 0.0%.

  FIX: make the deterioration signal CONTINUOUS and unbounded.

      PD_current = PD_origination * exp(-BETA * delta_FICO)

  delta_FICO = current FICO - origination FICO (negative = deterioration).
  BETA is ESTIMATED from the modelling set: the log-linear slope of default
  rate against origination FICO. It is not assumed.

  A grade G borrower who drops 80 points now gets a real PD increase,
  because the multiplier has no floor at the bottom of the grade scale.

  Implied grade is retained but ONLY as a narrative label for the agent's
  explanations. It no longer drives staging.

STAGE 3: loan_status in {Default, Late (31-120 days)}. IFRS 9's rebuttable
presumption is 90+ DPD; the data exposes only a 31-120 band that cannot be
split at 90, so adopting the whole band is conservative. Disclosed.

Usage:
    python notebooks\\04_staging.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402

PD_CURVES_PATH = "data/processed/pd_curves.parquet"
STAGED_PATH = "data/processed/live_book_staged.parquet"

# ---- SICR PARAMETERS. Every one is a judgement. Document each. ----
SICR_PD_RATIO = 2.0        # relative limb
SICR_PD_ABS = 0.05         # absolute limb, percentage points of lifetime PD
DPD_BACKSTOP = 30          # IFRS 9 rebuttable presumption
STAGE3_STATUSES = [S.ST_DEFAULT, S.ST_LATE_31_120]

MULT_FLOOR, MULT_CAP = 0.25, 8.0   # bounds on the FICO multiplier
FICO_FLOOR = 300
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def load():
    live = pd.read_parquet(S.LIVE_BOOK_PATH)
    curves = pd.read_parquet(PD_CURVES_PATH)
    print(f"live book: {len(live):,} loans, EAD ${live[S.OUT_PRNCP].sum():,.0f}")
    return live, curves


def estimate_fico_beta() -> float:
    """
    Log-linear slope of default rate against origination FICO, from closed
    loans. Gives PD sensitivity per FICO point. Estimated, not assumed.
    """
    m = pd.read_parquet(S.MODEL_SET_PATH)
    m = m[(m["fico_orig"] >= 600) & (m["fico_orig"] <= 850)]

    m["bin"] = (m["fico_orig"] // 10 * 10).astype(int)
    g = m.groupby("bin").agg(n=("defaulted", "size"), dr=("defaulted", "mean"))
    g = g[(g["n"] >= 500) & (g["dr"] > 0)]

    x = g.index.to_numpy(dtype=float)
    y = np.log(g["dr"].to_numpy())
    w = g["n"].to_numpy(dtype=float)

    beta, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
    beta = -beta  # positive: PD rises as FICO falls

    yhat = intercept - beta * x
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\n--- FICO ELASTICITY (estimated from closed loans) ---")
    print(f"  bins used: {len(g)}  (FICO 600-850, n>=500 each)")
    print(f"  beta = {beta:.5f} per FICO point   weighted R2 = {r2:.3f}")
    print(f"  implied PD multiplier for a drop of:")
    for d in [20, 40, 60, 80, 100]:
        print(f"    {d:>3} points -> {np.exp(beta * d):.2f}x")
    print(f"  sample default rates: "
          f"FICO {int(g.index.min())} -> {g['dr'].iloc[0]:.1%}, "
          f"FICO {int(g.index.max())} -> {g['dr'].iloc[-1]:.1%}")
    return beta


def curve_lookup(curves: pd.DataFrame):
    out = {}
    for (g, t), sub in curves.groupby(["grade", "term_m"]):
        sub = sub.sort_values("month")
        out[(str(g), int(t))] = {"cum_pd": sub["cum_pd"].to_numpy(),
                                 "survival": sub["survival"].to_numpy()}
    return out


def conditional_pd(lut, grade, term, m_on_book, horizon_m):
    out = np.zeros(len(grade))
    for (g, t), arrs in lut.items():
        mask = (grade == g) & (term == t)
        if not mask.any():
            continue
        cpd, surv = arrs["cum_pd"], arrs["survival"]
        m = np.clip(m_on_book[mask].astype(int), 0, t - 1)
        cpd_now = np.where(m == 0, 0.0, cpd[np.maximum(m - 1, 0)])
        s_now = np.where(m == 0, 1.0, surv[np.maximum(m - 1, 0)])
        end = np.full(int(mask.sum()), t) if horizon_m is None \
            else np.minimum(m + horizon_m, t)
        cpd_end = cpd[np.maximum(end - 1, 0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[mask] = np.clip(
                np.where(s_now > 1e-9, (cpd_end - cpd_now) / s_now, 1.0), 0.0, 1.0)
    return out


def narrative_grade(live, lut):
    """Descriptive only: which grade's lifetime PD is closest to this loan's
    current PD. Used by the agent to say 'now behaves like a C'. Not staging."""
    ref = {}
    for g in GRADE_ORDER:
        vals = [lut[(g, t)]["cum_pd"][-1] for t in (36, 60) if (g, t) in lut]
        if vals:
            ref[g] = float(np.mean(vals))
    gs = list(ref.keys())
    arr = np.array([ref[g] for g in gs])
    idx = np.abs(live["pd_lifetime_curr"].to_numpy()[:, None] - arr[None, :]).argmin(1)
    return np.array(gs)[idx]


def main():
    live, curves = load()
    lut = curve_lookup(curves)
    beta = estimate_fico_beta()

    live["grade_str"] = live[S.GRADE].astype(str)
    live["term_i"] = live["term_m"].astype(int)
    live["mob"] = live["months_on_book"].clip(lower=0).astype(int)

    # ---- FICO data quality ------------------------------------------------
    bad = live["fico_curr"].isna() | (live["fico_curr"] < FICO_FLOOR)
    print(f"\n--- CURRENT FICO DATA QUALITY ---")
    print(f"  unusable (0 or <{FICO_FLOOR}): {bad.sum():,} ({bad.mean():.3%})"
          f"  -> multiplier forced to 1.0, DPD backstop still applies")

    delta = live["fico_delta"].to_numpy().astype(float)
    delta[bad.to_numpy()] = 0.0
    live["fico_delta_used"] = delta

    print(f"\n--- FICO DRIFT BY ORIGINATION GRADE ---")
    print(f"  {'grade':<6} {'mean d':>8} {'median d':>9} {'share <=-40':>12}")
    for g in GRADE_ORDER:
        sub = live[live["grade_str"] == g]
        if len(sub) == 0:
            continue
        print(f"  {g:<6} {sub['fico_delta_used'].mean():>8.1f} "
              f"{sub['fico_delta_used'].median():>9.1f} "
              f"{(sub['fico_delta_used'] <= -40).mean():>11.1%}")

    # ---- PD at origination grade, then the continuous multiplier ----------
    print("\nComputing conditional PDs...")
    g_o = live["grade_str"].to_numpy()
    tm = live["term_i"].to_numpy()
    mb = live["mob"].to_numpy()

    live["pd_lifetime_orig"] = conditional_pd(lut, g_o, tm, mb, None)
    pd12_orig = conditional_pd(lut, g_o, tm, mb, 12)

    mult = np.clip(np.exp(beta * -delta), MULT_FLOOR, MULT_CAP)
    live["pd_multiplier"] = mult
    live["pd_lifetime_curr"] = np.clip(live["pd_lifetime_orig"] * mult, 0, 1)
    live["pd_12m"] = np.clip(pd12_orig * mult, 0, 1)

    print(f"  multiplier: mean={mult.mean():.3f}  median={np.median(mult):.3f}  "
          f"p90={np.percentile(mult, 90):.3f}  p99={np.percentile(mult, 99):.3f}")
    print(f"  at floor {MULT_FLOOR}: {(mult <= MULT_FLOOR + 1e-9).mean():.2%}   "
          f"at cap {MULT_CAP}: {(mult >= MULT_CAP - 1e-9).mean():.2%}")

    with np.errstate(divide="ignore", invalid="ignore"):
        live["pd_ratio"] = np.where(live["pd_lifetime_orig"] > 1e-6,
                                    live["pd_lifetime_curr"] / live["pd_lifetime_orig"],
                                    1.0)
    live["pd_delta"] = live["pd_lifetime_curr"] - live["pd_lifetime_orig"]
    live["grade_narrative"] = narrative_grade(live, lut)

    # ---- staging ----------------------------------------------------------
    live["dpd"] = live[S.STATUS].astype(str).map(S.DPD_BAND).astype(float)
    is_s3 = live[S.STATUS].astype(str).isin(STAGE3_STATUSES)
    t_rel = live["pd_ratio"] >= SICR_PD_RATIO
    t_abs = live["pd_delta"] >= SICR_PD_ABS
    t_dpd = live["dpd"] >= DPD_BACKSTOP

    live["sicr_relative"], live["sicr_absolute"], live["sicr_dpd"] = t_rel, t_abs, t_dpd
    live["stage"] = 1
    live.loc[t_rel | t_abs | t_dpd, "stage"] = 2
    live.loc[is_s3, "stage"] = 3

    # ---- diagnostics ------------------------------------------------------
    print("\n--- SICR LIMBS (share of live book) ---")
    print(f"  relative  PD ratio >= {SICR_PD_RATIO}   : {t_rel.mean():6.2%}")
    print(f"  absolute  PD up >= {SICR_PD_ABS:.0%}     : {t_abs.mean():6.2%}")
    print(f"  backstop  DPD >= {DPD_BACKSTOP}         : {t_dpd.mean():6.2%}")
    print(f"  absolute catches, relative misses: {(t_abs & ~t_rel).mean():6.2%}")
    print(f"  relative catches, absolute misses: {(t_rel & ~t_abs).mean():6.2%}")

    print("\n--- STAGE DISTRIBUTION ---")
    total_ead = live[S.OUT_PRNCP].sum()
    g = live.groupby("stage").agg(
        loans=(S.OUT_PRNCP, "size"), ead=(S.OUT_PRNCP, "sum"),
        pd12=("pd_12m", "mean"), pdlife=("pd_lifetime_curr", "mean"))
    for st, r in g.iterrows():
        print(f"  Stage {st}: n={int(r['loans']):>8,} ({r['loans']/len(live):6.2%})  "
              f"EAD=${r['ead']:>15,.0f} ({r['ead']/total_ead:6.2%})  "
              f"meanPD12={r['pd12']:6.2%}  meanPDlife={r['pdlife']:6.2%}")

    print("\n--- STAGE BY ORIGINATION GRADE (EAD share %) ---")
    ct = pd.crosstab(live["grade_str"], live["stage"],
                     values=live[S.OUT_PRNCP], aggfunc="sum", normalize="index")
    print((ct * 100).round(1).to_string())

    if 2 in ct.columns:
        r = ct[2].reindex(GRADE_ORDER).dropna()
        corr = np.corrcoef(np.arange(len(r)), r.to_numpy())[0, 1]
        print(f"\n  correlation of Stage 2 share with grade riskiness: {corr:+.2f}")
        print(f"  v1 = -0.98 (inverted). v3 target: near zero or positive.")
        print(f"  Note: SICR measures CHANGE since origination, so a roughly")
        print(f"  grade-neutral profile is the correct outcome, not a rising one.")

    print("\n--- STAGE 2 SHARE UNDER ALTERNATIVE THRESHOLDS ---")
    for ratio in [1.5, 2.0, 2.5]:
        for absol in [0.03, 0.05, 0.10]:
            s2 = ((live["pd_ratio"] >= ratio) | (live["pd_delta"] >= absol)
                  | t_dpd) & ~is_s3
            print(f"  rel>={ratio:<4} abs>={absol:.0%}: Stage 2 = {s2.mean():6.2%}, "
                  f"${live.loc[s2, S.OUT_PRNCP].sum()/1e9:5.2f}bn EAD")

    print("\n--- SANITY ---")
    print(f"  mean lifetime PD monotonic across stages: "
          f"{g['pdlife'].is_monotonic_increasing}")
    print(f"  every grade has non-zero Stage 2 EAD: "
          f"{bool((ct[2] > 0).all()) if 2 in ct.columns else False}")

    live.to_parquet(STAGED_PATH, index=False)
    print(f"\n--- WRITTEN ---")
    print(f"  {STAGED_PATH}  ({os.path.getsize(STAGED_PATH)/1e6:,.1f} MB)")
    print(f"  beta = {beta:.5f}  <- record this in ecl/config.py")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
