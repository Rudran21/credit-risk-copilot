"""
02_pd_curves.py   (v2 - competing risks)

Builds the PD TERM STRUCTURE by grade and term.

WHY v2: v1 treated prepayment as censoring, which estimates "probability of
default if the loan could never prepay". Roughly half of LC loans DO prepay,
and a loan paid off in month 14 cannot default in month 30. Combined with a
flat EAD assumption that double-counts and overstates the provision.

v2 uses a COMPETING RISKS framework. A loan leaves the book one of two ways:

    h_d[t] = defaults in month t  / at risk entering month t
    h_p[t] = payoffs  in month t  / at risk entering month t
    S[t]   = prod(1 - h_d[k] - h_p[k])  for k <= t     still on book
    mPD[t] = S[t-1] * h_d[t]                           marginal default
    cPD[t] = cumsum(mPD)                               cumulative incidence

cPD at contractual term should now reproduce the observed default rate.
That is the validation check: look at the err column, it should be ~0.

SEASONING: only vintages that could run their full term are used to fit.
36m loans issued <= Dec-2015, 60m loans issued <= Dec-2013.

Usage:
    python notebooks\\02_pd_curves.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402

PD_CURVES_PATH = "data/processed/pd_curves.parquet"

SEASONED_CUTOFF = {36: "2015-12-01", 60: "2013-12-01"}

MIN_N_FIT = 2000     # below this a grade cell falls back to the pooled shape
SMOOTH_WINDOW = 3
MAX_TOTAL_HAZARD = 0.99


def load_seasoned() -> pd.DataFrame:
    df = pd.read_parquet(S.MODEL_SET_PATH)
    print(f"modelling set: {len(df):,} closed loans")

    df = df[df["term_m"].isin([36, 60])].copy()

    keep = pd.Series(False, index=df.index)
    for term, cutoff in SEASONED_CUTOFF.items():
        m = (df["term_m"] == term) & (df["issue_dt"] <= pd.Timestamp(cutoff))
        keep |= m
        print(f"  {term}m issued on/before {cutoff[:7]}: {m.sum():,}")
    df = df[keep].copy()

    df["exit_m"] = df["months_to_last_pymnt"].clip(lower=1)
    df["exit_m"] = np.minimum(df["exit_m"], df["term_m"])
    df = df[df["exit_m"].notna()].copy()
    df["exit_m"] = df["exit_m"].astype(int)

    df["prepaid"] = (df[S.STATUS].astype(str) == S.ST_PAID).astype(int)

    print(f"\nseasoned fitting cohort: {len(df):,} loans")
    print(f"  observed default rate: {df['defaulted'].mean():.2%}")
    print(f"  observed payoff rate:  {df['prepaid'].mean():.2%}")
    print(f"  vintages: {df['vintage_year'].min():.0f} to {df['vintage_year'].max():.0f}")
    return df


def cell_hazards(sub: pd.DataFrame, term: int):
    """Cause-specific hazards for default and payoff, months 1..term."""
    months = np.arange(1, term + 1)
    exits = sub["exit_m"].to_numpy()
    died = sub["defaulted"].to_numpy().astype(bool)
    paid = sub["prepaid"].to_numpy().astype(bool)

    at_risk = np.array([(exits >= m).sum() for m in months], dtype=float)
    n_def = np.array([((exits == m) & died).sum() for m in months], dtype=float)
    n_pay = np.array([((exits == m) & paid).sum() for m in months], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        h_d = np.where(at_risk > 0, n_def / at_risk, 0.0)
        h_p = np.where(at_risk > 0, n_pay / at_risk, 0.0)
    return h_d, h_p, at_risk


def smooth(h: np.ndarray, w: int = SMOOTH_WINDOW) -> np.ndarray:
    s = pd.Series(h).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return np.clip(s, 0.0, 1.0)


def incidence(h_d: np.ndarray, h_p: np.ndarray):
    """Return survival, marginal default PD, cumulative default incidence."""
    total = np.clip(h_d + h_p, 0.0, MAX_TOTAL_HAZARD)
    scale = np.where(h_d + h_p > 0, total / np.maximum(h_d + h_p, 1e-12), 1.0)
    hd, hp = h_d * scale, h_p * scale

    surv = np.cumprod(1.0 - hd - hp)
    surv_prev = np.concatenate([[1.0], surv[:-1]])
    mpd = surv_prev * hd
    return surv, mpd, np.cumsum(mpd), hd, hp


def scale_default_hazard(h_d, h_p, target_cif):
    """Scale the default hazard so cumulative incidence hits a target."""
    if target_cif <= 0:
        return h_d * 0.0

    def cif(k):
        return incidence(k * h_d, h_p)[2][-1]

    lo, hi = 0.0, 50.0
    if cif(hi) < target_cif:
        return hi * h_d
    for _ in range(80):
        mid = (lo + hi) / 2
        if cif(mid) < target_cif:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * h_d


def build(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print("\n--- FITTING (competing risks) ---")

    for term in [36, 60]:
        cohort = df[df["term_m"] == term]
        if len(cohort) == 0:
            continue

        hd_pool, hp_pool, _ = cell_hazards(cohort, term)
        hd_pool, hp_pool = smooth(hd_pool), smooth(hp_pool)
        print(f"\n{term}m pooled: n={len(cohort):,}  "
              f"DR={cohort['defaulted'].mean():.2%}  "
              f"payoff={cohort['prepaid'].mean():.2%}")

        for grade in sorted(cohort[S.GRADE].astype(str).unique()):
            sub = cohort[cohort[S.GRADE].astype(str) == grade]
            n = len(sub)
            obs_dr = sub["defaulted"].mean() if n else 0.0

            hd_raw, hp_raw, at_risk = cell_hazards(sub, term)

            if n < MIN_N_FIT:
                hd = scale_default_hazard(hd_pool, hp_pool, obs_dr)
                hp = hp_pool
                method = "scaled-pool"
            else:
                hd, hp = smooth(hd_raw), smooth(hp_raw)
                method = "empirical"

            surv, mpd, cpd, hd, hp = incidence(hd, hp)

            err = cpd[-1] - obs_dr
            print(f"  {grade} {term}m  n={n:>7,}  obs={obs_dr:6.2%}  "
                  f"model={cpd[-1]:6.2%}  err={err:+.2%}  "
                  f"PD12={cpd[min(11, term-1)]:6.2%}  "
                  f"S@term={surv[-1]:5.1%}  [{method}]")

            for i, m in enumerate(range(1, term + 1)):
                rows.append({
                    "grade": grade, "term_m": term, "month": m,
                    "hazard_default": hd[i], "hazard_prepay": hp[i],
                    "survival": surv[i], "marginal_pd": mpd[i], "cum_pd": cpd[i],
                    "at_risk": at_risk[i], "method": method,
                    "n_fit": n, "observed_lifetime_dr": obs_dr,
                })

    return pd.DataFrame(rows)


def report(curves: pd.DataFrame) -> None:
    print("\n--- PD TERM STRUCTURE (cumulative default incidence) ---")
    print(f"{'seg':<9} {'12m':>8} {'24m':>8} {'36m':>8} {'48m':>8} {'60m':>8}")
    for (g, t), sub in curves.groupby(["grade", "term_m"]):
        sub = sub.set_index("month")
        cells = [f"{sub.loc[h,'cum_pd']:7.2%}" if h in sub.index else "      -"
                 for h in [12, 24, 36, 48, 60]]
        print(f"{g}-{t}m   " + " ".join(cells))

    print("\n--- CALIBRATION: model cPD at term vs observed DR ---")
    fin = curves.sort_values("month").groupby(["grade", "term_m"]).tail(1)
    err = (fin["cum_pd"] - fin["observed_lifetime_dr"]).abs()
    print(f"  max abs error: {err.max():.3%}   mean abs error: {err.mean():.3%}")
    worst = fin.loc[err.idxmax()]
    print(f"  worst cell: {worst['grade']}-{int(worst['term_m'])}m "
          f"(model {worst['cum_pd']:.2%} vs obs {worst['observed_lifetime_dr']:.2%})")

    print("\n--- SHAPE: peak default month, and how many survive to term ---")
    for t in [36, 60]:
        for g in ["A", "C", "F"]:
            sub = curves[(curves.term_m == t) & (curves.grade == g)]
            if len(sub):
                pk = int(sub.loc[sub["hazard_default"].idxmax(), "month"])
                s_end = sub.sort_values("month")["survival"].iloc[-1]
                print(f"  {g}-{t}m: default hazard peaks month {pk:>2},  "
                      f"still on book at term {s_end:5.1%}")

    mono = curves.groupby(["grade", "term_m"])["cum_pd"].apply(
        lambda s: s.is_monotonic_increasing)
    print(f"\n  all cPD curves monotonic increasing: {mono.all()}")


def main():
    df = load_seasoned()
    curves = build(df)
    report(curves)

    os.makedirs("data/processed", exist_ok=True)
    curves.to_parquet(PD_CURVES_PATH, index=False)
    print(f"\n--- WRITTEN ---")
    print(f"  {PD_CURVES_PATH}  ({os.path.getsize(PD_CURVES_PATH)/1e3:,.0f} KB, "
          f"{len(curves):,} rows)")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
