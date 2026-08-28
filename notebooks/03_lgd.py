"""
03_lgd.py

Estimates LOSS GIVEN DEFAULT from realised recoveries on charged-off loans.

    EAD_at_default = funded_amnt - total_rec_prncp      principal still owed
    net_recovery   = recoveries - collection_recovery_fee
    LGD            = 1 - PV(net_recovery) / EAD_at_default

TWO CORRECTIONS THAT MATTER:

1. RECOVERY SEASONING BIAS.
   Recoveries are measured as at the Dec-2018 snapshot. A loan charged off in
   mid-2018 has had almost no time to collect, so its recovery looks like zero
   and its LGD looks like 1.00. Including those loans biases portfolio LGD
   upward. We measure that bias explicitly, then fit only on charge-offs with
   a recovery window of at least RECOVERY_WINDOW_M months.

2. DISCOUNTING.
   IFRS 9 requires recoveries discounted to the default date at the original
   effective interest rate. Collection timing is not in this dataset, so we
   assume recoveries arrive RECOVERY_LAG_M months after default and discount
   at the loan's own interest rate. That is an assumption, not a measurement.
   It is a named, single-parameter lever and its sensitivity is reported.

CHARGE-OFF DATE PROXY: LendingClub charges off at roughly 120+ DPD, so the
charge-off date is taken as last payment date + CHARGEOFF_LAG_M months.

Usage:
    python notebooks\\03_lgd.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402

LGD_PATH = "data/processed/lgd_table.parquet"

CHARGEOFF_LAG_M = 4        # last payment -> charge-off
RECOVERY_WINDOW_M = 24     # months of collection history required to be "seasoned"
RECOVERY_LAG_M = 18        # assumed timing of recovery cash, months after default
MIN_N_CELL = 500           # below this a grade cell falls back to portfolio LGD


def load_chargeoffs() -> pd.DataFrame:
    df = pd.read_parquet(S.MODEL_SET_PATH)
    co = df[df[S.STATUS].astype(str) == S.ST_CHARGED_OFF].copy()
    print(f"charged-off loans: {len(co):,}")

    co["chargeoff_dt"] = co["last_pymnt_dt"] + pd.DateOffset(months=CHARGEOFF_LAG_M)
    snap = pd.Timestamp(S.SNAPSHOT_DATE)
    co["recovery_window_m"] = (
        (snap.year - co["chargeoff_dt"].dt.year) * 12
        + (snap.month - co["chargeoff_dt"].dt.month)
    )

    co["ead_at_default"] = co[S.FUNDED_AMNT] - co[S.TOTAL_REC_PRNCP]
    co["net_recovery"] = (co[S.RECOVERIES] - co[S.COLLECTION_FEE]).clip(lower=0)

    before = len(co)
    co = co[(co["ead_at_default"] > 0) & co["recovery_window_m"].notna()].copy()
    print(f"  dropped {before - len(co):,} with non-positive EAD or missing dates")
    return co


def add_lgd(co: pd.DataFrame) -> pd.DataFrame:
    co["recovery_rate_raw"] = (co["net_recovery"] / co["ead_at_default"]).clip(0, 1)
    co["lgd_raw"] = 1.0 - co["recovery_rate_raw"]

    # discount recovery cash back to the default date at the loan's own EIR
    disc = (1.0 + co["eir_annual"]) ** (RECOVERY_LAG_M / 12.0)
    co["pv_recovery"] = co["net_recovery"] / disc
    co["recovery_rate_pv"] = (co["pv_recovery"] / co["ead_at_default"]).clip(0, 1)
    co["lgd_pv"] = 1.0 - co["recovery_rate_pv"]
    return co


def report_seasoning(co: pd.DataFrame) -> pd.DataFrame:
    print("\n--- RECOVERY SEASONING BIAS ---")
    print(f"{'window':<14} {'n':>9} {'mean LGD':>10} {'zero-recovery share':>21}")
    bands = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 999)]
    for lo, hi in bands:
        m = (co["recovery_window_m"] >= lo) & (co["recovery_window_m"] < hi)
        sub = co[m]
        if len(sub) == 0:
            continue
        zero = (sub["net_recovery"] == 0).mean()
        print(f"{lo:>3}-{hi if hi < 999 else '+':<9} {len(sub):>9,} "
              f"{sub['lgd_raw'].mean():>10.3f} {zero:>20.1%}")

    seasoned = co[co["recovery_window_m"] >= RECOVERY_WINDOW_M].copy()
    print(f"\n  all charge-offs      : LGD = {co['lgd_raw'].mean():.4f}  (n={len(co):,})")
    print(f"  seasoned {RECOVERY_WINDOW_M}m+ only  : LGD = {seasoned['lgd_raw'].mean():.4f}  "
          f"(n={len(seasoned):,})")
    bias = co["lgd_raw"].mean() - seasoned["lgd_raw"].mean()
    print(f"  upward bias from unseasoned cases: {bias:+.4f}")
    print(f"\n  fitting on the seasoned subset.")
    return seasoned


def build_table(seasoned: pd.DataFrame) -> pd.DataFrame:
    port_lgd = seasoned["lgd_pv"].mean()
    print(f"\n--- LGD BY GRADE (discounted, {RECOVERY_LAG_M}m recovery lag) ---")
    print(f"{'grade':<6} {'n':>8} {'LGD raw':>9} {'LGD PV':>9} {'rec rate':>9} "
          f"{'zero-rec':>9} {'method':>13}")

    rows = []
    for grade in sorted(seasoned[S.GRADE].astype(str).unique()):
        sub = seasoned[seasoned[S.GRADE].astype(str) == grade]
        n = len(sub)
        if n < MIN_N_CELL:
            lgd_pv, lgd_raw, rec, zero = port_lgd, np.nan, np.nan, np.nan
            method = "portfolio"
        else:
            lgd_pv = sub["lgd_pv"].mean()
            lgd_raw = sub["lgd_raw"].mean()
            rec = sub["recovery_rate_pv"].mean()
            zero = (sub["net_recovery"] == 0).mean()
            method = "grade"
        print(f"{grade:<6} {n:>8,} "
              f"{lgd_raw if not np.isnan(lgd_raw) else float('nan'):>9.4f} "
              f"{lgd_pv:>9.4f} "
              f"{rec if not np.isnan(rec) else float('nan'):>9.4f} "
              f"{zero if not np.isnan(zero) else float('nan'):>9.1%} {method:>13}")
        rows.append({"grade": grade, "n": n, "lgd": lgd_pv,
                     "lgd_undiscounted": lgd_raw, "method": method})

    rows.append({"grade": "PORTFOLIO", "n": len(seasoned), "lgd": port_lgd,
                 "lgd_undiscounted": seasoned["lgd_raw"].mean(), "method": "portfolio"})
    return pd.DataFrame(rows)


def report_extras(seasoned: pd.DataFrame) -> None:
    print("\n--- LGD BY TERM (does term matter enough to segment on?) ---")
    for t, sub in seasoned.groupby("term_m"):
        print(f"  {int(t)}m  n={len(sub):>7,}  LGD_PV={sub['lgd_pv'].mean():.4f}")

    print("\n--- LGD DISTRIBUTION (seasoned, discounted) ---")
    q = seasoned["lgd_pv"].describe(percentiles=[.1, .25, .5, .75, .9])
    for k in ["mean", "10%", "25%", "50%", "75%", "90%"]:
        print(f"  {k:>5}: {q[k]:.4f}")
    print(f"  share with LGD = 1.00 (no recovery at all): "
          f"{(seasoned['lgd_pv'] >= 0.9999).mean():.1%}")

    print(f"\n--- SENSITIVITY TO THE RECOVERY-LAG ASSUMPTION ---")
    for lag in [0, 6, 12, 18, 24, 36]:
        disc = (1.0 + seasoned["eir_annual"]) ** (lag / 12.0)
        pv = seasoned["net_recovery"] / disc
        lgd = (1.0 - (pv / seasoned["ead_at_default"]).clip(0, 1)).mean()
        mark = "  <- assumed" if lag == RECOVERY_LAG_M else ""
        print(f"  lag {lag:>2}m -> portfolio LGD {lgd:.4f}{mark}")


def main():
    co = load_chargeoffs()
    co = add_lgd(co)
    seasoned = report_seasoning(co)
    table = build_table(seasoned)
    report_extras(seasoned)

    os.makedirs("data/processed", exist_ok=True)
    table.to_parquet(LGD_PATH, index=False)
    print(f"\n--- WRITTEN ---")
    print(f"  {LGD_PATH}  ({len(table)} rows)")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
