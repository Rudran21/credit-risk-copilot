"""
07_build_deploy_artifact.py

Trims the staged live book into something Streamlit Community Cloud can hold.

THE PROBLEM
  live_book_staged.parquet is ~43 MB on disk and roughly 45 columns across
  912,609 loans. Loaded into pandas as float64 that is several hundred MB,
  before the three cached scenario columns the engine adds. The free tier has
  limited memory and will restart the app rather than warn you.

THE FIX
  Keep every loan, drop every column the engine and app never read, and
  downcast what remains. No sampling: the headline provision must stay exactly
  $905,350,344, and sampling would change it.

Usage:
    python notebooks\\07_build_deploy_artifact.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402

STAGED_PATH = "data/processed/live_book_staged.parquet"
DEPLOY_PATH = "data/processed/live_book_deploy.parquet"

# Everything the engine and app actually touch. Anything not here is dropped.
KEEP = [
    S.ID, S.OUT_PRNCP, S.INT_RATE, S.STATUS, S.PURPOSE, S.STATE,
    "grade_str", "term_m", "stage", "mob", "remaining_m", "months_on_book",
    "eir_annual", "vintage_year", "dpd",
    "pd_multiplier", "pd_12m", "pd_lifetime_curr", "pd_lifetime_orig",
    "pd_ratio", "pd_delta",
    "fico_orig", "fico_curr", "fico_delta_used",
    "grade_narrative",
    "sicr_relative", "sicr_absolute", "sicr_dpd",
]

FLOAT32 = [
    S.INT_RATE, "eir_annual", "pd_multiplier", "pd_12m", "pd_lifetime_curr",
    "pd_lifetime_orig", "pd_ratio", "pd_delta", "fico_orig", "fico_curr",
    "fico_delta_used", "dpd",
]
INT16 = ["term_m", "stage", "mob", "remaining_m", "months_on_book",
         "vintage_year"]
CATEGORY = [S.STATUS, S.PURPOSE, S.STATE, "grade_str", "grade_narrative"]


def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1e6


def main():
    if not os.path.exists(STAGED_PATH):
        print(f"NOT FOUND: {STAGED_PATH}. Run notebooks/04_staging.py first.")
        sys.exit(1)

    df = pd.read_parquet(STAGED_PATH)
    print(f"source: {len(df):,} loans, {len(df.columns)} columns")
    print(f"  on disk : {os.path.getsize(STAGED_PATH)/1e6:,.1f} MB")
    print(f"  in memory: {mem_mb(df):,.0f} MB")

    ead_before = float(df[S.OUT_PRNCP].sum())
    n_before = len(df)

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        print(f"\n!! columns expected but absent: {missing}")
        print("!! re-run 04_staging.py; do not deploy from this file")
        sys.exit(1)

    out = df[KEEP].copy()

    # out_prncp stays float64: it drives the headline currency figure and
    # float32 would introduce rounding at the cent level across 912k rows.
    out[S.OUT_PRNCP] = out[S.OUT_PRNCP].astype("float64")

    for c in FLOAT32:
        if c in out.columns:
            out[c] = out[c].astype("float32")
    for c in INT16:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype("int16")
    for c in CATEGORY:
        if c in out.columns:
            out[c] = out[c].astype("category")
    for c in ["sicr_relative", "sicr_absolute", "sicr_dpd"]:
        if c in out.columns:
            out[c] = out[c].astype("bool")
    out[S.ID] = out[S.ID].astype(str)

    # ---- integrity: the book must be unchanged --------------------------
    print("\n--- INTEGRITY ---")
    ead_after = float(out[S.OUT_PRNCP].sum())
    print(f"  loans  : {n_before:,} -> {len(out):,}")
    print(f"  EAD    : ${ead_before:,.2f} -> ${ead_after:,.2f}")
    ok = (len(out) == n_before) and abs(ead_after - ead_before) < 0.01
    print(f"  identical: {ok}")
    if not ok:
        print("  !! do not deploy this artifact")
        sys.exit(1)

    out.to_parquet(DEPLOY_PATH, index=False, compression="zstd")

    print("\n--- RESULT ---")
    print(f"  columns  : {len(df.columns)} -> {len(out.columns)}")
    print(f"  on disk  : {os.path.getsize(STAGED_PATH)/1e6:,.1f} MB -> "
          f"{os.path.getsize(DEPLOY_PATH)/1e6:,.1f} MB")
    print(f"  in memory: {mem_mb(df):,.0f} MB -> {mem_mb(out):,.0f} MB")

    est = mem_mb(out) + 3 * len(out) * 8 / 1e6
    print(f"\n  estimated peak with 3 cached scenario columns: {est:,.0f} MB")
    if est > 700:
        print("  WARNING: still large for the free tier. Consider dropping")
        print("  purpose/state, or filtering to a subset of vintages.")
    else:
        print("  fits the free tier with headroom.")

    print(f"\n  wrote {DEPLOY_PATH}")
    print("\n  Next: engine.py prefers this file automatically if present.")
    print("  Verify the headline is unchanged:")
    print("    python notebooks\\05_run_ecl.py")
    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
