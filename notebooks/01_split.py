"""
01_split.py

Turns 2.26M raw rows into two clean tables:

  data/processed/model_set.parquet   closed loans, known outcome -> fit PD & LGD
  data/processed/live_book.parquet   open loans at snapshot      -> compute ECL

This mirrors how a bank actually does IFRS 9: estimate on closed cohorts,
apply to the current book.

Usage (from project root, venv active):
    python notebooks\\01_split.py

Runtime: 2-4 minutes, mostly the CSV read. Peak memory ~2 GB.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402


def load_raw() -> pd.DataFrame:
    print(f"Reading {S.RAW_PATH} (23 of 151 columns)...")
    df = pd.read_csv(S.RAW_PATH, usecols=S.USECOLS, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    # LC files carry trailing junk rows (footer text, blank lines).
    # Anything without an issue date is not a loan.
    before = len(df)
    df = df[df[S.ISSUE_D].notna()].copy()
    if before - len(df):
        print(f"  dropped {before - len(df):,} rows with no issue_d (footer junk)")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning...")

    # term: " 36 months" or "36 months" -> 36
    df["term_m"] = df[S.TERM].astype(str).str.extract(r"(\d+)").astype(float)

    # dates
    df["issue_dt"] = pd.to_datetime(df[S.ISSUE_D], format=S.DATE_FMT, errors="coerce")
    df["last_pymnt_dt"] = pd.to_datetime(
        df[S.LAST_PYMNT_D], format=S.DATE_FMT, errors="coerce"
    )

    # months on book at the point the loan stopped paying (proxy for default month)
    df["months_to_last_pymnt"] = (
        (df["last_pymnt_dt"].dt.year - df["issue_dt"].dt.year) * 12
        + (df["last_pymnt_dt"].dt.month - df["issue_dt"].dt.month)
    )

    # months on book as at the snapshot
    snap = pd.Timestamp(S.SNAPSHOT_DATE)
    df["months_on_book"] = (
        (snap.year - df["issue_dt"].dt.year) * 12
        + (snap.month - df["issue_dt"].dt.month)
    )

    df["vintage_year"] = df["issue_dt"].dt.year

    # FICO midpoints: origination vs most recent. The gap is the SICR signal.
    df["fico_orig"] = (df[S.FICO_LOW] + df[S.FICO_HIGH]) / 2
    df["fico_curr"] = (df[S.LAST_FICO_LOW] + df[S.LAST_FICO_HIGH]) / 2
    df["fico_delta"] = df["fico_curr"] - df["fico_orig"]

    # effective interest rate, monthly, for discounting
    df["eir_annual"] = df[S.INT_RATE] / 100.0

    for c in [S.GRADE, S.SUB_GRADE, S.STATUS, S.PURPOSE, S.STATE, S.APP_TYPE]:
        df[c] = df[c].astype("category")

    return df


def report_status(df: pd.DataFrame) -> None:
    print("\n--- FULL-FILE loan_status (this is the real distribution) ---")
    vc = df[S.STATUS].value_counts(dropna=False)
    for k, v in vc.items():
        print(f"  {str(k):<22} {v:>10,}  ({v / len(df):6.2%})")

    unknown = set(vc.index.astype(str)) - set(S.TERMINAL_STATUSES) - set(S.LIVE_STATUSES)
    if unknown:
        print(f"\n  !! UNCLASSIFIED STATUSES: {unknown}")
        print("  !! These fall through the split. Tell me before continuing.")


def split(df: pd.DataFrame):
    model = df[df[S.STATUS].isin(S.TERMINAL_STATUSES)].copy()
    live = df[df[S.STATUS].isin(S.LIVE_STATUSES)].copy()

    model["defaulted"] = (model[S.STATUS] == S.ST_CHARGED_OFF).astype(int)
    live["dpd"] = live[S.STATUS].map(S.DPD_BAND).astype(float)

    # remaining contractual life, floored at 1 month
    live["remaining_m"] = (live["term_m"] - live["months_on_book"]).clip(lower=1)

    return model, live


def report_model_set(m: pd.DataFrame) -> None:
    print("\n--- MODELLING SET (closed loans, used to fit PD and LGD) ---")
    print(f"  loans: {len(m):,}")
    print(f"  overall default rate: {m['defaulted'].mean():.2%}")

    print("\n  default rate by grade:")
    g = m.groupby(S.GRADE, observed=True).agg(
        n=("defaulted", "size"), dr=("defaulted", "mean")
    )
    for grade, row in g.iterrows():
        print(f"    {grade}  n={int(row['n']):>8,}  DR={row['dr']:7.2%}")

    print("\n  default rate by term:")
    for t, row in m.groupby("term_m", observed=True)["defaulted"].agg(["size", "mean"]).iterrows():
        print(f"    {int(t)}m  n={int(row['size']):>8,}  DR={row['mean']:7.2%}")

    co = m[m["defaulted"] == 1]
    print(f"\n  time-to-default (months, from issue to last payment), n={len(co):,}:")
    print(f"    median={co['months_to_last_pymnt'].median():.0f}  "
          f"p25={co['months_to_last_pymnt'].quantile(.25):.0f}  "
          f"p75={co['months_to_last_pymnt'].quantile(.75):.0f}  "
          f"max={co['months_to_last_pymnt'].max():.0f}")

    ead_at_default = co[S.FUNDED_AMNT] - co[S.TOTAL_REC_PRNCP]
    net_rec = (co[S.RECOVERIES] - co[S.COLLECTION_FEE]).clip(lower=0)
    lgd = (1 - net_rec / ead_at_default.replace(0, np.nan)).clip(0, 1)
    print(f"\n  implied LGD (1 - net recovery / principal outstanding at default):")
    print(f"    mean={lgd.mean():.3f}  median={lgd.median():.3f}  "
          f"p10={lgd.quantile(.10):.3f}  p90={lgd.quantile(.90):.3f}")


def report_live_book(l: pd.DataFrame) -> None:
    print("\n--- LIVE BOOK (open at 2018-12-31, this is the balance sheet) ---")
    print(f"  loans: {len(l):,}")
    print(f"  total outstanding principal (EAD): ${l[S.OUT_PRNCP].sum():,.0f}")
    print(f"  mean exposure: ${l[S.OUT_PRNCP].mean():,.0f}")

    print("\n  by status:")
    for st, row in l.groupby(S.STATUS, observed=True)[S.OUT_PRNCP].agg(["size", "sum"]).iterrows():
        if row["size"] == 0:
            continue
        print(f"    {str(st):<22} n={int(row['size']):>8,}  EAD=${row['sum']:>15,.0f}")

    print("\n  by grade:")
    for gr, row in l.groupby(S.GRADE, observed=True)[S.OUT_PRNCP].agg(["size", "sum"]).iterrows():
        print(f"    {gr}  n={int(row['size']):>8,}  EAD=${row['sum']:>15,.0f}")

    print("\n  by vintage year:")
    for yr, row in l.groupby("vintage_year")[S.OUT_PRNCP].agg(["size", "sum"]).iterrows():
        print(f"    {int(yr)}  n={int(row['size']):>8,}  EAD=${row['sum']:>15,.0f}")

    print("\n  FICO drift since origination (the SICR signal):")
    print(f"    mean={l['fico_delta'].mean():+.1f}  median={l['fico_delta'].median():+.1f}")
    for thr in [0, -20, -40, -60, -80]:
        pct = (l["fico_delta"] <= thr).mean()
        print(f"    share with drop of {abs(thr):>2}+ points: {pct:6.2%}")

    print("\n  remaining life (months):")
    print(f"    median={l['remaining_m'].median():.0f}  "
          f"min={l['remaining_m'].min():.0f}  max={l['remaining_m'].max():.0f}")


def main():
    if not os.path.exists(S.RAW_PATH):
        print(f"NOT FOUND: {S.RAW_PATH}")
        sys.exit(1)

    df = load_raw()
    df = clean(df)
    report_status(df)

    model, live = split(df)
    report_model_set(model)
    report_live_book(live)

    os.makedirs("data/processed", exist_ok=True)
    model.to_parquet(S.MODEL_SET_PATH, index=False)
    live.to_parquet(S.LIVE_BOOK_PATH, index=False)

    print(f"\n--- WRITTEN ---")
    for p in [S.MODEL_SET_PATH, S.LIVE_BOOK_PATH]:
        print(f"  {p}  ({os.path.getsize(p) / 1e6:,.1f} MB)")

    print("\n=== DONE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
