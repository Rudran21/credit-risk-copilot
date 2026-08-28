"""
00_schema_audit.py

Audits the Lending Club accepted-loans file. Writes nothing, changes nothing.
Run this BEFORE any modelling code exists. Its whole job is to replace
assumptions about column names with facts.

Usage (from project root, venv active):
    python notebooks\\00_schema_audit.py
"""

import gzip
import os
import sys

import pandas as pd

RAW = os.path.join("data", "raw", "accepted_raw.csv")

# Columns the ECL engine cannot work without.
REQUIRED = [
    "id", "issue_d", "loan_status", "term", "grade", "sub_grade",
    "int_rate", "funded_amnt", "out_prncp", "total_rec_prncp",
    "recoveries", "last_pymnt_d", "installment",
]

# Columns that drive the SICR trigger and segmentation. Degrade gracefully.
OPTIONAL = [
    "fico_range_low", "fico_range_high",
    "last_fico_range_low", "last_fico_range_high",
    "dti", "annual_inc", "purpose", "addr_state", "application_type",
]


def main():
    if not os.path.exists(RAW):
        print(f"FILE NOT FOUND: {RAW}")
        print("Check the filename. List what is actually there:")
        raw_dir = os.path.join("data", "raw")
        if os.path.isdir(raw_dir):
            for f in os.listdir(raw_dir):
                print("   ", f)
        sys.exit(1)

    size_mb = os.path.getsize(RAW) / 1e6
    print(f"FILE: {RAW}  ({size_mb:,.0f} MB compressed)\n")

    # ---- 1. Header only -----------------------------------------------
    header = pd.read_csv(RAW, nrows=0, low_memory=False)
    cols = list(header.columns)
    print(f"TOTAL COLUMNS: {len(cols)}\n")
    print("ALL COLUMN NAMES:")
    print(cols)
    print()

    missing_req = [c for c in REQUIRED if c not in cols]
    missing_opt = [c for c in OPTIONAL if c not in cols]
    print("REQUIRED MISSING:", missing_req if missing_req else "none - good")
    print("OPTIONAL MISSING:", missing_opt if missing_opt else "none - good")

    if missing_req:
        print("\n>>> STOP. Paste the ALL COLUMN NAMES list above into the chat.")
        print(">>> The engine design depends on finding the real names first.")
        sys.exit(1)

    # ---- 2. Sample for dtypes and null rates ---------------------------
    use = REQUIRED + [c for c in OPTIONAL if c in cols]
    print("\nReading 200k-row sample...")
    s = pd.read_csv(RAW, usecols=use, nrows=200_000, low_memory=False)

    print("\n--- DTYPES AND NULL RATES (200k sample) ---")
    for c in use:
        print(f"{c:<24} {str(s[c].dtype):<10} null={s[c].isna().mean():7.2%}")

    # ---- 3. Fields the design hinges on --------------------------------
    print("\n--- loan_status VALUE COUNTS ---")
    print(s["loan_status"].value_counts(dropna=False))

    print("\n--- term RAW VALUES (watch for leading space) ---")
    print(s["term"].value_counts(dropna=False))

    print("\n--- DATE / RATE RAW FORMAT ---")
    print(s[["issue_d", "last_pymnt_d", "int_rate"]].head())
    print("int_rate dtype:", s["int_rate"].dtype)

    print("\n--- out_prncp BY STATUS (is it zero for terminal loans?) ---")
    g = s.groupby("loan_status")["out_prncp"].agg(
        n="count", mean="mean", max="max",
        pct_nonzero=lambda x: (x > 0).mean(),
    )
    print(g)

    print("\n--- recoveries ON CHARGED OFF ONLY ---")
    co = s[s["loan_status"] == "Charged Off"]
    print(f"n = {len(co)}")
    if len(co):
        print(co["recoveries"].describe())
        print("pct with zero recovery:", round((co["recoveries"] == 0).mean(), 4))

    # ---- 4. Full-file row count, streamed ------------------------------
    print("\nCounting rows in full file (streamed, takes a minute)...")
    with open(RAW, "r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f) - 1
    print(f"TOTAL ROWS: {n:,}")

    print("\n=== AUDIT COMPLETE - paste this whole output into the chat ===")


if __name__ == "__main__":
    main()
