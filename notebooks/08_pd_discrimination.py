"""
08_pd_discrimination.py

OUT-OF-TIME VALIDATION of the PD model: discrimination and calibration.

Why this exists
---------------
Sections 3 and 9 of the validation note report CALIBRATION (does predicted
default rate match observed) but no DISCRIMINATION (does the model rank
riskier borrowers above safer ones). Those are different properties. A model
can be perfectly calibrated on average and rank randomly.

Design
------
OUT-OF-TIME split, not random. A random holdout would leak the economic cycle
between train and test and flatter the result.

    TRAIN  loans issued on or before Dec-2013
    TEST   36-month loans issued Jan-2014 to Dec-2015

Both drawn only from the seasoned cohort, so every test loan had time to run
its full contractual term before the Dec-2018 snapshot.

Two scores compared
-------------------
1. MODEL PD. The engine's lifetime PD for the loan's grade and term. Note this
   is a deliberately low-dimensional model: grade x term gives only 7 distinct
   scores per term. Its discrimination ceiling is set by that design.

2. LOGISTIC BENCHMARK. Logistic regression on origination-only features
   (FICO at origination, rate, DTI, income, amount, term, purpose). Answers
   "what would a conventional scorecard get on the same data?"

No leakage: nothing measured after origination enters either score. In
particular last_fico is EXCLUDED. It is measured at payoff or charge-off, so
defaulted loans have crashed scores by construction and including it would
inflate AUC dishonestly.

Metrics
-------
AUC   probability a random defaulter is ranked riskier than a random non-defaulter
Gini  2 * AUC - 1
KS    maximum separation between the two cumulative distributions

Usage:
    pip install scikit-learn
    python notebooks\\08_pd_discrimination.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from ecl import schema as S  # noqa: E402

PD_CURVES_PATH = "data/processed/pd_curves.parquet"
OUT_PATH = "docs/pd_discrimination.md"

TRAIN_END = "2013-12-01"
TEST_START = "2014-01-01"
TEST_END = "2015-12-01"


# =============================================================================
# metrics, implemented directly so the numbers are inspectable
# =============================================================================
def auc_score(y, score):
    """AUC via the rank (Mann-Whitney) identity. Handles ties correctly."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied groups
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ks_stat(y, score):
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y_ord = y[order]
    cum_bad = np.cumsum(y_ord) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y_ord) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def decile_table(y, score, n=10):
    df = pd.DataFrame({"y": np.asarray(y).astype(int),
                       "s": np.asarray(score, dtype=float)})
    df["band"] = pd.qcut(df["s"].rank(method="first"), n,
                         labels=range(1, n + 1))
    g = df.groupby("band", observed=True).agg(
        n=("y", "size"), defaults=("y", "sum"),
        actual=("y", "mean"), predicted=("s", "mean"))
    g["lift"] = g["actual"] / df["y"].mean()
    return g


# =============================================================================
# data
# =============================================================================
def load_cohort():
    df = pd.read_parquet(S.MODEL_SET_PATH)
    df = df[df["term_m"].isin([36, 60])].copy()

    seasoned = ((df["term_m"] == 36) & (df["issue_dt"] <= pd.Timestamp("2015-12-01"))) | \
               ((df["term_m"] == 60) & (df["issue_dt"] <= pd.Timestamp("2013-12-01")))
    df = df[seasoned].copy()

    train = df[df["issue_dt"] <= pd.Timestamp(TRAIN_END)].copy()
    test = df[(df["term_m"] == 36)
              & (df["issue_dt"] >= pd.Timestamp(TEST_START))
              & (df["issue_dt"] <= pd.Timestamp(TEST_END))].copy()

    print("--- OUT-OF-TIME SPLIT ---")
    print(f"  TRAIN  issued <= {TRAIN_END[:7]}          "
          f"n={len(train):>8,}  default rate {train['defaulted'].mean():.2%}")
    print(f"  TEST   36m issued {TEST_START[:7]} to {TEST_END[:7]}  "
          f"n={len(test):>8,}  default rate {test['defaulted'].mean():.2%}")
    print(f"  no overlap: {len(set(train[S.ID]) & set(test[S.ID])) == 0}")
    return train, test


def attach_model_pd(df):
    """Lifetime PD from the fitted term structure, by grade and term."""
    curves = pd.read_parquet(PD_CURVES_PATH)
    final = (curves.sort_values("month")
             .groupby(["grade", "term_m"]).tail(1)[["grade", "term_m", "cum_pd"]])
    final["grade"] = final["grade"].astype(str)
    final["term_m"] = final["term_m"].astype(int)

    out = df.copy()
    out["grade_s"] = out[S.GRADE].astype(str)
    out["term_i"] = out["term_m"].astype(int)
    out = out.merge(final.rename(columns={"grade": "grade_s",
                                          "term_m": "term_i",
                                          "cum_pd": "model_pd"}),
                    on=["grade_s", "term_i"], how="left")
    return out


# =============================================================================
# benchmark
# =============================================================================
def fit_logistic(train, test):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\n  scikit-learn not installed; skipping the benchmark.")
        print("  pip install scikit-learn")
        return None, None

    num = ["fico_orig", S.INT_RATE, S.DTI, "term_m", S.FUNDED_AMNT]
    top_purposes = train[S.PURPOSE].astype(str).value_counts().head(6).index.tolist()

    def design(df):
        X = df[num].copy()
        X["log_income"] = np.log1p(df[S.ANNUAL_INC].clip(lower=0))
        for p in top_purposes:
            X[f"purp_{p}"] = (df[S.PURPOSE].astype(str) == p).astype(int)
        return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    Xtr, ytr = design(train), train["defaulted"].astype(int)
    Xte = design(test)

    model = Pipeline([("scale", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    model.fit(Xtr, ytr)

    print(f"\n  benchmark features ({Xtr.shape[1]}): {list(Xtr.columns)}")
    coefs = pd.Series(model.named_steps["lr"].coef_[0], index=Xtr.columns)
    print("  strongest coefficients (standardised):")
    for k, v in coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(5).items():
        print(f"    {k:<22} {v:+.4f}")

    return model.predict_proba(Xte)[:, 1], model.predict_proba(Xtr)[:, 1]


# =============================================================================
def report(name, y, score):
    a = auc_score(y, score)
    print(f"\n  {name}")
    print(f"    AUC  {a:.4f}")
    print(f"    Gini {2*a-1:.4f}")
    print(f"    KS   {ks_stat(y, score):.4f}")
    print(f"    distinct scores: {len(np.unique(score)):,}")
    return {"auc": a, "gini": 2 * a - 1, "ks": ks_stat(y, score),
            "distinct": int(len(np.unique(score)))}


def main():
    train, test = load_cohort()
    train = attach_model_pd(train)
    test = attach_model_pd(test)

    if test["model_pd"].isna().any():
        print(f"  !! {test['model_pd'].isna().sum()} test loans have no curve")
        test = test[test["model_pd"].notna()].copy()

    y_te = test["defaulted"].astype(int).to_numpy()
    y_tr = train["defaulted"].astype(int).to_numpy()

    print("\n" + "=" * 66)
    print("DISCRIMINATION")
    print("=" * 66)

    m_te = report("MODEL PD, out-of-time test", y_te, test["model_pd"].to_numpy())
    m_tr = report("MODEL PD, in-sample train (reference)", y_tr,
                  train["model_pd"].to_numpy())

    bench_te, bench_tr = fit_logistic(train, test)
    b_te = b_tr = None
    if bench_te is not None:
        b_te = report("LOGISTIC BENCHMARK, out-of-time test", y_te, bench_te)
        b_tr = report("LOGISTIC BENCHMARK, in-sample train", y_tr, bench_tr)

    print("\n" + "=" * 66)
    print("CALIBRATION ON THE OUT-OF-TIME TEST SET")
    print("=" * 66)
    dt = decile_table(y_te, test["model_pd"].to_numpy())
    print(f"  {'band':<6} {'n':>8} {'predicted':>11} {'actual':>9} {'lift':>7}")
    for band, r in dt.iterrows():
        print(f"  {int(band):<6} {int(r['n']):>8,} {r['predicted']:>10.2%} "
              f"{r['actual']:>9.2%} {r['lift']:>7.2f}x")

    overall_pred = float(test["model_pd"].mean())
    overall_act = float(y_te.mean())
    print(f"\n  overall predicted {overall_pred:.2%} vs actual {overall_act:.2%}"
          f"   ({(overall_pred-overall_act)*100:+.2f}pp)")

    print("\n" + "=" * 66)
    print("READING THIS")
    print("=" * 66)
    print("  The model PD uses grade x term only, so 36-month loans carry just")
    print("  7 distinct scores. Its discrimination ceiling is set by that")
    print("  design choice, not by the fitting method. The benchmark shows what")
    print("  a conventional scorecard extracts from the same origination data.")
    print("  Calibration, not discrimination, is what this model was built for:")
    print("  ECL requires the predicted default RATE to be right, whereas")
    print("  ranking matters most for approve/decline decisions.")

    write_md(m_te, m_tr, b_te, b_tr, dt, len(train), len(test),
             y_tr.mean(), y_te.mean(), overall_pred, overall_act)
    print(f"\n  written to {OUT_PATH}")
    print("\n=== DONE - paste this whole output into the chat ===")


def write_md(m_te, m_tr, b_te, b_tr, dt, n_tr, n_te, dr_tr, dr_te,
             pred, act):
    L = ["# PD Discrimination and Out-of-Time Validation\n"]
    L.append("## Design\n")
    L.append("Out-of-time split, not random, so the economic cycle cannot leak "
             "between train and test.\n")
    L.append("| | Definition | n | Default rate |")
    L.append("|---|---|---|---|")
    L.append(f"| Train | issued on or before {TRAIN_END[:7]} | {n_tr:,} | {dr_tr:.2%} |")
    L.append(f"| Test | 36m issued {TEST_START[:7]} to {TEST_END[:7]} | {n_te:,} | {dr_te:.2%} |\n")
    L.append("Both drawn from the seasoned cohort, so every test loan had time "
             "to run its full contractual term before the snapshot. No feature "
             "measured after origination enters either score; `last_fico` is "
             "excluded because it is observed at payoff or charge-off and would "
             "leak the outcome.\n")

    L.append("## Results\n")
    L.append("| Model | Sample | AUC | Gini | KS | Distinct scores |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| Grade x term PD | **out-of-time** | **{m_te['auc']:.4f}** | "
             f"**{m_te['gini']:.4f}** | **{m_te['ks']:.4f}** | {m_te['distinct']} |")
    L.append(f"| Grade x term PD | in-sample | {m_tr['auc']:.4f} | "
             f"{m_tr['gini']:.4f} | {m_tr['ks']:.4f} | {m_tr['distinct']} |")
    if b_te:
        L.append(f"| Logistic benchmark | **out-of-time** | **{b_te['auc']:.4f}** | "
                 f"**{b_te['gini']:.4f}** | **{b_te['ks']:.4f}** | {b_te['distinct']:,} |")
        L.append(f"| Logistic benchmark | in-sample | {b_tr['auc']:.4f} | "
                 f"{b_tr['gini']:.4f} | {b_tr['ks']:.4f} | {b_tr['distinct']:,} |")
    L.append("")

    L.append("## Calibration on the out-of-time test set\n")
    L.append("| Band | n | Predicted | Actual | Lift |")
    L.append("|---|---|---|---|---|")
    for band, r in dt.iterrows():
        L.append(f"| {int(band)} | {int(r['n']):,} | {r['predicted']:.2%} | "
                 f"{r['actual']:.2%} | {r['lift']:.2f}x |")
    L.append(f"\nOverall predicted **{pred:.2%}** versus actual **{act:.2%}**, "
             f"a gap of **{(pred-act)*100:+.2f}pp** on loans the model never saw.\n")

    L.append("## Interpretation\n")
    L.append("The PD model is deliberately low-dimensional: grade and term "
             "only, giving seven distinct scores per term. Its discrimination "
             "ceiling is set by that design choice rather than by the fitting "
             "method. The logistic benchmark shows what a conventional "
             "scorecard extracts from the same origination fields.\n")
    L.append("For expected credit loss, calibration matters more than ranking. "
             "ECL is a provision, so the predicted default RATE has to be "
             "right in level. Ranking power matters most for approve/decline "
             "decisions, which this model does not make. Both are reported "
             "because a model that is calibrated on average can still rank "
             "randomly, and the two properties should never be conflated.\n")

    os.makedirs("docs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
