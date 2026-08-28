# Model Validation Note

**Model.** IFRS 9 expected credit loss, US unsecured consumer loans
**Portfolio.** LendingClub accepted loans, open positions as at 31 December 2018
**Exposure.** $9,510,384,856 across 912,609 loans
**Provision.** $905.4m probability-weighted, 9.52% coverage
**Prepared by.** Rudran | **Status.** Personal project, not production

---

## 1. Scope and purpose

Estimates lifetime and 12-month expected credit loss under IFRS 9 on a public
consumer loan book. Every figure is produced by a deterministic engine
(`ecl/`). A language model sits on top as an orchestration and explanation
layer only; it holds no model logic and performs no arithmetic.

This note records what was assumed, what was estimated, and where the model
should not be trusted.

---

## 2. Data

| | |
|---|---|
| Source | LendingClub accepted loans 2007-2018Q4, Kaggle `wordsforthewise/lending-club`, CC0 |
| Raw records | 2,260,701 |
| Excluded, no issue date | 33 (file footer artefacts) |
| Excluded, off-policy | 2,749 ("Does not meet the credit policy") |
| Modelling set (closed loans) | 1,345,310 |
| Live book (open at snapshot) | 912,609 |

**Population split.** The file is a snapshot, not a panel. There is no monthly
performance history. Loans in a terminal state (`Fully Paid`, `Charged Off`)
form the modelling set used to estimate PD and LGD. Loans open at the snapshot
(`Current`, `In Grace Period`, `Late`, `Default`) form the live book on which
ECL is computed. This mirrors standard practice: estimate on closed cohorts,
apply to the current book.

The split was validated: `out_prncp` is exactly 0.00 for every terminal loan
and non-zero for 99%+ of open loans.

**Off-policy exclusion.** 2,749 loans flagged "Does not meet the credit policy"
are 2007-2010 originations LendingClub later deemed off-standard. They were
underwritten under a different regime, so fitting on them and applying the
result to a 2018 book is a policy mismatch. 0.12% of records.

---

## 3. PD: competing-risks term structure

**Method.** Discrete-time hazard with two competing exits.

```
h_d[t] = defaults in month t / at risk entering month t
h_p[t] = payoffs  in month t / at risk entering month t
S[t]   = prod(1 - h_d[k] - h_p[k])          still on book
mPD[t] = S[t-1] * h_d[t]                     marginal default
cPD[t] = cumsum(mPD)                         cumulative incidence
```

12-month PD is `cPD[12]`; lifetime PD is `cPD` at remaining term. Both come
from one object, which is why staging changes the horizon rather than the
model.

**Why competing risks.** The first implementation treated prepayment as
censoring. That estimates default probability *in the absence of prepayment*,
which overstated cumulative PD by 2.4pp at grade A rising to 15.8pp at F-60m.
Roughly 85% of LendingClub loans pay off early, and a loan settled in month 14
cannot default in month 30. Combined with a flat EAD assumption this would
double-count. Moving to competing risks reduced maximum calibration error from
15.75pp to **0.051pp**.

**Seasoning restriction.** Only vintages that could run their full contractual
term are used to fit: 36-month loans issued on or before Dec-2015, 60-month on
or before Dec-2013. Fitting cohort 673,039 loans, observed default rate 14.75%.
Without this, recent closed loans (which either prepaid immediately or defaulted
immediately) bias the hazard curve upward at short horizons.

**Calibration.** Model cumulative incidence at contractual term versus observed
lifetime default rate: **maximum absolute error 0.051%, mean 0.021%**. All
curves monotonic increasing.

**Shape.** Default hazard peaks at month 15 for grade F, 17 for C, 28 for A.
Riskier borrowers fail earlier, which is the expected seasoning pattern.

**Thin cells.** G-36m (n=517), A-60m (n=1,205) and G-60m (n=1,345) fall below
the 2,000-loan threshold and use the pooled term-level hazard shape scaled to
match the observed grade default rate.

### 3.1 Discrimination and out-of-time validation

Calibration and discrimination are different properties. A model can be
perfectly calibrated on average and still rank borrowers randomly. Both are
reported.

**Split.** Out-of-time, not random, so the economic cycle cannot leak between
train and test.

| | Definition | n | Default rate |
|---|---|---|---|
| Train | issued on or before Dec-2013 | 227,957 | 15.50% |
| Test | 36-month, issued Jan-2014 to Dec-2015 | 445,596 | 14.46% |

No feature observed after origination enters either score. `last_fico` is
excluded specifically: it is measured at payoff or charge-off, so defaulted
loans carry crashed scores by construction and including it would inflate
discrimination dishonestly.

**Results.**

| Model | Sample | AUC | Gini | KS | Distinct scores |
|---|---|---|---|---|---|
| Grade x term PD | **out-of-time** | **0.6608** | **0.3216** | **0.2522** | 7 |
| Grade x term PD | in-sample | 0.6609 | 0.3217 | 0.2362 | 14 |
| Logistic benchmark | out-of-time | 0.6678 | 0.3355 | 0.2452 | 445,570 |
| Logistic benchmark | in-sample | 0.6850 | 0.3701 | 0.2679 | 227,954 |

Benchmark: logistic regression on 12 origination-only features (FICO at
origination, rate, DTI, term, amount, log income, six purpose indicators).
Strongest standardised coefficients: interest rate +0.358, log income -0.295,
term +0.209, origination FICO -0.145.

**Two observations.**

*The model does not overfit.* Out-of-time AUC 0.6608 against in-sample 0.6609,
a gap of 0.0001. A seven-parameter model has nothing to overfit with. The
logistic benchmark degrades by 0.0172 across the same split.

*The benchmark barely beats it.* Out-of-time, twelve fitted features gain
0.0070 AUC over seven grade buckets. LendingClub's grade is itself assigned
from those origination fields, so it already carries nearly all the available
signal. This is why no scorecard was adopted: it was built, compared, and did
not earn its complexity.

**Absolute level.** AUC 0.66 is modest. It is bounded by the design: grade and
term give seven distinct scores for 36-month loans, and no fitting method
raises a seven-value score above that ceiling. For expected credit loss this is
an acceptable trade, because ECL requires the predicted default RATE to be
correct in level. Ranking power matters most for approve/decline decisions,
which this model does not make. Reporting only calibration, or only
discrimination, would misrepresent the model either way.

---

## 4. LGD

**Method.** `LGD = 1 - PV(net recovery) / EAD at default`, where EAD at default
is `funded_amnt - total_rec_prncp` and net recovery is
`recoveries - collection_recovery_fee`, floored at zero.

**Adopted value: 0.9180**, portfolio-wide, no segmentation.

**Recovery seasoning.** Recoveries are observed as at the snapshot, so recent
charge-offs have had little time to collect. Zero-recovery share falls cleanly
from 43.3% at 0-6 months of collection history to 5.3% at 48+. However mean LGD
barely moves, because recovered amounts are small even when recovery occurs.
Measured bias: **+0.0108**. Fitting restricted to charge-offs with 24+ months of
history (n=94,601).

**Discounting.** IFRS 9 requires recoveries discounted to the default date at
the original EIR. Collection timing is not in the data; an 18-month lag is
assumed.

| Assumed lag | Portfolio LGD |
|---|---|
| 0m | 0.8978 |
| 6m | 0.9051 |
| 12m | 0.9118 |
| **18m (adopted)** | **0.9180** |
| 24m | 0.9238 |
| 36m | 0.9340 |

**Segmentation rejected.** Grade-level LGD spans 0.9130 (A) to 0.9222 (G), a
1.0pp range. Term spans 0.9157 to 0.9216, 0.6pp. Neither justifies segmentation.

Note that discounting **inverts** the grade ordering: undiscounted LGD *falls*
with risk (A 0.9029 to G 0.8902) while discounted LGD *rises* (A 0.9130 to G
0.9222). Riskier grades carry higher EIRs, so IFRS 9's own rule discounts their
recoveries harder. The ordering is therefore a function of the discount rate,
not of collection behaviour, which is a further reason not to build on it.

**Consequence.** At 0.918 LGD is close to its 1.0 ceiling. Scenario stress acts
almost entirely through PD, not severity. Stage 3 provision moves only +4.4%
from base to severe.

---

## 5. EAD

Outstanding principal at the reporting date, amortised forward on the
contractual annuity schedule at the loan's own monthly rate. Discounting at
`(1 + monthly EIR)^-t`.

Prepayment is **not** applied to the EAD profile, because the competing-risks
PD already nets prepayment out of the default probability. Applying it to both
would double-count.

---

## 6. Staging and SICR

| Stage | EAD | Share | Coverage |
|---|---|---|---|
| 1 | $7,767.3m | 81.67% | 3.77% |
| 2 | $1,498.4m | 15.76% | 16.95% |
| 3 | $244.6m | 2.57% | 91.80% |

Stage 3 coverage equals LGD exactly, as it must: PD is 1 by definition.

**Stage 3 trigger.** `loan_status` in {`Default`, `Late (31-120 days)`}. IFRS 9's
rebuttable presumption is 90+ DPD. The data exposes a 31-120 band that cannot be
split at 90, so the whole band is adopted. This is conservative relative to a
strict 90+ cut. Disclosed, not concealed.

**SICR: dual test.** Either limb triggers Stage 2.

- Relative: lifetime PD at reporting date ≥ 2.0x lifetime PD at origination
- Absolute: lifetime PD up ≥ 5 percentage points
- Backstop: 30+ DPD

**Two rejected designs, recorded because the reasons matter.**

*Relative-only.* A grade G loan starts at ~40% lifetime PD; doubling requires
80%, which no grade reaches. Grade G was structurally incapable of triggering
SICR. Stage 2 was 25.3% of grade A EAD and **0.0%** of grade G EAD. No threshold
choice fixed this.

*Implied-grade deterioration.* Adding an absolute limb did not help while the
deterioration signal was routed through an inferred current grade, because G is
the worst grade that exists — a G loan's implied grade can only be G or better,
so `pd_delta ≤ 0` always. The grade scale also compresses at the bottom
(F-60m 35.6% versus G-60m 36.4%, a 0.9pp gap), leaving no room to fall.

**Adopted design.** A continuous, unbounded PD multiplier driven by FICO
movement:

```
PD_current = PD_origination * exp(-BETA * delta_FICO)
```

`BETA = 0.01006`, estimated as the size-weighted log-linear slope of default
rate on origination FICO across 18 bins, **weighted R² = 0.995**. Implied
multipliers: 40-point drop 1.50x, 80-point drop 2.24x. Bounded to [0.25, 8.0];
under 0.15% of the book binds either bound.

**Resulting profile.** Stage 2 EAD share rises monotonically from 4.1% at grade
A to 39.4% at grade G. This is *not* an artefact: mean FICO drift is **+3.7 for
grade A and −25.8 for grade G**. Weaker borrowers genuinely deteriorate more, so
a rising Stage 2 profile is the correct outcome for a test measuring change
since origination.

**Threshold sensitivity.**

| Relative | Absolute | Stage 2 |
|---|---|---|
| 1.5x | 3pp | 22.8% |
| **2.0x** | **5pp** | **16.0%** |
| 2.5x | 10pp | 8.7% |

**Data quality.** 1,343 loans (0.147%) carry a current FICO of 0, a data
artefact. Their multiplier is forced to 1.0; the DPD backstop still applies.
Uncaught, these would have appeared as catastrophic deterioration.

---

## 7. Macroeconomic scenarios

| Scenario | Weight | PD scalar | LGD add-on | Unemployment |
|---|---|---|---|---|
| Base | 50% | 1.00 | 0.000 | flat |
| Adverse | 35% | 1.35 | 0.020 | +2.0pp |
| Severe | 15% | 1.80 | 0.040 | +4.0pp |

**These are assumed, not estimated, and that was deliberate.**

The panel runs 2007-2018: one crisis at the very start when the book was tiny
and unseasoned, then a decade of falling unemployment. A regression of default
rate on unemployment over that window would fit noise and present it as a
coefficient. Scalars are instead anchored on a consumer-unsecured rule of thumb
of roughly 15-20% relative PD increase per 1pp of unemployment, stated as
judgement with sensitivity published.

The scalar is applied to the monthly default **hazard**, with survival
recomputed — not to cumulative PD, which would be an approximation.

| | Base | Adverse | Severe | Weighted |
|---|---|---|---|---|
| ECL | $771.4m | $966.7m | $1,208.6m | **$905.4m** |
| Coverage | 8.11% | 10.16% | 12.71% | **9.52%** |

Ad-hoc sensitivity: hazard ×1.5 gives $1,018.3m (+32.0%); with LGD +2pp,
$1,040.5m (+34.9%).

---

## 8. Results

| Grade | EAD | ECL (base) | Coverage |
|---|---|---|---|
| A | $2,002.7m | $45.5m | 2.27% |
| B | $2,676.0m | $156.7m | 5.86% |
| C | $2,847.3m | $274.8m | 9.65% |
| D | $1,368.7m | $183.0m | 13.37% |
| E | $455.4m | $76.7m | 16.83% |
| F | $120.2m | $25.1m | 20.85% |
| G | $40.2m | $9.7m | 24.21% |

Monotone across grades. Comparable unsecured consumer books typically carry
8-15% allowance; 9.52% sits inside that range on a model built from primitives
rather than calibrated to a target.

---

## 9. Limitations

**9.1 Vintage mismatch — the most material limitation, and now quantified.**
PD is fitted on 2007-2015 vintages (60-month: 2007-2013) and applied to a book
that is 94% 2016-2018 originations by exposure. LendingClub's underwriting
standards, borrower mix and product terms shifted materially across that gap.
This is through-the-cycle estimation applied to a point-in-time book.

The out-of-time test in section 3.1 measures the direction and size of the
resulting bias. Training on 2007-2013 and testing on 2014-2015 vintages, the
model predicts a **13.85% default rate against an actual 14.46%** — an
under-prediction of **0.61pp, or 4.4% in relative terms**. The gap is
systematic rather than noise: the model under-predicts in nine of ten risk
deciles.

**Implication for the headline.** The live book is 2016-2018, further from the
fitting window than the test set, so the drift is plausibly larger than 0.61pp.
The direction is the concern: **the $905.4m provision is more likely
understated than overstated.** A naive scaling of the measured 4.4% relative
under-prediction would put the shortfall near $40m, and the true figure could
exceed that. This cannot be resolved within this dataset, which ends at 2018Q4
with no seasoned post-2015 cohort to test against.

**9.2 Default date is proxied.** Time-to-default is `last_pymnt_d - issue_d`.
LendingClub charges off at roughly 120+ DPD, so the default month is understated
by approximately four months. This shifts the hazard curve earlier, which is
conservative for 12-month ECL and roughly neutral for lifetime.

**9.3 FICO is a proxy for creditworthiness, not a re-estimated PD.** The SICR
test scales origination PD by FICO movement rather than re-fitting a PD model at
the reporting date. FICO refresh timing is not disclosed in the data. The
elasticity is estimated from *cross-sectional* variation in origination FICO and
assumed to apply to *longitudinal* FICO movement — an assumption, not a
measurement.

**9.4 Macro overlay is judgement.** Section 7. The provision moves materially
with scalars nobody estimated.

**9.5 LGD is near saturation.** At 0.918 there is little headroom below the
ceiling, so downturn LGD effects are compressed. In a portfolio with collateral
this would be a serious understatement of severity risk.

**9.6 Recovery timing is assumed.** The 18-month lag moves portfolio LGD across
a 0.898-0.934 range (Section 4). Not observable in the data.

**9.7 Deployment artefact carries $13 of rounding.** The deployed parquet
downcasts selected float64 columns to float32, producing $905,350,357 against
$905,350,344 from the full-precision model. Relative difference 1.4 × 10⁻⁸.
Immaterial at any reporting granularity, disclosed for completeness.

**9.8 Single reporting date; partial validation only.** Out-of-time
discrimination and calibration are now tested (section 3.1), but on 2014-2015
vintages rather than on the live book itself, which is not yet seasoned. There
is still no backtest of realised versus provisioned loss, no roll-rate
analysis, and only one reporting date. A production model would require all
three.

---

## 10. Controls over the language model layer

The agent never computes. Every figure it states must trace to a tool call or to
a parameter declared in `ecl/config.py`. Both are auditable.

A numeric grounding check runs on every answer, extracting figures from prose
and matching them against values the tools returned, tolerant of display scaling
and rounding. 15 golden questions; truth values are computed by calling the
engine at evaluation time rather than hardcoded, so the evaluation cannot drift
when a parameter changes.

**Final results: 15/15 answer accuracy, 14/15 numeric grounding, 15/15 tool
selection, 108 figures checked, 1 ungrounded.**

**The check verifies figures, not mechanisms.** During development the agent
produced a fully cited answer describing exposure "migrating to Stages 2 and 3
under stress". Stage assignment in this model is computed once and is
scenario-invariant; no migration occurs. Every number in that sentence was
grounded and the causal claim was false. A numeric guardrail cannot catch that
class of error, and this is the most important limitation of the control.

Full failure taxonomy in `evals/results.md`.
