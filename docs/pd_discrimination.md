# PD Discrimination and Out-of-Time Validation

## Design

Out-of-time split, not random, so the economic cycle cannot leak between train and test. The PD curves are **refitted on the train cohort only**, so the test loans played no part in fitting.

| | Definition | n | Default rate |
|---|---|---|---|
| Train | issued on or before 2013-12 (36m and 60m) | 227,957 | 15.50% |
| Test | 36m issued 2014-01 to 2015-12 | 445,596 | 14.46% |

Both drawn from the seasoned cohort, so every test loan had time to run its full contractual term before the snapshot. No feature measured after origination enters either score; `last_fico` is excluded because it is observed at payoff or charge-off and would leak the outcome.

## Results

In-sample rows use 36-month train loans only, so they are comparable with the 36-month test set.

| Model | Sample | AUC | Gini | KS | Distinct scores |
|---|---|---|---|---|---|
| Grade x term PD | **out-of-time** | **0.6608** | **0.3216** | **0.2522** | 7 |
| Grade x term PD | in-sample (36m) | 0.6282 | 0.2564 | 0.1999 | 7 |
| Logistic benchmark | **out-of-time** | **0.6678** | **0.3355** | **0.2452** | 445,570 |
| Logistic benchmark | in-sample (36m) | 0.6610 | 0.3220 | 0.2357 | 172,986 |

## Calibration on the out-of-time test set, by grade

| Grade | n | Predicted | Actual | Gap (pp) |
|---|---|---|---|---|
| A | 105,451 | 5.60% | 5.41% | +0.19 |
| B | 145,185 | 10.52% | 11.46% | -0.94 |
| C | 121,465 | 15.62% | 18.63% | -3.00 |
| D | 53,221 | 20.03% | 24.66% | -4.63 |
| E | 16,509 | 22.07% | 30.35% | -8.28 |
| F | 3,341 | 25.33% | 34.75% | -9.42 |
| G | 424 | 30.53% | 41.98% | -11.45 |

Overall predicted **12.44%** versus actual **14.46%**, a gap of **-2.02pp** on loans excluded from fitting.

For reference, curves fitted on all vintages (which include the test loans) predict 13.85%, a gap of -0.61pp. That version flatters the result and is not used as evidence.

## Interpretation

The PD model is deliberately low-dimensional: grade and term only, giving seven distinct scores per term. Its discrimination ceiling is set by that design choice rather than by the fitting method. The logistic benchmark shows what a conventional scorecard extracts from the same origination fields.

For expected credit loss, calibration matters more than ranking. ECL is a provision, so the predicted default RATE has to be right in level. Ranking power matters most for approve/decline decisions, which this model does not make. Both are reported because a model that is calibrated on average can still rank randomly, and the two properties should never be conflated.
