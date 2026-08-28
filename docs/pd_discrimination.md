# PD Discrimination and Out-of-Time Validation

## Design

Out-of-time split, not random, so the economic cycle cannot leak between train and test.

| | Definition | n | Default rate |
|---|---|---|---|
| Train | issued on or before 2013-12 | 227,957 | 15.50% |
| Test | 36m issued 2014-01 to 2015-12 | 445,596 | 14.46% |

Both drawn from the seasoned cohort, so every test loan had time to run its full contractual term before the snapshot. No feature measured after origination enters either score; `last_fico` is excluded because it is observed at payoff or charge-off and would leak the outcome.

## Results

| Model | Sample | AUC | Gini | KS | Distinct scores |
|---|---|---|---|---|---|
| Grade x term PD | **out-of-time** | **0.6608** | **0.3216** | **0.2522** | 7 |
| Grade x term PD | in-sample | 0.6609 | 0.3217 | 0.2362 | 14 |
| Logistic benchmark | **out-of-time** | **0.6678** | **0.3355** | **0.2452** | 445,570 |
| Logistic benchmark | in-sample | 0.6850 | 0.3701 | 0.2679 | 227,954 |

## Calibration on the out-of-time test set

| Band | n | Predicted | Actual | Lift |
|---|---|---|---|---|
| 1 | 44,560 | 5.43% | 5.20% | 0.36x |
| 2 | 44,560 | 5.43% | 5.61% | 0.39x |
| 3 | 44,559 | 9.03% | 9.72% | 0.67x |
| 4 | 44,560 | 11.11% | 11.66% | 0.81x |
| 5 | 44,559 | 11.11% | 11.39% | 0.79x |
| 6 | 44,560 | 13.62% | 14.27% | 0.99x |
| 7 | 44,559 | 17.79% | 19.27% | 1.33x |
| 8 | 44,560 | 17.79% | 18.00% | 1.24x |
| 9 | 44,559 | 21.28% | 22.94% | 1.59x |
| 10 | 44,560 | 25.93% | 26.56% | 1.84x |

Overall predicted **13.85%** versus actual **14.46%**, a gap of **-0.61pp** on loans the model never saw.

## Interpretation

The PD model is deliberately low-dimensional: grade and term only, giving seven distinct scores per term. Its discrimination ceiling is set by that design choice rather than by the fitting method. The logistic benchmark shows what a conventional scorecard extracts from the same origination fields.

For expected credit loss, calibration matters more than ranking. ECL is a provision, so the predicted default RATE has to be right in level. Ranking power matters most for approve/decline decisions, which this model does not make. Both are reported because a model that is calibrated on average can still rank randomly, and the two properties should never be conflated.
