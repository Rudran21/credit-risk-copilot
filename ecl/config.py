"""
config.py

Every model parameter and judgement in one place. If a number influences the
provision, it lives here and nowhere else. This file is the audit trail.

Each entry records: the value, where it came from, and whether it was
ESTIMATED from data or ASSUMED by judgement. A model reviewer should be able
to read this file alone and know what they are being asked to accept.
"""

# =============================================================================
# LGD
# =============================================================================
# ESTIMATED. notebooks/03_lgd.py, seasoned charge-offs (24m+ recovery window,
# n=94,601), recoveries discounted at the loan's own EIR over an assumed
# 18-month collection lag.
# Grade segmentation was tested: spread across A-G was 1.0pp (0.913 to 0.922),
# and discounting inverted the ordering because riskier grades carry higher
# EIRs. Segmentation rejected for parsimony. Term spread was 0.6pp, also
# rejected.
LGD_PORTFOLIO = 0.9180

# Sensitivity to the recovery-lag assumption (from 03_lgd.py):
#   0m -> 0.8978 | 6m -> 0.9051 | 12m -> 0.9118
#   18m -> 0.9180 (adopted) | 24m -> 0.9238 | 36m -> 0.9340
LGD_LAG_SENSITIVITY = {0: 0.8978, 6: 0.9051, 12: 0.9118,
                       18: 0.9180, 24: 0.9238, 36: 0.9340}

# Unseasoned-inclusive alternative, carried as a disclosed sensitivity.
# All 266,224 charge-offs, undiscounted: 0.9086. Bias from unseasoned
# cases measured at +0.0108, judged immaterial.
LGD_ALL_CHARGEOFFS_UNDISCOUNTED = 0.9086

# =============================================================================
# SICR
# =============================================================================
# ESTIMATED. notebooks/04_staging.py. Log-linear slope of default rate on
# origination FICO, weighted by bin size. 18 bins, weighted R2 = 0.995.
# Implied multipliers: -40pts -> 1.50x, -80pts -> 2.24x.
FICO_BETA = 0.01006

MULT_FLOOR = 0.25   # ASSUMED. Bounds on the FICO multiplier to stop
MULT_CAP = 8.0      # extreme scores producing absurd PDs. <0.15% of book binds.

# ASSUMED. Dual test, either limb triggers.
# Relative-only was tested first and rejected: it structurally exempted
# grade G, because G is the worst grade and PD could not rise. See the
# v1/v2 header notes in 04_staging.py.
SICR_PD_RATIO = 2.0    # lifetime PD doubles
SICR_PD_ABS = 0.05     # or rises 5+ percentage points

# ASSUMED, per IFRS 9 rebuttable presumption.
DPD_BACKSTOP = 30

# Stage 3 trigger. The dataset exposes a 31-120 DPD band that cannot be split
# at 90, so the whole band is adopted. Conservative relative to a strict 90+
# cut. Disclosed limitation, not a hidden choice.
STAGE3_STATUSES = ["Default", "Late (31-120 days)"]

# The DPD band the data exposes, and the presumption it cannot be split at.
# Declared here so the guardrail can allowlist them: these are documented
# model constants, not portfolio figures.
STAGE3_DPD_BAND = (31, 120)
IFRS9_DEFAULT_PRESUMPTION_DPD = 90

# Vintages used to fit PD, versus the vintages actually in the live book.
# The gap between them is the model's largest limitation.
PD_FIT_VINTAGE_RANGE = (2007, 2015)
LIVE_BOOK_VINTAGE_RANGE = (2013, 2018)

# =============================================================================
# MACROECONOMIC SCENARIOS
# =============================================================================
# ASSUMED, all of it. Deliberately not estimated.
#
# WHY NOT ESTIMATED: the LendingClub panel runs 2007-2018. It contains one
# crisis, at the very start, when the book was tiny and unseasoned, followed
# by a decade of falling unemployment. A regression of default rate on
# unemployment over that window would be fitting noise and presenting it as
# a coefficient. Stating an assumption and publishing its sensitivity is
# more defensible than a fitted parameter that cannot be defended.
#
# PD scalars are anchored on a commonly used consumer-unsecured rule of
# thumb: roughly 15-20% relative PD increase per 1pp rise in unemployment.
#   adverse: +2.0pp unemployment -> 1.35x
#   severe:  +4.0pp unemployment -> 1.80x
# The scalar is applied to the monthly DEFAULT HAZARD and survival is
# recomputed, not applied to cumulative PD directly.
#
# LGD add-ons are small because base LGD is already 0.918. There is barely
# any headroom below the 1.0 ceiling, so in this portfolio the macro overlay
# acts almost entirely through PD. That is a genuine property of unsecured
# consumer lending, not a modelling shortcut.
SCENARIOS = {
    "base": {
        "weight": 0.50,
        "pd_scalar": 1.00,
        "lgd_addon": 0.000,
        "unemployment_delta_pp": 0.0,
        "narrative": "Unemployment stable near the reporting-date level.",
    },
    "adverse": {
        "weight": 0.35,
        "pd_scalar": 1.35,
        "lgd_addon": 0.020,
        "unemployment_delta_pp": 2.0,
        "narrative": "Moderate recession, unemployment +2pp over 8 quarters.",
    },
    "severe": {
        "weight": 0.15,
        "pd_scalar": 1.80,
        "lgd_addon": 0.040,
        "unemployment_delta_pp": 4.0,
        "narrative": "Severe recession, unemployment +4pp, weak collections.",
    },
}

# ASSUMED. Probability weights. Sum must be 1.0.
assert abs(sum(s["weight"] for s in SCENARIOS.values()) - 1.0) < 1e-9

# =============================================================================
# ECL MECHANICS
# =============================================================================
STAGE1_HORIZON_M = 12      # IFRS 9: 12-month ECL for Stage 1
LGD_CAP = 1.0

# EAD profile: contractual amortising balance, annuity formula, using the
# loan's own monthly rate over its remaining term. Prepayment is NOT applied
# to EAD, because the competing-risks PD already nets prepayment out of the
# default probability. Applying it to both would double-count.
EAD_AMORTISING = True

REPORTING_DATE = "2018-12-31"
CURRENCY = "USD"
