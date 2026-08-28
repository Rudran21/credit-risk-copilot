"""
schema.py

Column names and status values, VERIFIED against the real file.
Audit run 2026-08-23 on data/raw/accepted_raw.csv:
    2,260,701 rows, 151 columns, all required fields present.

Nothing here is guessed. If you change the source file, re-run the audit
and update this module before touching anything else.
"""

RAW_PATH = "data/raw/accepted_raw.csv"

LIVE_BOOK_PATH = "data/processed/live_book.parquet"
MODEL_SET_PATH = "data/processed/model_set.parquet"

# ---------------------------------------------------------------- origination
ID = "id"
ISSUE_D = "issue_d"
TERM = "term"
GRADE = "grade"
SUB_GRADE = "sub_grade"
INT_RATE = "int_rate"
FUNDED_AMNT = "funded_amnt"
INSTALLMENT = "installment"

# ---------------------------------------------------------------- performance
STATUS = "loan_status"
OUT_PRNCP = "out_prncp"
TOTAL_REC_PRNCP = "total_rec_prncp"
RECOVERIES = "recoveries"
COLLECTION_FEE = "collection_recovery_fee"
LAST_PYMNT_D = "last_pymnt_d"

# ------------------------------------------------- credit quality (SICR input)
FICO_LOW = "fico_range_low"
FICO_HIGH = "fico_range_high"
LAST_FICO_LOW = "last_fico_range_low"
LAST_FICO_HIGH = "last_fico_range_high"

# --------------------------------------------------------------- segmentation
PURPOSE = "purpose"
STATE = "addr_state"
DTI = "dti"
ANNUAL_INC = "annual_inc"
APP_TYPE = "application_type"

# Only these get read from the 1.6 GB file. Everything else is dead weight.
USECOLS = [
    ID, ISSUE_D, TERM, GRADE, SUB_GRADE, INT_RATE, FUNDED_AMNT, INSTALLMENT,
    STATUS, OUT_PRNCP, TOTAL_REC_PRNCP, RECOVERIES, COLLECTION_FEE, LAST_PYMNT_D,
    FICO_LOW, FICO_HIGH, LAST_FICO_LOW, LAST_FICO_HIGH,
    PURPOSE, STATE, DTI, ANNUAL_INC, APP_TYPE,
]

# --------------------------------------------- loan_status values (VERIFIED)
# Terminal: the loan is closed. Outcome is known. Use these to FIT the models.
ST_PAID = "Fully Paid"
ST_CHARGED_OFF = "Charged Off"
TERMINAL_STATUSES = [ST_PAID, ST_CHARGED_OFF]

# Live: the loan is open at the snapshot date. out_prncp > 0.
# This is the balance sheet. ECL is computed on THIS and nothing else.
ST_CURRENT = "Current"
ST_GRACE = "In Grace Period"
ST_LATE_16_30 = "Late (16-30 days)"
ST_LATE_31_120 = "Late (31-120 days)"
ST_DEFAULT = "Default"
LIVE_STATUSES = [ST_CURRENT, ST_GRACE, ST_LATE_16_30, ST_LATE_31_120, ST_DEFAULT]

# Days-past-due band implied by each live status. Drives the SICR backstop.
DPD_BAND = {
    ST_CURRENT: 0,
    ST_GRACE: 10,
    ST_LATE_16_30: 23,
    ST_LATE_31_120: 75,
    ST_DEFAULT: 121,
}

# ----------------------------------------------------------------- formatting
DATE_FMT = "%b-%Y"          # e.g. "Dec-2015"
SNAPSHOT_DATE = "2018-12-31"
