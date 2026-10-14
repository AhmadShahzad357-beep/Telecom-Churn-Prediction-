"""Central settings for the project. Change values here, not inside the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "telecom_churn_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# High-value customers = top 30% by average spend (calls + data recharge) in June-July
HIGH_VALUE_QUANTILE = 0.70
GOOD_MONTHS = (6, 7)
CHURN_MONTH = 9
ID_COLUMN = "mobile_number"
TARGET = "churn"

# Columns with a single value in the raw data (verified on the full file).
# They carry no information, so they are dropped.
CONSTANT_COLUMNS = [
    "circle_id",
    "loc_og_t2o_mou", "std_og_t2o_mou", "loc_ic_t2o_mou",
    "last_date_of_month_6", "last_date_of_month_7", "last_date_of_month_8",
    "std_og_t2c_mou_6", "std_og_t2c_mou_7", "std_og_t2c_mou_8",
    "std_ic_t2o_mou_6", "std_ic_t2o_mou_7", "std_ic_t2o_mou_8",
]

# Data-recharge block: NaN means the customer did not recharge data that month -> 0
DATA_RECHARGE_PREFIXES = [
    "total_rech_data", "max_rech_data", "count_rech_2g", "count_rech_3g",
    "av_rech_amt_data", "arpu_2g", "arpu_3g", "night_pck_user", "fb_user",
]

DATE_PREFIXES = ["date_of_last_rech", "date_of_last_rech_data"]

# ---- Step 5: business assumptions (ILLUSTRATIVE - change to the company's real numbers) ----
# Used only for the profit curve and the business-optimal threshold.
OFFER_COST_RS = 100        # cost of one retention offer / call per contacted customer
SAVE_RATE = 0.30           # share of contacted real churners who stay because of the offer
VALUE_MONTHS = 3           # months of revenue kept when a churner is saved
# Risk tiers on the calibrated churn probability
RISK_TIERS = [(0.0, 0.10, "Low"), (0.10, 0.30, "Medium"), (0.30, 0.60, "High"), (0.60, 1.01, "Very high")]