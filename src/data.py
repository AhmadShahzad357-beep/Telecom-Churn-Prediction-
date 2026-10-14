"""
Step 1 - Raw data -> churn label -> high-value filter -> rule-based cleaning -> train/test split.

Only rules that look at one customer's own row are applied here.
Anything that is *learned* from the data (medians, limits, encodings) happens
later, on the training set only, so the test set stays truly unseen.

Run from the project root:
    python -m src.data
"""
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C

USAGE_RE = re.compile(r"(mou|og_others|ic_others)_[678]$")


def load_raw(path=C.RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df[C.ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer ids found in raw data")
    return df.set_index(C.ID_COLUMN)


def monthly_spend(df: pd.DataFrame, m: int) -> pd.Series:
    """Call recharge + data recharge for month m (total_rech_amt does not include data)."""
    data_amt = df[f"total_rech_data_{m}"].fillna(0) * df[f"av_rech_amt_data_{m}"].fillna(0)
    return df[f"total_rech_amt_{m}"] + data_amt


def filter_high_value(df: pd.DataFrame):
    avg_spend = sum(monthly_spend(df, m) for m in C.GOOD_MONTHS) / len(C.GOOD_MONTHS)
    threshold = float(avg_spend.quantile(C.HIGH_VALUE_QUANTILE))
    return df.loc[avg_spend >= threshold].copy(), threshold


def add_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """Churn = in month 9 the customer neither recharged nor used the service at all."""
    m = C.CHURN_MONTH
    no_recharge = (df[f"total_rech_amt_{m}"].fillna(0) == 0) & (df[f"total_rech_data_{m}"].fillna(0) == 0)
    no_usage = (
        (df[f"total_ic_mou_{m}"].fillna(0) == 0)
        & (df[f"total_og_mou_{m}"].fillna(0) == 0)
        & (df[f"vol_2g_mb_{m}"].fillna(0) == 0)
        & (df[f"vol_3g_mb_{m}"].fillna(0) == 0)
    )
    df[C.TARGET] = (no_recharge & no_usage).astype(int)
    return df


def drop_unusable_columns(df: pd.DataFrame):
    month9 = [c for c in df.columns if c.endswith("_9")] + ["sep_vbc_3g"]
    constant = [c for c in C.CONSTANT_COLUMNS if c in df.columns]
    return df.drop(columns=month9 + constant), month9, constant


def fill_missing_by_meaning(df: pd.DataFrame):
    """
    NaN in this data is not random - it means 'nothing happened':
      * usage block (mou columns) is NaN together when total usage that month is 0
      * data-recharge block is NaN together when no data recharge was made
      * recharge dates are NaN when there was no recharge that month (kept as NaT)
    Any NaN outside these known groups stops the run instead of being guessed.
    """
    usage_cols = [c for c in df.columns if USAGE_RE.search(c)]
    data_cols = [c for c in df.columns
                 if any(re.fullmatch(rf"{p}_[678]", c) for p in C.DATA_RECHARGE_PREFIXES)]
    date_cols = [c for c in df.columns
                 if any(re.fullmatch(rf"{p}_[678]", c) for p in C.DATE_PREFIXES)]

    df[usage_cols] = df[usage_cols].fillna(0)
    df[data_cols] = df[data_cols].fillna(0)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y")

    unexpected = [c for c in df.columns if c not in date_cols and df[c].isna().any()]
    if unexpected:
        raise ValueError(f"Unexpected missing values in: {unexpected}")
    return df, {"usage_filled_with_0": len(usage_cols),
                "data_recharge_filled_with_0": len(data_cols),
                "date_columns_kept_as_NaT": len(date_cols)}


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE,
                            stratify=df[C.TARGET])


def run_checks(train: pd.DataFrame, test: pd.DataFrame, n_hv: int) -> None:
    cols = train.columns
    assert not any(c.endswith("_9") for c in cols), "month-9 column left"
    assert "sep_vbc_3g" not in cols, "sep_vbc_3g left"
    assert C.ID_COLUMN not in cols, "customer id is a column"
    assert train.index.intersection(test.index).empty, "train/test overlap"
    assert len(train) + len(test) == n_hv, "rows lost in split"
    assert abs(train[C.TARGET].mean() - test[C.TARGET].mean()) < 0.005, "churn rate differs"
    constant_now = [c for c in cols if train[c].nunique(dropna=False) <= 1]
    assert not constant_now, f"constant columns in train: {constant_now}"


def main() -> None:
    raw = load_raw()
    hv, threshold = filter_high_value(raw)
    hv = add_churn_label(hv)
    hv, month9, constant = drop_unusable_columns(hv)
    hv, fill_info = fill_missing_by_meaning(hv)
    train, test = split(hv)
    run_checks(train, test, len(hv))

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(C.PROCESSED_DIR / "train.parquet")
    test.to_parquet(C.PROCESSED_DIR / "test.parquet")

    summary = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1] + 1),
        "high_value_threshold_avg_spend_jun_jul": round(threshold, 2),
        "high_value_customers": int(len(hv)),
        "churners": int(hv[C.TARGET].sum()),
        "churn_rate": round(float(hv[C.TARGET].mean()), 4),
        "dropped_month9_columns": len(month9),
        "dropped_constant_columns": len(constant),
        "missing_value_handling": fill_info,
        "features_after_cleaning": int(hv.shape[1] - 1),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_churn_rate": round(float(train[C.TARGET].mean()), 4),
        "test_churn_rate": round(float(test[C.TARGET].mean()), 4),
    }
    (C.REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))

    print("Step 1 complete - all checks passed")
    for k, v in summary.items():
        print(f"  {k:<40} {v}")


if __name__ == "__main__":
    main()