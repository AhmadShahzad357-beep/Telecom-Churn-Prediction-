"""
Step 3 - Feature engineering.

Every feature here is computed from ONE customer's own row (their June-July
baseline vs their August behaviour). Nothing is learned from other customers,
so the same function can be applied to train, test and new customers in the
app without any leakage. It is used inside the model pipeline in Step 4.

Run from the project root to check the features on the training set:
    python -m src.features
"""
import numpy as np
import pandas as pd

from src import config as C

MONTH_END = {6: pd.Timestamp("2014-06-30"), 7: pd.Timestamp("2014-07-31"), 8: pd.Timestamp("2014-08-31")}
NO_RECHARGE_DAYS = 32          # "no recharge in the month" = longer ago than any day in the month

# Metrics compared August vs the June-July baseline (EDA section 2-3)
CHANGE_METRICS = [
    "spend", "arpu", "total_rech_amt", "total_rech_num", "max_rech_amt", "last_day_rch_amt",
    "total_og_mou", "total_ic_mou", "onnet_mou", "offnet_mou", "loc_og_mou", "loc_ic_mou",
    "std_og_mou", "std_ic_mou", "data_mb", "total_rech_data", "av_rech_amt_data",
]
# arpu can be negative (refunds), so a ratio is not meaningful for it - difference only
NO_RATIO = {"arpu"}


def _add_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    for m in (6, 7, 8):
        df[f"spend_{m}"] = df[f"total_rech_amt_{m}"] + df[f"total_rech_data_{m}"] * df[f"av_rech_amt_data_{m}"]
        df[f"data_mb_{m}"] = df[f"vol_2g_mb_{m}"] + df[f"vol_3g_mb_{m}"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw cleaned columns in -> model-ready numeric columns out (target is not touched)."""
    df = _add_monthly_totals(df.copy())
    new = {}

    # 1. August vs own June-July baseline (the 'action phase' collapse)
    for base in CHANGE_METRICS:
        good = (df[f"{base}_6"] + df[f"{base}_7"]) / 2
        new[f"good_{base}"] = good
        new[f"diff_{base}"] = df[f"{base}_8"] - good
        if base not in NO_RATIO:
            # +1 smoothing keeps the ratio finite when the baseline is 0
            new[f"ratio_{base}"] = (df[f"{base}_8"] + 1) / (good + 1)

    # 2. Recharge recency (dates -> days before month end; no recharge -> 32)
    for prefix, name in (("date_of_last_rech", "rech"), ("date_of_last_rech_data", "data_rech")):
        for m in (6, 7, 8):
            days = (MONTH_END[m] - df[f"{prefix}_{m}"]).dt.days
            new[f"days_since_last_{name}_{m}"] = days.fillna(NO_RECHARGE_DAYS)
        new[f"no_{name}_8"] = df[f"{prefix}_8"].isna().astype(int)

    # 3. Behaviour flags seen in the EDA
    new["no_calls_good_phase"] = ((df["total_og_mou_6"] + df["total_og_mou_7"]) == 0).astype(int)
    new["used_data_8"] = (df["data_mb_8"] > 0).astype(int)
    new["roamed_8"] = ((df["roam_og_mou_8"] + df["roam_ic_mou_8"]) > 0).astype(int)
    new["monthly_pack_8"] = ((df["monthly_2g_8"] + df["monthly_3g_8"]) > 0).astype(int)
    new["sachet_pack_8"] = ((df["sachet_2g_8"] + df["sachet_3g_8"]) > 0).astype(int)
    new["months_with_recharge"] = sum((df[f"total_rech_num_{m}"] > 0).astype(int) for m in (6, 7, 8))
    new["tenure_years"] = df["aon"] / 365

    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    date_cols = [c for c in out.columns if c.startswith("date_of_last_rech")]
    return out.drop(columns=date_cols)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != C.TARGET]


def check_features(feats: pd.DataFrame) -> None:
    X = feats.drop(columns=[C.TARGET], errors="ignore")
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert not non_numeric, f"non-numeric columns: {non_numeric}"
    assert not X.isna().any().any(), f"NaN in: {X.columns[X.isna().any()].tolist()}"
    assert np.isfinite(X.to_numpy()).all(), "infinite values found"
    assert C.ID_COLUMN not in X.columns, "customer id leaked into features"


def ablation(X_raw: pd.DataFrame, X_all: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Quick 5-fold CV check (train only): do the engineered features help a model?
    Fixed, untuned LightGBM - this is evidence for the features, not the final model."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    model = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, random_state=C.RANDOM_STATE,
                           verbose=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for name, X in (("Raw columns only", X_raw), ("Raw + engineered", X_all)):
        r = cross_validate(model, X, y, cv=cv, scoring=["average_precision", "roc_auc"])
        rows.append({"feature_set": name, "n_features": X.shape[1],
                     "pr_auc_mean": r["test_average_precision"].mean(),
                     "pr_auc_std": r["test_average_precision"].std(),
                     "roc_auc_mean": r["test_roc_auc"].mean(),
                     "roc_auc_std": r["test_roc_auc"].std()})
    return pd.DataFrame(rows)


def plot_feature_strength(strength: pd.DataFrame, top: int = 15):
    import matplotlib.pyplot as plt
    from src import eda

    top_df = strength.head(top).iloc[::-1]
    colors = {"raw": eda.SINGLE["violet"], "engineered": eda.SINGLE["aqua"]}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top_df["feature"], top_df["strength"], color=[colors[t] for t in top_df["type"]], height=0.62)
    for y, v in enumerate(top_df["strength"]):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=8.5, color=eda.TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("engineered", "raw")]
    ax.legend(handles, ["Engineered feature", "Raw column"], loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    ax.set_xlim(0, top_df["strength"].max() * 1.12)
    ax.set_xlabel("Separation strength = |AUC - 0.5| x 2   (train set)")
    ax.set_title(f"Top {top} single features: an engineered ratio ranks first", pad=30)
    return fig


def plot_ablation(res: pd.DataFrame):
    import matplotlib.pyplot as plt
    from src import eda

    colors = [eda.SINGLE["violet"], eda.SINGLE["aqua"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in ((axes[0], "pr_auc", "PR-AUC (main metric)"), (axes[1], "roc_auc", "ROC-AUC")):
        x = range(len(res))
        bars = ax.bar(x, res[f"{metric}_mean"], yerr=res[f"{metric}_std"], color=colors, width=0.55,
                      capsize=6, error_kw=dict(ecolor=eda.MUTED, lw=1))
        for i, (m, s) in enumerate(zip(res[f"{metric}_mean"], res[f"{metric}_std"])):
            ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=10, color=eda.TEXT)
        ax.set_xticks(list(x), [f"{n}\n({k} features)" for n, k in zip(res["feature_set"], res["n_features"])])
        ax.set_ylim(0, 1.08)
        ax.set_title(label, fontsize=11)
        ax.legend(bars, list(res["feature_set"]), loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, fontsize=9)
    fig.suptitle("Engineered features improve the model (5-fold CV on train, error bar = 1 std)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    from src.eda import univariate_strength, set_style

    train = pd.read_parquet(C.PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(C.PROCESSED_DIR / "test.parquet")
    f_train, f_test = build_features(train), build_features(test)
    check_features(f_train)
    check_features(f_test)
    assert list(f_train.columns) == list(f_test.columns), "train/test columns differ"

    raw_cols = set(train.columns)
    strength = univariate_strength(f_train)
    strength["type"] = np.where(strength["feature"].isin(raw_cols), "raw", "engineered")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    strength.to_csv(C.REPORTS_DIR / "feature_strength_train.csv", index=False)

    n_raw = len([c for c in feature_columns(f_train) if c in raw_cols])
    n_new = len(feature_columns(f_train)) - n_raw
    print("Step 3 complete - all feature checks passed")
    print(f"  raw features kept        {n_raw}")
    print(f"  engineered features      {n_new}")
    print(f"  total model features     {n_raw + n_new}")
    print(f"  train / test shape       {f_train.shape} / {f_test.shape}")
    print("\nTop 15 features by separation strength (train only):")
    print(strength.head(15)[["feature", "type", "strength", "direction"]].to_string(index=False))
    best_raw = strength.loc[strength["type"] == "raw", "strength"].max()
    best_new = strength.loc[strength["type"] == "engineered", "strength"].max()
    print(f"\nBest raw feature strength        {best_raw:.3f}")
    print(f"Best engineered feature strength {best_new:.3f}")

    X_all = f_train.drop(columns=[C.TARGET])
    X_raw = X_all[[c for c in X_all.columns if c in raw_cols]]
    res = ablation(X_raw, X_all, f_train[C.TARGET])
    res.to_csv(C.REPORTS_DIR / "feature_ablation_cv.csv", index=False)
    print("\nAblation - 5-fold CV on train (untuned LightGBM):")
    for _, r in res.iterrows():
        print(f"  {r.feature_set:<18} {r.n_features:>3} features   "
              f"PR-AUC {r.pr_auc_mean:.4f} +/- {r.pr_auc_std:.4f}   "
              f"ROC-AUC {r.roc_auc_mean:.4f} +/- {r.roc_auc_std:.4f}")

    set_style()
    fig_dir = C.REPORTS_DIR / "figures" / "features"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_strength(strength).savefig(fig_dir / "01_top_feature_strength.png", bbox_inches="tight")
    plot_ablation(res).savefig(fig_dir / "02_feature_ablation.png", bbox_inches="tight")
    print(f"\nFigures saved to {fig_dir}")


if __name__ == "__main__":
    main()