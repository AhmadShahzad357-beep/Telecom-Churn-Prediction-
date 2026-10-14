"""
Step 7b - Temporal (out-of-time) validation.

Question: does a model trained on one period still work on the NEXT period?

The main model needs 4 months (Jun-Jul baseline, Aug action, Sep churn), and the
data only has 4 months, so the main model cannot be shifted in time. This check
uses the same recipe with a 1-month baseline, which fits twice:

    Window A (train):  baseline Jun | action Jul | churn label Aug
    Window B (test):   baseline Jul | action Aug | churn label Sep

The model is trained only on window A (it never sees any September data) and is
scored once on window B. Same LightGBM settings as the final model, no re-tuning.

Run from the project root:
    python -m src.temporal
"""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src import data as D
from src import eda
from src.features import CHANGE_METRICS, NO_RATIO, NO_RECHARGE_DAYS

FIG_DIR = C.REPORTS_DIR / "figures" / "temporal"
MONTH_NAME = {6: "June", 7: "July", 8: "August", 9: "September"}
MONTH_END = {m: pd.Timestamp(d) for m, d in
             {6: "2014-06-30", 7: "2014-07-31", 8: "2014-08-31", 9: "2014-09-30"}.items()}
VBC = {6: "jun_vbc_3g", 7: "jul_vbc_3g", 8: "aug_vbc_3g", 9: "sep_vbc_3g"}
WINDOWS = {"A": {"base": 6, "action": 7, "label": 8}, "B": {"base": 7, "action": 8, "label": 9}}
CONSTANT_BASES = {"last_date_of_month", "std_og_t2c_mou", "std_ic_t2o_mou"}
DATE_BASES = {"date_of_last_rech", "date_of_last_rech_data"}
USAGE_BASE = re.compile(r"(mou|og_others|ic_others)$")
COLORS = {"cv_a": "#6250d6", "oot": "#eb6834", "cv_b": "#1baf7a"}
NAMES = {"cv_a": "In-time CV (window A)", "oot": "Out-of-time: train A -> test B",
         "cv_b": "In-time CV on B (reference)"}


def churn_flag(raw: pd.DataFrame, m: int) -> pd.Series:
    """Churn definition C for any month m: no recharge AND no usage at all in that month."""
    no_recharge = (raw[f"total_rech_amt_{m}"].fillna(0) == 0) & (raw[f"total_rech_data_{m}"].fillna(0) == 0)
    no_usage = ((raw[f"total_ic_mou_{m}"].fillna(0) == 0) & (raw[f"total_og_mou_{m}"].fillna(0) == 0)
                & (raw[f"vol_2g_mb_{m}"].fillna(0) == 0) & (raw[f"vol_3g_mb_{m}"].fillna(0) == 0))
    return (no_recharge & no_usage).astype(int)


def monthly_bases(columns) -> list:
    bases = {c[:-2] for c in columns if re.search(r"_[6-9]$", c)}
    return sorted(bases - CONSTANT_BASES - DATE_BASES)


def build_window(raw: pd.DataFrame, base: int, action: int, label: int):
    """One customer per row: features from (base, action) months only, target from the label month."""
    spend = D.monthly_spend(raw, base)
    hv = spend >= spend.quantile(C.HIGH_VALUE_QUANTILE)
    df = raw.loc[hv]
    y = churn_flag(df, label)

    cols = {}
    for b in monthly_bases(raw.columns):
        for m, tag in ((base, "prev"), (action, "now")):
            cols[f"{b}_{tag}"] = df[f"{b}_{m}"]
    X = pd.DataFrame(cols, index=df.index)
    X["vbc_3g_prev"], X["vbc_3g_now"] = df[VBC[base]], df[VBC[action]]
    X["aon"] = df["aon"]

    # missing values by meaning (same rules as Step 1)
    data_bases = tuple(C.DATA_RECHARGE_PREFIXES)
    fill0 = [c for c in X.columns if USAGE_BASE.search(c.rsplit("_", 1)[0]) or c.startswith(data_bases)]
    X[fill0] = X[fill0].fillna(0)
    left = [c for c in X.columns if X[c].isna().any()]
    if left:
        raise ValueError(f"Unexpected missing values in window {base}-{action}: {left}")

    # features (same ideas as src/features.py, with a 1-month baseline)
    for tag in ("prev", "now"):
        X[f"spend_{tag}"] = X[f"total_rech_amt_{tag}"] + X[f"total_rech_data_{tag}"] * X[f"av_rech_amt_data_{tag}"]
        X[f"data_mb_{tag}"] = X[f"vol_2g_mb_{tag}"] + X[f"vol_3g_mb_{tag}"]
    new = {}
    for b in CHANGE_METRICS:
        new[f"diff_{b}"] = X[f"{b}_now"] - X[f"{b}_prev"]
        if b not in NO_RATIO:
            new[f"ratio_{b}"] = (X[f"{b}_now"] + 1) / (X[f"{b}_prev"] + 1)
    for prefix, name in (("date_of_last_rech", "rech"), ("date_of_last_rech_data", "data_rech")):
        for m, tag in ((base, "prev"), (action, "now")):
            dates = pd.to_datetime(df[f"{prefix}_{m}"], format="%m/%d/%Y")
            new[f"days_since_last_{name}_{tag}"] = (MONTH_END[m] - dates).dt.days.fillna(NO_RECHARGE_DAYS)
        new[f"no_{name}_now"] = df[f"{prefix}_{action}"].isna().astype(int).to_numpy()
    new["used_data_now"] = (X["data_mb_now"] > 0).astype(int)
    new["roamed_now"] = ((X["roam_og_mou_now"] + X["roam_ic_mou_now"]) > 0).astype(int)
    new["monthly_pack_now"] = ((X["monthly_2g_now"] + X["monthly_3g_now"]) > 0).astype(int)
    new["sachet_pack_now"] = ((X["sachet_2g_now"] + X["sachet_3g_now"]) > 0).astype(int)
    new["no_calls_prev"] = (X["total_og_mou_prev"] == 0).astype(int)
    new["months_with_recharge"] = (X["total_rech_num_prev"] > 0).astype(int) + (X["total_rech_num_now"] > 0).astype(int)
    new["tenure_years"] = X["aon"] / 365
    X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    return X, y


def make_model(pos_weight: float) -> LGBMClassifier:
    params = json.loads((C.REPORTS_DIR / "model_summary.json").read_text())["params"]
    return LGBMClassifier(**params, subsample=0.8, subsample_freq=1, max_bin=63, scale_pos_weight=pos_weight,
                          n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1)


def oof_scores(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    oof = np.zeros(len(y))
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    for tr, va in cv.split(X, y):
        m = make_model((y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum())
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def capture(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    k = int(round(len(y) * frac))
    return float(y[np.argsort(-s)[:k]].sum() / y.sum())


def summarise(y, s) -> dict:
    y = np.asarray(y)
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "top10_capture": capture(y, s, 0.10), "top20_capture": capture(y, s, 0.20),
            "churn_rate": float(y.mean()), "n": int(len(y))}


def top_features(model, X: pd.DataFrame, k: int = 10) -> list:
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    return list(pd.Series(np.abs(contrib).mean(0), index=X.columns).sort_values(ascending=False).index[:k])


# ---------------------------------------------------------------- figures
def plot_windows():
    roles = {"Baseline": "#6250d6", "Action (features)": "#eda100", "Churn label": "#eb6834",
             "Not used": "#e7e6e1"}
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for row, (w, spec) in enumerate(reversed(list(WINDOWS.items()))):
        for i, m in enumerate((6, 7, 8, 9)):
            role = ("Baseline" if m == spec["base"] else "Action (features)" if m == spec["action"]
                    else "Churn label" if m == spec["label"] else "Not used")
            ax.barh(row, 0.94, left=i + 0.03, height=0.62, color=roles[role])
            ax.text(i + 0.5, row, MONTH_NAME[m], ha="center", va="center", fontsize=9,
                    color="#ffffff" if role != "Not used" else eda.MUTED)
    ax.set_yticks([0, 1], ["Window B\n(test only)", "Window A\n(train)"])
    ax.set_xticks([])
    ax.set_xlim(0, 4)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in roles.values()], roles.keys(),
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=9)
    ax.set_title("Temporal validation design - the model never sees September while training", pad=34)
    return fig


def plot_metrics(res: dict):
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("top10_capture", "Churners in top 10%")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    keys = ["cv_a", "oot", "cv_b"]
    for ax, (m, title) in zip(axes, metrics):
        vals = [res[k][m] for k in keys]
        bars = ax.bar(range(3), vals, color=[COLORS[k] for k in keys], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}" if m != "top10_capture" else f"{v:.0%}", ha="center", fontsize=9)
        if m == "pr_auc":
            base = ax.axhline(res["oot"]["churn_rate"], color=eda.MUTED, lw=1, ls="--")
        ax.set_xticks([])
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=11, pad=8)
    fig.legend(list(bars) + [base], [NAMES[k] for k in keys] + ["Random guess (PR-AUC)"],
               loc="lower left", bbox_to_anchor=(0.01, 0.98), ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Does the model hold up one month later?", x=0.01, ha="left", y=1.1,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pr(curves: dict, res: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLORS[k], lw=2, label=f"{NAMES[k]}  (PR-AUC {res[k]['pr_auc']:.3f})")
    ax.axhline(res["oot"]["churn_rate"], color=eda.MUTED, lw=1, ls="--",
               label=f"Random guess on B ({res['oot']['churn_rate']:.1%})")
    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of flagged who churn)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9)
    ax.set_title("Precision-recall: in-time vs out-of-time", pad=78)
    return fig


def plot_gains(curves: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fr = np.linspace(0, 1, 101)
    for k in ("oot", "cv_b"):
        y, s = curves[k]
        cap = [0.0] + [capture(np.asarray(y), s, f) for f in fr[1:]]
        ax.plot(fr * 100, np.array(cap) * 100, color=COLORS[k], lw=2, label=NAMES[k])
    ax.plot([0, 100], [0, 100], color=eda.MUTED, lw=1, ls="--", label="Random calling")
    y, s = curves["oot"]
    for f in (0.1, 0.2):
        v = capture(np.asarray(y), s, f) * 100
        ax.scatter([f * 100], [v], color=COLORS["oot"], zorder=3)
        ax.annotate(f"top {f:.0%} -> {v:.0f}% of churners", (f * 100, v), xytext=(10, -14),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Customers contacted, highest risk first (%)")
    ax.set_ylabel("Churners caught (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9, ncol=1)
    ax.set_title("Gains on September (window B)", pad=62)
    return fig


def main() -> None:
    eda.set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIG_DIR.glob("*.png"):
        f.unlink()
    raw = D.load_raw()

    # consistency check: churn_flag(9) is exactly the Step 1 label
    hv, _ = D.filter_high_value(raw)
    assert (churn_flag(hv, 9) == D.add_churn_label(hv.copy())[C.TARGET]).all(), "churn definition mismatch"

    print("7b-1  Building the two time windows...")
    XA, yA = build_window(raw, **WINDOWS["A"])
    XB, yB = build_window(raw, **WINDOWS["B"])
    assert list(XA.columns) == list(XB.columns)
    for name, X, y in (("A", XA, yA), ("B", XB, yB)):
        w = WINDOWS[name]
        print(f"    Window {name}: baseline {MONTH_NAME[w['base']]}, action {MONTH_NAME[w['action']]}, "
              f"label {MONTH_NAME[w['label']]} - {len(X):,} customers, churn {y.mean():.2%}, {X.shape[1]} features")
    both = XA.index.intersection(XB.index)
    print(f"    Customers in both windows: {len(both):,} (normal for a time split - the label month differs)")

    print("7b-2  In-time CV on window A...")
    oof_a = oof_scores(XA, yA)
    print("7b-3  Train on ALL of window A, score window B once (out-of-time)...")
    model_a = make_model((yA == 0).sum() / (yA == 1).sum()).fit(XA, yA)
    oot = model_a.predict_proba(XB)[:, 1]
    print("7b-4  Reference: in-time CV on window B (best case if we could train on September)...")
    oof_b = oof_scores(XB, yB)

    res = {"cv_a": summarise(yA, oof_a), "oot": summarise(yB, oot), "cv_b": summarise(yB, oof_b)}
    model_b = make_model((yB == 0).sum() / (yB == 1).sum()).fit(XB, yB)
    top_a, top_b = top_features(model_a, XB), top_features(model_b, XB)
    overlap = sorted(set(top_a) & set(top_b))

    table = pd.DataFrame(res).T.rename(index=NAMES)[["n", "churn_rate", "pr_auc", "roc_auc",
                                                     "top10_capture", "top20_capture"]]
    print("\n" + table.to_string(formatters={"n": "{:,.0f}".format, "churn_rate": "{:.2%}".format,
                                             "pr_auc": "{:.3f}".format, "roc_auc": "{:.3f}".format,
                                             "top10_capture": "{:.1%}".format, "top20_capture": "{:.1%}".format}))
    drop = res["cv_a"]["pr_auc"] - res["oot"]["pr_auc"]
    gap = res["cv_b"]["pr_auc"] - res["oot"]["pr_auc"]
    print(f"\n    PR-AUC change from window A (CV) to the next month: {-drop:+.3f}")
    print(f"    Out-of-time vs best case on B: {gap:.3f} PR-AUC lost by training one month earlier")
    print(f"    Top-10 drivers shared by the A-model and a B-model: {len(overlap)} of 10")

    out = {"windows": WINDOWS, "results": res, "pr_auc_change_a_to_oot": -drop, "pr_auc_gap_to_best_case_b": gap,
           "top10_drivers_model_a": top_a, "top10_drivers_model_b": top_b, "shared_top10": overlap,
           "n_customers_in_both_windows": int(len(both))}
    (C.REPORTS_DIR / "temporal_validation.json").write_text(json.dumps(out, indent=2))
    table.to_csv(C.REPORTS_DIR / "temporal_validation.csv")

    curves = {"cv_a": (yA, oof_a), "oot": (yB, oot), "cv_b": (yB, oof_b)}
    for name, fig in (("01_windows_design", plot_windows()), ("02_metrics_in_vs_out_of_time", plot_metrics(res)),
                      ("03_pr_curves", plot_pr(curves, res)), ("04_gains_september", plot_gains(curves))):
        fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"\nStep 7b complete - {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
