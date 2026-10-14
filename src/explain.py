"""
Step 6 - SHAP explainability on the REAL test customers, in original units.

  6a  Global: which features drive churn risk (bar + beeswarm)
  6b  Shape of the effect for the top drivers (dependence plots)
  6c  Single customers: why THIS customer is high / low risk (waterfall)
  6d  Reason codes: top 3 risk-raising reasons for every test customer (used by the app)

SHAP values are in log-odds (the model's internal scale): positive = pushes the
customer towards churn, negative = towards staying.

Run from the project root:
    python -m src.explain
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as C
from src import eda
from src.features import build_features
from src.labels import GROUP_COLORS, describe, group, pretty

MODEL_DIR = C.ROOT / "models"
FIG_DIR = C.REPORTS_DIR / "figures" / "explain"


def plot_importance(imp: pd.Series, top: int = 15):
    top_s = imp.head(top).iloc[::-1]
    groups = [group(f) for f in top_s.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([pretty(f) for f in top_s.index], top_s.values, color=[GROUP_COLORS[g] for g in groups], height=0.62)
    for y, v in enumerate(top_s.values):
        ax.text(v + top_s.max() * 0.01, y, f"{v:.3f}", va="center", fontsize=8.5, color=eda.TEXT)
    shown = [g for g in GROUP_COLORS if g in groups]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in shown], shown,
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(shown), fontsize=9)
    ax.set_xlim(0, top_s.max() * 1.12)
    ax.set_xlabel("Mean |SHAP value| on test customers (average push on churn risk, log-odds)")
    ax.set_title(f"Top {top} churn drivers - what the model relies on most", pad=30)
    return fig


def plot_beeswarm(sv, top: int = 15):
    plt.figure(figsize=(10, 7.5))          # fresh figure: shap draws on the *current* figure
    shap.plots.beeswarm(sv, max_display=top, show=False, color_bar_label="Feature value (low -> high)",
                        plot_size=None)
    fig = plt.gcf()
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel("SHAP value (left = pushes towards staying, right = pushes towards churn)")
    ax.set_title("How each driver moves risk: red = high feature value, blue = low", loc="left",
                 fontsize=13, fontweight="bold")
    return fig


def plot_dependence(F: pd.DataFrame, sv_values: np.ndarray, y: np.ndarray, features: list):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, f in zip(axes.ravel(), features):
        j = F.columns.get_loc(f)
        x = F[f].to_numpy()
        hi = np.nanpercentile(x, 99)
        keep = x <= hi
        for cls, color, name in ((0, eda.NON_CHURN, "Stayed"), (1, eda.CHURN, "Churned")):
            sel = keep & (y == cls)
            ax.scatter(x[sel], sv_values[sel, j], s=8 if cls == 0 else 14, alpha=0.35 if cls == 0 else 0.7,
                       color=color, label=name, edgecolors="none")
        ax.axhline(0, color=eda.MUTED, lw=1)
        ax.set_xlabel(pretty(f) + ("  (top 1% trimmed)" if keep.mean() < 1 else ""))
        ax.set_ylabel("SHAP value")
        ax.set_title(pretty(f), fontsize=11, pad=24)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, markerscale=2)
    fig.suptitle("Dependence: how the top 4 drivers change risk as their value changes (test customers)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_waterfall(sv_row, title: str):
    plt.figure(figsize=(10, 6.5))          # fresh figure: shap draws on the *current* figure
    shap.plots.waterfall(sv_row, max_display=10, show=False)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.grid(False)
    fig.axes[0].set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    return fig


def reason_codes(F: pd.DataFrame, sv_values: np.ndarray, k: int = 3) -> pd.DataFrame:
    """For every customer: the k features that push risk UP the most, with their actual values."""
    rows = []
    cols = F.columns.to_numpy()
    for i in range(len(F)):
        order = np.argsort(-sv_values[i])[:k]
        row = {}
        for r, j in enumerate(order, start=1):
            if sv_values[i, j] <= 0:
                row[f"reason_{r}"] = ""
                continue
            row[f"reason_{r}"] = describe(cols[j], F.iat[i, j])
        rows.append(row)
    return pd.DataFrame(rows, index=F.index)


def main() -> None:
    eda.set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIG_DIR.glob("*.png"):
        f.unlink()
    pipe = joblib.load(MODEL_DIR / "churn_model.joblib")
    model = pipe.named_steps["model"]
    test = pd.read_parquet(C.PROCESSED_DIR / "test.parquet")
    y = test[C.TARGET].to_numpy()
    F = build_features(test.drop(columns=[C.TARGET]))

    print("6a  SHAP values for all test customers...")
    explainer = shap.TreeExplainer(model)
    sv = explainer(F)
    raw = model.predict_proba(F)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-(sv.values.sum(1) + sv.base_values))), raw, atol=1e-4), \
        "SHAP values do not add up to the model prediction"
    sv_pretty = shap.Explanation(sv.values, sv.base_values, data=F.to_numpy(),
                                 feature_names=[pretty(c) for c in F.columns])

    imp = pd.Series(np.abs(sv.values).mean(0), index=F.columns).sort_values(ascending=False)
    direction = pd.Series({c: np.corrcoef(F[c], sv.values[:, F.columns.get_loc(c)])[0, 1]
                           if F[c].std() > 0 else 0.0 for c in imp.index[:30]})
    drivers = pd.DataFrame({"feature": imp.index[:30], "label": [pretty(c) for c in imp.index[:30]],
                            "group": [group(c) for c in imp.index[:30]], "mean_abs_shap": imp.values[:30],
                            "effect": ["higher value -> more churn" if direction[c] > 0 else "lower value -> more churn"
                                       for c in imp.index[:30]]})
    drivers.to_csv(C.REPORTS_DIR / "shap_drivers.csv", index=False)
    print(drivers.head(10)[["label", "mean_abs_shap", "effect"]].to_string(index=False))
    share = imp.groupby([group(c) for c in imp.index]).sum().sort_values(ascending=False) / imp.sum()
    print("\n    Share of total SHAP importance by group:")
    for g, v in share.items():
        print(f"      {g:<20} {v:.1%}")

    print("\n6c  Example customers...")
    pred = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv").set_index("customer_id")
    pred = pred.loc[F.index]
    hi_pos = int(np.argmax(np.where(y == 1, pred["revenue_at_risk_rs"].to_numpy(), -1)))
    lo_pos = int(np.argmin(np.where(y == 0, raw, 2)))
    for pos, tag in ((hi_pos, "high-risk churner"), (lo_pos, "loyal customer")):
        print(f"    {tag}: id {F.index[pos]}  churn probability {pred['churn_probability'].iloc[pos]:.1%}  "
              f"monthly value Rs {pred['monthly_value_rs'].iloc[pos]:,.0f}")

    print("\n6d  Reason codes for every test customer...")
    reasons = reason_codes(F, sv.values)
    out = pred.join(reasons)
    out.to_csv(C.REPORTS_DIR / "test_predictions_with_reasons.csv")
    top3 = out.sort_values("revenue_at_risk_rs", ascending=False).head(3)
    print(top3[["churn_probability", "revenue_at_risk_rs", "reason_1", "reason_2", "reason_3"]].to_string())

    joblib.dump(explainer, MODEL_DIR / "shap_explainer.joblib")
    def save(name, fig):
        fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    # each figure is drawn and saved before the next one starts (shap reuses the current figure)
    save("01_shap_importance", plot_importance(imp))
    save("02_shap_beeswarm", plot_beeswarm(sv_pretty))
    save("03_shap_dependence_top4", plot_dependence(F, sv.values, y, list(imp.index[:4])))
    save("04_waterfall_high_risk_customer", plot_waterfall(
        sv_pretty[hi_pos], f"Why customer {F.index[hi_pos]} is high risk "
                           f"(churn probability {pred['churn_probability'].iloc[hi_pos]:.0%})"))
    save("05_waterfall_loyal_customer", plot_waterfall(
        sv_pretty[lo_pos], f"Why customer {F.index[lo_pos]} is low risk "
                           f"(churn probability {pred['churn_probability'].iloc[lo_pos]:.1%})"))
    figs = sorted(FIG_DIR.glob("*.png"))
    print(f"\nStep 6 complete - {len(figs)} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
