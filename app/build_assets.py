"""
Prepare everything the web app needs into app/data/ (run once after Steps 1-6).

  app/data/customers.parquet     cleaned TEST customers (never used in training) for the live checker
  app/data/scored.parquet        their predictions + reasons (from the saved model, same as Step 5/6)
  app/data/dashboard.json        headline numbers, risk tiers, gains curve, top drivers
  app/data/input_columns.json    the raw columns a new file must contain
  app/static/sample_customers.csv  25 raw rows (Kaggle format, Jun-Aug only) to try the file upload

Run from the project root:
    python -m app.build_assets
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.labels import GROUP_COLORS, group
from src.predict import ChurnScorer

APP = C.ROOT / "app"
OUT = APP / "data"


def gains_curve(y: np.ndarray, score: np.ndarray, steps: int = 50) -> list:
    order = np.argsort(-score)
    hits = np.cumsum(y[order])
    n, total = len(y), y.sum()
    pts = [{"contacted": 0.0, "captured": 0.0}]
    for k in np.linspace(n / steps, n, steps):
        k = int(round(k))
        pts.append({"contacted": round(k / n, 4), "captured": round(float(hits[k - 1] / total), 4)})
    return pts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(C.PROCESSED_DIR / "test.parquet")
    input_cols = [c for c in test.columns if c != C.TARGET]
    (OUT / "input_columns.json").write_text(json.dumps(input_cols))
    test.to_parquet(OUT / "customers.parquet")

    scorer = ChurnScorer(input_cols)
    clean = test[input_cols]
    scored = scorer.score(clean)
    scored.insert(0, "actual_churn", test[C.TARGET].astype(int))
    scored.to_parquet(OUT / "scored.parquet")

    ref = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv", index_col=0).loc[scored.index]
    gap = float(np.abs(ref["churn_probability"] - scored["churn_probability"]).max())
    assert gap < 1e-9, f"App scores differ from Step 5 by {gap}"

    metrics = json.loads((C.REPORTS_DIR / "test_metrics.json").read_text())
    data_summary = json.loads((C.REPORTS_DIR / "data_summary.json").read_text())
    model_summary = json.loads((C.REPORTS_DIR / "model_summary.json").read_text())
    drivers = pd.read_csv(C.REPORTS_DIR / "shap_drivers.csv").head(12)

    tiers = []
    for name in scorer.tier_names:
        part = scored[scored["risk_tier"] == name]
        tiers.append({"tier": name, "customers": int(len(part)),
                      "actual_churn_rate": float(part["actual_churn"].mean()) if len(part) else 0.0,
                      "revenue_at_risk_rs": float(part["revenue_at_risk_rs"].sum())})

    F = scorer.features.transform(clean)
    contrib = scorer.model.predict(F, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=F.columns)
    share = imp.groupby([group(c) for c in imp.index]).sum() / imp.sum()

    dashboard = {
        "data": data_summary,
        "model": {k: model_summary[k] for k in ("best_family", "imbalance_strategy", "final_cv_pr_auc", "train_cv_gap")},
        "metrics": metrics,
        "decision": scorer.decision,
        "tiers": tiers,
        "gains": gains_curve(scored["actual_churn"].to_numpy(), scored["model_score"].to_numpy()),
        "drivers": [{"label": r.label, "group": r.group, "value": round(float(r.mean_abs_shap), 4),
                     "effect": r.effect} for r in drivers.itertuples()],
        "group_share": [{"group": g, "share": round(float(v), 4), "color": GROUP_COLORS[g]}
                        for g, v in share.sort_values(ascending=False).items()],
        "group_colors": GROUP_COLORS,
        "revenue_at_risk_total": float(scored["revenue_at_risk_rs"].sum()),
        "flagged_customers": int(scored["flagged"].sum()),
        "n_customers": int(len(scored)),
        "actual_churners": int(scored["actual_churn"].sum()),
    }
    (OUT / "dashboard.json").write_text(json.dumps(dashboard, indent=1))

    # sample upload file in the ORIGINAL raw format, Jun-Aug only (September is the future)
    raw = pd.read_csv(C.RAW_PATH)
    pick = (scored.sort_values("churn_probability", ascending=False).index[:8].tolist()
            + scored[scored["risk_tier"] == "Medium"].index[:5].tolist()
            + scored[scored["risk_tier"] == "Low"].index[:12].tolist())
    sample = raw[raw[C.ID_COLUMN].isin(pick)]
    sample = sample.drop(columns=[c for c in sample.columns if c.endswith("_9") or c == "sep_vbc_3g"])
    sample.sample(frac=1, random_state=C.RANDOM_STATE).to_csv(APP / "static" / "sample_customers.csv", index=False)

    print("App assets ready - scores match Step 5 exactly")
    print(f"  customers {len(scored):,}   churners {dashboard['actual_churners']}   "
          f"flagged {dashboard['flagged_customers']}   revenue at risk Rs {dashboard['revenue_at_risk_total']:,.0f}")
    print(f"  sample upload file: {len(sample)} rows x {sample.shape[1]} columns")


if __name__ == "__main__":
    main()
