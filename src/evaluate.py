"""
Step 5 - Honest evaluation on the untouched TEST set + calibration + business decision.

  5a  Out-of-fold (OOF) predictions on TRAIN (5-fold). Used for the CV diagnostic
      graphs, to fit the probability calibrator and to choose thresholds.
      Nothing in 5a looks at the test set.
  5b  The test set is opened ONCE: final metrics at the threshold chosen in 5a.
  5c  Calibration check, gains/lift, profit curve and revenue-at-risk tiers.

Run from the project root:
    python -m src.evaluate
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src.features import build_features

MODEL_DIR = C.ROOT / "models"
FIG_DIR = C.REPORTS_DIR / "figures" / "evaluation"


def monthly_value(df: pd.DataFrame) -> pd.Series:
    """Revenue a customer normally brings per month = June-July average ARPU (never below 0)."""
    return ((df["arpu_6"] + df["arpu_7"]) / 2).clip(lower=0)


def metrics_at(y, score, threshold) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": round(float(threshold), 3), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "flagged": int(pred.sum()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def net_value(y, score, value, threshold) -> float:
    """Campaign value: saved revenue from real churners contacted minus the cost of every offer."""
    contacted = score >= threshold
    saved = (C.SAVE_RATE * C.VALUE_MONTHS * value[contacted & (y == 1)]).sum()
    return float(saved - C.OFFER_COST_RS * contacted.sum())


def threshold_curve(y, score, value) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        m = metrics_at(y, score, t)
        rows.append({**m, "net_rs": net_value(y, score, value, t)})
    return pd.DataFrame(rows)


def profit_by_fraction(y, score, value) -> pd.DataFrame:
    order = np.argsort(-score)
    ys, vs = np.asarray(y)[order], np.asarray(value)[order]
    gain = np.cumsum(C.SAVE_RATE * C.VALUE_MONTHS * vs * ys)
    cost = C.OFFER_COST_RS * np.arange(1, len(ys) + 1)
    return pd.DataFrame({"contacted_pct": np.arange(1, len(ys) + 1) / len(ys) * 100, "net_rs": gain - cost})


def bootstrap_ci(y, score, n: int = 1000) -> dict:
    """95% interval of test PR-AUC / ROC-AUC - shows how much the score moves just from sampling."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    pr, roc = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        pr.append(average_precision_score(y[i], score[i]))
        roc.append(roc_auc_score(y[i], score[i]))
    return {"pr_auc_95ci": np.percentile(pr, [2.5, 97.5]).round(3).tolist(),
            "roc_auc_95ci": np.percentile(roc, [2.5, 97.5]).round(3).tolist()}


def main() -> None:
    from src import plots_eval as P

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIG_DIR.glob("*.png"):
        f.unlink()
    pipe = joblib.load(MODEL_DIR / "churn_model.joblib")
    train = pd.read_parquet(C.PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(C.PROCESSED_DIR / "test.parquet")
    X_tr, y_tr = train.drop(columns=[C.TARGET]), train[C.TARGET].to_numpy()
    X_te, y_te = test.drop(columns=[C.TARGET]), test[C.TARGET].to_numpy()
    F_tr = build_features(X_tr)
    v_tr, v_te = monthly_value(train).to_numpy(), monthly_value(test).to_numpy()
    base_rate = float(y_tr.mean())

    # ---------------- 5a  out-of-fold predictions on TRAIN ----------------
    print("5a  Out-of-fold predictions on train (5 folds)...")
    model = pipe.named_steps["model"]
    oof = np.zeros(len(y_tr))
    folds = []
    for tr_idx, va_idx in StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE).split(F_tr, y_tr):
        m = clone(model).fit(F_tr.iloc[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict_proba(F_tr.iloc[va_idx])[:, 1]
        folds.append((y_tr[va_idx], oof[va_idx]))

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y_tr)
    curve = threshold_curve(y_tr, oof, v_tr)
    chosen = {"f1": float(curve.loc[curve["f1"].idxmax(), "threshold"]),
              "profit": float(curve.loc[curve["net_rs"].idxmax(), "threshold"])}
    decision_threshold = chosen["profit"]
    print(f"    OOF PR-AUC {average_precision_score(y_tr, oof):.4f}   ROC-AUC {roc_auc_score(y_tr, oof):.4f}")
    print(f"    Best-F1 threshold {chosen['f1']:.2f}   Best-profit threshold {chosen['profit']:.2f} (used)")

    # ---------------- 5b  TEST set, opened once ----------------
    print("\n5b  Final evaluation on the untouched test set...")
    raw_te = pipe.predict_proba(X_te)[:, 1]
    cal_te = calibrator.predict(raw_te)
    raw_tr_in = pipe.predict_proba(X_tr)[:, 1]
    at_profit = metrics_at(y_te, raw_te, decision_threshold)
    at_f1 = metrics_at(y_te, raw_te, chosen["f1"])
    at_05 = metrics_at(y_te, raw_te, 0.5)
    summary_sets = pd.DataFrame([
        {"set": "Train (in-sample)", "pr_auc": average_precision_score(y_tr, raw_tr_in), "roc_auc": roc_auc_score(y_tr, raw_tr_in)},
        {"set": "Cross-validation", "pr_auc": average_precision_score(y_tr, oof), "roc_auc": roc_auc_score(y_tr, oof)},
        {"set": "Test (unseen)", "pr_auc": average_precision_score(y_te, raw_te), "roc_auc": roc_auc_score(y_te, raw_te)},
    ])
    for _, r in summary_sets.iterrows():
        print(f"    {r['set']:<18} PR-AUC {r.pr_auc:.4f}   ROC-AUC {r.roc_auc:.4f}")
    print(f"    Random-guess PR-AUC {y_te.mean():.4f}")
    ci = bootstrap_ci(y_te, raw_te)
    print(f"    Test 95% interval (bootstrap): PR-AUC {ci['pr_auc_95ci']}   ROC-AUC {ci['roc_auc_95ci']}  "
          f"({int(y_te.sum())} churners in test)")
    print("\n    Threshold metrics on TEST:")
    for name, m in (("default 0.50", at_05), (f"best-F1 {chosen['f1']:.2f}", at_f1),
                    (f"best-profit {decision_threshold:.2f}", at_profit)):
        print(f"    {name:<18} accuracy {m['accuracy']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}  flagged {m['flagged']}  "
              f"(TP {m['tp']}, FP {m['fp']}, FN {m['fn']})")
    brier_raw, brier_cal = brier_score_loss(y_te, raw_te), brier_score_loss(y_te, cal_te)
    print(f"\n    Calibration (Brier, lower is better): before {brier_raw:.4f}   after {brier_cal:.4f}")
    print(f"    Mean predicted probability: before {raw_te.mean():.3f}   after {cal_te.mean():.3f}   "
          f"actual churn rate {y_te.mean():.3f}")

    # ---------------- 5c  business view ----------------
    order = np.argsort(-raw_te)
    top10 = y_te[order[: len(order) // 10]].sum() / y_te.sum()
    top20 = y_te[order[: len(order) // 5]].sum() / y_te.sum()
    profit_curve = profit_by_fraction(y_te, raw_te, v_te)
    test_net = net_value(y_te, raw_te, v_te, decision_threshold)
    contact_frac = float((raw_te >= decision_threshold).mean())
    tiers_raw = pd.cut(cal_te, [t[0] for t in C.RISK_TIERS] + [C.RISK_TIERS[-1][1]],
                       labels=[t[2] for t in C.RISK_TIERS], right=False)
    tiers = (pd.DataFrame({"tier": tiers_raw, "y": y_te, "rar": cal_te * v_te})
             .groupby("tier", observed=False)
             .agg(customers=("y", "size"), actual_churn_rate=("y", "mean"), revenue_at_risk_rs=("rar", "sum"))
             .reset_index())
    tiers["actual_churn_rate"] = tiers["actual_churn_rate"].fillna(0)
    print(f"\n5c  Business view on test")
    print(f"    Top 10% riskiest customers contain {top10:.1%} of all churners; top 20% contain {top20:.1%}")
    print(f"    Campaign at the chosen threshold: contact {contact_frac:.1%} of customers, "
          f"net value Rs {test_net:,.0f}  (assumptions in src/config.py)")
    print(f"    Total monthly revenue at risk (calibrated prob x monthly ARPU): Rs {(cal_te * v_te).sum():,.0f}")
    print(tiers.to_string(index=False, formatters={"actual_churn_rate": "{:.1%}".format,
                                                   "revenue_at_risk_rs": "Rs {:,.0f}".format}))

    # ---------------- save everything ----------------
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")
    decision = {"threshold": decision_threshold, "threshold_best_f1": chosen["f1"], "chosen_on": "train out-of-fold",
                "offer_cost_rs": C.OFFER_COST_RS, "save_rate": C.SAVE_RATE, "value_months": C.VALUE_MONTHS,
                "risk_tiers": C.RISK_TIERS}
    (MODEL_DIR / "decision.json").write_text(json.dumps(decision, indent=2))
    test_metrics = {
        "pr_auc": average_precision_score(y_te, raw_te), "roc_auc": roc_auc_score(y_te, raw_te),
        "random_pr_auc": float(y_te.mean()), **ci, "brier_before": brier_raw, "brier_after": brier_cal,
        "at_default_0_5": at_05, "at_best_f1": at_f1, "at_decision_threshold": at_profit,
        "top10_capture": top10, "top20_capture": top20, "campaign_net_rs": test_net,
        "contacted_fraction": contact_frac, "n_test": int(len(y_te)),
    }
    (C.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, default=float))
    pd.DataFrame({"customer_id": X_te.index, "actual_churn": y_te, "model_score": raw_te,
                  "churn_probability": cal_te, "monthly_value_rs": v_te,
                  "revenue_at_risk_rs": cal_te * v_te, "risk_tier": tiers_raw.astype(str),
                  "flagged": (raw_te >= decision_threshold).astype(int)}
                 ).sort_values("revenue_at_risk_rs", ascending=False).to_csv(
        C.REPORTS_DIR / "test_predictions.csv", index=False)
    summary_sets.to_csv(C.REPORTS_DIR / "train_cv_test_scores.csv", index=False)

    assumptions = (f"Assumptions: offer cost Rs {C.OFFER_COST_RS}, {C.SAVE_RATE:.0%} of contacted churners saved, "
                   f"{C.VALUE_MONTHS} months of ARPU kept")
    figs = [
        ("01_cv_roc_curves", P.plot_cv_roc(folds)),
        ("02_cv_pr_curves", P.plot_cv_pr(folds, base_rate)),
        ("03_cv_score_distribution", P.plot_score_distribution(y_tr, oof, "Model scores on train (out-of-fold): churners sit on the right")),
        ("04_cv_threshold_tradeoff", P.plot_threshold_tradeoff(curve, chosen)),
        ("05_test_roc_pr", P.plot_test_curves(y_te, raw_te, float(y_te.mean()))),
        ("06_test_confusion_matrix", P.plot_confusion(confusion_matrix(y_te, (raw_te >= decision_threshold).astype(int)), decision_threshold)),
        ("07_test_calibration", P.plot_calibration(y_te, raw_te, cal_te)),
        ("08_test_gains_lift", P.plot_gains_lift(y_te, raw_te)),
        ("09_test_profit_curve", P.plot_profit(profit_curve, contact_frac, assumptions)),
        ("10_test_risk_tiers", P.plot_risk_tiers(tiers)),
        ("11_train_cv_test", P.plot_train_cv_test(summary_sets)),
    ]
    for name, fig in figs:
        fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    print(f"\nStep 5 complete - {len(figs)} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()