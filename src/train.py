"""
Step 4 - Model comparison, imbalance handling, tuning and overfitting checks.

Everything here uses the TRAINING set only. The test set stays closed until
Step 5, so the final test score is a genuinely unseen estimate.

  4a  Compare model families with 5-fold CV (same folds for every model)
  4b  Compare imbalance strategies for the best family - SMOTE is applied
      INSIDE each CV fold (imblearn Pipeline), never before the split
  4c  Tune the best set-up with 3-fold CV (RandomizedSearch). The unbiased
      check of the tuned model is the untouched test set in Step 5.
      Optional --nested runs nested CV as well (slow: 120+ model fits).
  4d  Overfitting check: train vs CV scores and a learning curve
  4e  Fit the final pipeline on all training data and save it

Run from the project root:
    python -m src.train                       # full run
    python -m src.train --reuse-comparison    # skip 4a/4b, reuse saved results
    python -m src.train --nested              # also run nested CV (slow)
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import config as C
from src.features import build_features

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = C.ROOT / "models"
FIG_DIR = C.REPORTS_DIR / "figures" / "model"
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
RS = C.RANDOM_STATE


def signed_log(X):
    """log transform that keeps the sign (arpu can be negative) - used only for the linear model."""
    return np.sign(X) * np.log1p(np.abs(X))


def features_step():
    return FunctionTransformer(build_features, validate=False)


def make_pipeline(name: str, pos_weight: float, imbalance: str = "weights", with_features: bool = False):
    """(features) -> (optional SMOTE) -> model. CV runs on pre-built features for speed
    (features are row-wise, so building them once cannot leak); the saved model includes them."""
    w = pos_weight if imbalance == "weights" else 1.0
    if name == "Logistic Regression":
        model = LogisticRegression(C=0.1, class_weight="balanced" if imbalance == "weights" else None,
                                   max_iter=3000)
        steps = [("log", FunctionTransformer(signed_log)), ("scale", StandardScaler()), ("model", model)]
    elif name == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                       class_weight="balanced_subsample" if imbalance == "weights" else None,
                                       n_jobs=-1, random_state=RS)
        steps = [("model", model)]
    elif name == "XGBoost":
        model = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=5, scale_pos_weight=w,
                              eval_metric="aucpr", n_jobs=-1, random_state=RS)
        steps = [("model", model)]
    elif name == "LightGBM":
        # max_bin=63: ~2x faster than the default 255 with the same CV score (checked)
        model = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, max_bin=63,
                               scale_pos_weight=w, n_jobs=-1, random_state=RS, verbose=-1)
        steps = [("model", model)]
    else:
        raise ValueError(name)
    if with_features:
        steps.insert(0, ("features", features_step()))
    if imbalance == "smote":
        steps.insert(-1, ("smote", SMOTE(random_state=RS)))
        return ImbPipeline(steps)
    return Pipeline(steps)


def cv_scores(pipe, X, y, cv) -> dict:
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=1)
    out = {}
    for k in SCORING:
        out[f"cv_{k}"] = r[f"test_{k}"].mean()
        out[f"cv_{k}_std"] = r[f"test_{k}"].std()
        out[f"train_{k}"] = r[f"train_{k}"].mean()
    out["fit_seconds"] = r["fit_time"].mean()
    return out


def param_space(name: str) -> dict:
    if name == "LightGBM":
        # small trees + large leaves + L2 = regularisation against the train/CV gap seen in 4a
        return {"model__num_leaves": [7, 15, 31], "model__learning_rate": [0.02, 0.03, 0.05],
                "model__n_estimators": [300, 500, 800], "model__min_child_samples": [50, 100, 200],
                "model__colsample_bytree": [0.5, 0.7], "model__reg_lambda": [0, 5, 10]}
    if name == "XGBoost":
        return {"model__max_depth": [3, 4, 5, 6], "model__learning_rate": [0.02, 0.03, 0.05],
                "model__n_estimators": [300, 500, 800], "model__min_child_weight": [1, 5, 10],
                "model__colsample_bytree": [0.5, 0.7, 0.9], "model__reg_lambda": [1, 5, 10]}
    if name == "Random Forest":
        return {"model__n_estimators": [300, 500], "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.2, 0.4], "model__max_depth": [None, 12, 20]}
    return {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]}


def main(n_iter: int = 8, reuse_comparison: bool = False, nested: bool = False) -> None:
    t0 = time.time()
    train = pd.read_parquet(C.PROCESSED_DIR / "train.parquet")
    X, y = train.drop(columns=[C.TARGET]), train[C.TARGET]
    F = build_features(X)                      # row-wise features, built once
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    inner = StratifiedKFold(3, shuffle=True, random_state=RS)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comparison_csv = C.REPORTS_DIR / "model_comparison_cv.csv"

    if reuse_comparison and comparison_csv.exists():
        saved = pd.read_csv(comparison_csv)
        families, strategies = saved.iloc[:4].reset_index(drop=True), saved.iloc[4:].reset_index(drop=True)
        print("4a/4b  Reusing saved comparison from reports/model_comparison_cv.csv")
        for _, r in families.iterrows():
            print(f"    {r.model:<20} PR-AUC {r.cv_pr_auc:.4f} +/- {r.cv_pr_auc_std:.4f}")
        for _, r in strategies.iterrows():
            print(f"    {r.imbalance:<20} PR-AUC {r.cv_pr_auc:.4f} +/- {r.cv_pr_auc_std:.4f}")
    else:
        # 4a - model families (class weights for all, same folds)
        print("4a  Comparing model families (5-fold CV, class weights)...")
        rows = []
        for name in ("Logistic Regression", "Random Forest", "XGBoost", "LightGBM"):
            s = cv_scores(make_pipeline(name, pos_weight), F, y, outer)
            rows.append({"model": name, "imbalance": "class weights", **s})
            print(f"    {name:<20} PR-AUC {s['cv_pr_auc']:.4f} +/- {s['cv_pr_auc_std']:.4f}   "
                  f"ROC-AUC {s['cv_roc_auc']:.4f}   train PR-AUC {s['train_pr_auc']:.4f}")
        families = pd.DataFrame(rows)

        # 4b - imbalance strategy for the best family (SMOTE only inside folds)
        best = families.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
        print(f"\n4b  Imbalance strategies for {best} (5-fold CV)...")
        rows = []
        for strategy, label in (("none", "no handling"), ("weights", "class weights"),
                                ("smote", "SMOTE inside CV")):
            s = cv_scores(make_pipeline(best, pos_weight, strategy), F, y, outer)
            rows.append({"model": best, "imbalance": label, **s})
            print(f"    {label:<16} PR-AUC {s['cv_pr_auc']:.4f} +/- {s['cv_pr_auc_std']:.4f}   "
                  f"ROC-AUC {s['cv_roc_auc']:.4f}")
        strategies = pd.DataFrame(rows)
        pd.concat([families, strategies]).to_csv(comparison_csv, index=False)

    best_family = families.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
    best_strategy = strategies.sort_values("cv_pr_auc", ascending=False).iloc[0]["imbalance"]
    strategy_key = {"no handling": "none", "class weights": "weights", "SMOTE inside CV": "smote"}[best_strategy]
    print(f"    -> best: {best_family} + {best_strategy}")

    # 4c - tuning with 3-fold CV (test set in Step 5 gives the unbiased check)
    print(f"\n4c  Tuning {best_family} ({n_iter} settings x 3-fold CV)...")
    base = make_pipeline(best_family, pos_weight, strategy_key)
    search = RandomizedSearchCV(base, param_space(best_family), n_iter=n_iter, cv=inner,
                                scoring="average_precision", random_state=RS, n_jobs=1)
    search.fit(F, y)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    tuned = search.best_estimator_
    print(f"    Best 3-fold CV PR-AUC {search.best_score_:.4f}")
    print(f"    Final parameters: {best_params}")

    nested_pr = nested_roc = None
    if nested:
        print("    Nested CV (outer 5 x inner 3)...")
        res = cross_validate(search, F, y, cv=outer, scoring=SCORING, n_jobs=1)
        nested_pr, nested_roc = res["test_pr_auc"], res["test_roc_auc"]
        print(f"    Nested CV PR-AUC {nested_pr.mean():.4f} +/- {nested_pr.std():.4f}   "
              f"ROC-AUC {nested_roc.mean():.4f}")

    # 4d - overfitting check on the final settings
    print("\n4d  Overfitting check (train vs 5-fold CV, final settings)...")
    fit_check = cv_scores(tuned, F, y, outer)
    gap = fit_check["train_pr_auc"] - fit_check["cv_pr_auc"]
    print(f"    Train PR-AUC {fit_check['train_pr_auc']:.4f}   CV PR-AUC {fit_check['cv_pr_auc']:.4f}   gap {gap:.4f}")
    sizes, tr, va = learning_curve(tuned, F, y, cv=inner, scoring="average_precision",
                                   train_sizes=np.linspace(0.25, 1.0, 4), n_jobs=1, shuffle=True,
                                   random_state=RS)
    curve = pd.DataFrame({"train_size": sizes, "train_pr_auc": tr.mean(1), "train_std": tr.std(1),
                          "cv_pr_auc": va.mean(1), "cv_std": va.std(1)})
    curve.to_csv(C.REPORTS_DIR / "learning_curve.csv", index=False)

    # 4e - save the full pipeline (features + model) for Step 5 and the app
    final = make_pipeline(best_family, pos_weight, strategy_key, with_features=True)
    final.set_params(**search.best_params_)
    final.fit(X, y)
    assert np.allclose(final.predict_proba(X.head(200))[:, 1], tuned.predict_proba(F.head(200))[:, 1]), \
        "saved pipeline differs from the tuned model"
    joblib.dump(final, MODEL_DIR / "churn_model.joblib")
    summary = {
        "best_family": best_family, "imbalance_strategy": best_strategy, "params": best_params,
        "tuning_cv3_pr_auc": round(float(search.best_score_), 4),
        "nested_cv_pr_auc": None if nested_pr is None else round(float(nested_pr.mean()), 4),
        "nested_cv_roc_auc": None if nested_roc is None else round(float(nested_roc.mean()), 4),
        "final_train_pr_auc": round(fit_check["train_pr_auc"], 4), "final_cv_pr_auc": round(fit_check["cv_pr_auc"], 4),
        "train_cv_gap": round(gap, 4), "baseline_pr_auc_random": round(float(y.mean()), 4),
        "n_train": int(len(y)), "minutes": round((time.time() - t0) / 60, 1),
    }
    (C.REPORTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    from src.plots_model import plot_family_comparison, plot_strategies, plot_learning_curve
    plot_family_comparison(families).savefig(FIG_DIR / "01_model_families.png", bbox_inches="tight")
    plot_strategies(strategies).savefig(FIG_DIR / "02_imbalance_strategies.png", bbox_inches="tight")
    plot_learning_curve(curve).savefig(FIG_DIR / "03_learning_curve.png", bbox_inches="tight")

    print(f"\nStep 4 complete in {summary['minutes']} min")
    print(f"  model saved   {MODEL_DIR / 'churn_model.joblib'}")
    print(f"  figures       {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-comparison", action="store_true", help="skip 4a/4b, reuse saved CSV")
    ap.add_argument("--nested", action="store_true", help="also run nested CV (slow)")
    ap.add_argument("--n-iter", type=int, default=8)
    a = ap.parse_args()
    main(n_iter=a.n_iter, reuse_comparison=a.reuse_comparison, nested=a.nested)