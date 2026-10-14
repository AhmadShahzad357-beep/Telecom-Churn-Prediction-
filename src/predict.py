"""
Step 7 - Scoring service: raw customer rows in -> churn probability, risk tier,
revenue at risk and plain-English reasons out. Used by the web app (app/main.py).

Same rules as training, nothing new is learned here:
  * cleaning     = src.data (drop month-9 / constant columns, fill NaN by meaning)
  * features     = the saved pipeline (src.features.build_features inside it)
  * probability  = saved isotonic calibrator on top of the model score
  * flag / tiers = models/decision.json (threshold chosen on train out-of-fold)
  * reasons      = exact SHAP values from LightGBM itself (pred_contrib=True),
                   identical to shap.TreeExplainer, so the app does not need the shap package.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src.labels import describe, group, pretty

MODEL_DIR = C.ROOT / "models"
AUG_END = pd.Timestamp("2014-08-31")
DATA_BLOCK_8 = ["total_rech_data_8", "max_rech_data_8", "count_rech_2g_8", "count_rech_3g_8", "av_rech_amt_data_8",
                "arpu_2g_8", "arpu_3g_8", "night_pck_user_8", "fb_user_8",
                "monthly_2g_8", "sachet_2g_8", "monthly_3g_8", "sachet_3g_8"]
RECHARGE_AMOUNTS_8 = ["total_rech_amt_8", "max_rech_amt_8", "last_day_rch_amt_8"]


def monthly_value(df: pd.DataFrame) -> pd.Series:
    """Same definition as Step 5: June-July average ARPU, never below 0."""
    return ((df["arpu_6"] + df["arpu_7"]) / 2).clip(lower=0)


def avg_spend_good_phase(df: pd.DataFrame) -> pd.Series:
    return sum(D.monthly_spend(df, m) for m in C.GOOD_MONTHS) / len(C.GOOD_MONTHS)


def _usage_cols_8(columns, incoming: bool) -> list:
    cols = [c for c in columns if D.USAGE_RE.search(c) and c.endswith("_8")]
    is_in = [("_ic_" in c) or c.startswith("ic_") for c in cols]
    return [c for c, i in zip(cols, is_in) if i == incoming]


class ChurnScorer:
    def __init__(self, input_columns: list):
        self.pipe = joblib.load(MODEL_DIR / "churn_model.joblib")
        self.features = self.pipe.named_steps["features"]
        self.model = self.pipe.named_steps["model"]
        self.calibrator = joblib.load(MODEL_DIR / "calibrator.joblib")
        self.decision = json.loads((MODEL_DIR / "decision.json").read_text())
        summary = json.loads((C.REPORTS_DIR / "data_summary.json").read_text())
        self.hv_threshold = float(summary["high_value_threshold_avg_spend_jun_jul"])
        self.input_columns = list(input_columns)
        self.date_columns = [c for c in self.input_columns if c.startswith("date_of_last_rech")]
        self.tier_edges = [t[0] for t in self.decision["risk_tiers"]] + [self.decision["risk_tiers"][-1][1]]
        self.tier_names = [t[2] for t in self.decision["risk_tiers"]]

    # ---------- raw file -> cleaned rows (Step 1 rules) ----------
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if C.ID_COLUMN in df.columns:
            if df[C.ID_COLUMN].duplicated().any():
                raise ValueError("The file has duplicate mobile_number values.")
            df = df.set_index(C.ID_COLUMN)
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:8]) + (f" ... (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            raise ValueError(f"{len(missing)} required column(s) are missing: {shown}")
        df = df[self.input_columns].copy()
        for c in self.input_columns:
            if c not in self.date_columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        try:
            df, _ = D.fill_missing_by_meaning(df)
        except ValueError as e:
            raise ValueError(f"{e}. Only usage, data-recharge and date columns may be empty.") from None
        df.index.name = C.ID_COLUMN
        return df

    # ---------- cleaned rows -> scores ----------
    def _contrib(self, clean: pd.DataFrame):
        F = self.features.transform(clean)
        c = self.model.predict(F, pred_contrib=True)
        return F, c[:, :-1], c[:, -1]

    def _tier(self, prob: np.ndarray) -> list:
        return pd.cut(prob, self.tier_edges, labels=self.tier_names, right=False).astype(str).tolist()

    def score(self, clean: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        F, sv, _ = self._contrib(clean)
        raw = self.model.predict_proba(F)[:, 1]
        prob = self.calibrator.predict(raw)
        value = monthly_value(clean).to_numpy()
        out = pd.DataFrame({
            "model_score": raw, "churn_probability": prob, "risk_tier": self._tier(prob),
            "flagged": (raw >= self.decision["threshold"]).astype(int),
            "monthly_value_rs": value, "revenue_at_risk_rs": prob * value,
            "high_value": (avg_spend_good_phase(clean) >= self.hv_threshold).to_numpy().astype(int),
        }, index=clean.index)
        cols = F.columns.to_numpy()
        for r in range(1, k + 1):
            out[f"reason_{r}"] = ""
        for i in range(len(F)):
            order = np.argsort(-sv[i])[:k]
            for r, j in enumerate(order, start=1):
                if sv[i, j] > 0:
                    out.iat[i, out.columns.get_loc(f"reason_{r}")] = describe(cols[j], F.iat[i, j])
        return out

    def explain_one(self, clean_row: pd.DataFrame, up: int = 6, down: int = 4) -> dict:
        """Full picture for one customer: probability + the strongest pushes up and down."""
        F, sv, base = self._contrib(clean_row)
        s = self.score(clean_row).iloc[0]
        v, cols = sv[0], F.columns.to_numpy()

        def item(j):
            return {"feature": cols[j], "label": pretty(cols[j]), "group": group(cols[j]),
                    "text": describe(cols[j], F.iat[0, j]), "shap": round(float(v[j]), 4)}
        order = np.argsort(-v)
        return {
            "churn_probability": float(s["churn_probability"]), "model_score": float(s["model_score"]),
            "risk_tier": s["risk_tier"], "flagged": bool(s["flagged"]), "high_value": bool(s["high_value"]),
            "monthly_value_rs": float(s["monthly_value_rs"]), "revenue_at_risk_rs": float(s["revenue_at_risk_rs"]),
            "base_log_odds": float(base[0]),
            "raising": [item(j) for j in order[:up] if v[j] > 0],
            "lowering": [item(j) for j in order[::-1][:down] if v[j] < 0],
        }

    # ---------- what-if simulation on August behaviour ----------
    @staticmethod
    def august_profile(clean_row: pd.DataFrame) -> dict:
        """Current August values + June-July averages, shown next to the what-if controls."""
        r = clean_row.iloc[0]
        d8, dd8 = r["date_of_last_rech_8"], r["date_of_last_rech_data_8"]
        had_data_before = bool((r["total_rech_data_6"] > 0) or (r["total_rech_data_7"] > 0))

        def avg(base):
            return float((r[f"{base}_6"] + r[f"{base}_7"]) / 2)
        return {
            "recharge_day": None if pd.isna(d8) else int(d8.day),
            "recharge_amount": float(r["total_rech_amt_8"]), "recharge_amount_avg": avg("total_rech_amt"),
            "incoming_min": float(r["total_ic_mou_8"]), "incoming_min_avg": avg("total_ic_mou"),
            "outgoing_min": float(r["total_og_mou_8"]), "outgoing_min_avg": avg("total_og_mou"),
            "data_mb": float(r["vol_2g_mb_8"] + r["vol_3g_mb_8"]),
            "data_mb_avg": float((r["vol_2g_mb_6"] + r["vol_3g_mb_6"] + r["vol_2g_mb_7"] + r["vol_3g_mb_7"]) / 2),
            "data_recharge_day": None if pd.isna(dd8) else int(dd8.day),
            "data_recharge_possible": bool(not pd.isna(dd8) or had_data_before),
            "tenure_years": float(r["aon"] / 365),
        }

    @staticmethod
    def _set_total(row: pd.Series, cols: list, total_col: str, new_total: float, fallback_col: str) -> None:
        """Change a usage total and move all its parts by the same factor (keeps the row consistent)."""
        new_total = max(float(new_total), 0.0)
        old = float(row[total_col])
        if old > 0:
            f = new_total / old
            for c in cols:
                row[c] = row[c] * f
            return
        base = {c: (row[c[:-2] + "_6"] + row[c[:-2] + "_7"]) / 2 for c in cols}
        base_total = base[total_col]
        for c in cols:
            row[c] = base[c] * new_total / base_total if base_total > 0 else 0.0
        if base_total <= 0 and new_total > 0:
            for c in (fallback_col, total_col):
                row[c] = new_total
            parent = fallback_col.split("_t2")[0] + "_mou_8" if "_t2" in fallback_col else None
            if parent and parent in row.index:
                row[parent] = new_total

    def apply_what_if(self, clean_row: pd.DataFrame, changes: dict) -> pd.DataFrame:
        row = clean_row.iloc[0].copy()
        if "recharge_day" in changes or "recharge_amount" in changes:
            day = changes.get("recharge_day", "keep")
            if day is None:
                row["date_of_last_rech_8"] = pd.NaT
                row["total_rech_num_8"] = 0
                for c in RECHARGE_AMOUNTS_8:
                    row[c] = 0.0
            else:
                if day != "keep":
                    row["date_of_last_rech_8"] = AUG_END.replace(day=int(day))
                if pd.isna(row["date_of_last_rech_8"]):
                    raise ValueError("Pick a recharge day before setting a recharge amount.")
                amt = float(changes.get("recharge_amount", row["total_rech_amt_8"]))
                old = float(row["total_rech_amt_8"])
                if old > 0:
                    for c in RECHARGE_AMOUNTS_8:
                        row[c] = row[c] * amt / old
                else:
                    for c in RECHARGE_AMOUNTS_8:
                        row[c] = amt
                row["total_rech_num_8"] = max(int(row["total_rech_num_8"]), 1)
        if "data_recharge_day" in changes:
            day = changes["data_recharge_day"]
            if day is None:
                row["date_of_last_rech_data_8"] = pd.NaT
                for c in DATA_BLOCK_8:
                    row[c] = 0
            else:
                if float(row["total_rech_data_8"]) == 0:
                    src_m = 7 if row["total_rech_data_7"] > 0 else 6
                    if row[f"total_rech_data_{src_m}"] <= 0:
                        raise ValueError("This customer never recharged data, so there is no data pack to copy.")
                    for c in DATA_BLOCK_8:          # same data pack as their latest active month
                        row[c] = row[c[:-2] + f"_{src_m}"]
                row["date_of_last_rech_data_8"] = AUG_END.replace(day=int(day))
        cols = list(clean_row.columns)
        if "incoming_min" in changes:
            self._set_total(row, _usage_cols_8(cols, True), "total_ic_mou_8", changes["incoming_min"], "loc_ic_t2m_mou_8")
        if "outgoing_min" in changes:
            self._set_total(row, _usage_cols_8(cols, False), "total_og_mou_8", changes["outgoing_min"], "loc_og_t2m_mou_8")
        if "data_mb" in changes:
            new = max(float(changes["data_mb"]), 0.0)
            old = float(row["vol_2g_mb_8"] + row["vol_3g_mb_8"])
            if old > 0:
                row["vol_2g_mb_8"] *= new / old
                row["vol_3g_mb_8"] *= new / old
            else:
                g2 = float(row["vol_2g_mb_6"] + row["vol_2g_mb_7"])
                g3 = float(row["vol_3g_mb_6"] + row["vol_3g_mb_7"])
                share3 = g3 / (g2 + g3) if (g2 + g3) > 0 else 1.0
                row["vol_3g_mb_8"], row["vol_2g_mb_8"] = new * share3, new * (1 - share3)
        out = row.to_frame().T.astype(clean_row.dtypes.to_dict())
        out.index.name = clean_row.index.name
        return out
