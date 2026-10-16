"""
Step 7 - Churn Radar web app: FastAPI backend + plain HTML/CSS/JS dashboard (app/static).

Run from the project root:
    python -m uvicorn app.main:app --port 8000
then open http://127.0.0.1:8000

Deployed as-is on Vercel: it detects `app` in app/main.py (see vercel.json).
"""
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so `src` imports work on Vercel too


def _ensure_libgomp():
    """LightGBM needs libgomp.so.1 (OpenMP). Vercel's Linux runtime does not have it, so the app crashed with
    500 FUNCTION_INVOCATION_FAILED. scikit-learn ships its own copy under a renamed file; copy it to /tmp with
    the standard name and load it once, before LightGBM is imported. Does nothing on Windows or when the
    system already has libgomp (a normal laptop / server)."""
    import ctypes
    import glob
    import os
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    import sklearn
    libs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(sklearn.__file__)), "scikit_learn.libs", "libgomp*.so*"))
    if not libs:
        return
    data = Path(libs[0]).read_bytes()
    old, new = os.path.basename(libs[0]).encode(), b"libgomp.so.1"
    data = data.replace(old + b"\0", new + b"\0" * (len(old) - len(new) + 1))
    dst = Path("/tmp/libgomp.so.1")
    dst.write_bytes(data)
    ctypes.CDLL(str(dst), mode=ctypes.RTLD_GLOBAL)


_ensure_libgomp()

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config as C
from src.predict import ChurnScorer

APP_DIR = C.ROOT / "app"
DATA = APP_DIR / "data"
MAX_UPLOAD_MB = 4          # Vercel limits a request body to 4.5 MB
MAX_ROWS = 20000

INPUT_COLS = json.loads((DATA / "input_columns.json").read_text())
scorer = ChurnScorer(INPUT_COLS)
DATE_COLS = [c for c in INPUT_COLS if c.startswith("date_of_last_rech")]
customers = pd.read_csv(DATA / "customers.csv.gz", index_col=C.ID_COLUMN, parse_dates=DATE_COLS)
scored = pd.read_csv(DATA / "scored.csv.gz", index_col=C.ID_COLUMN, keep_default_na=False)
dashboard = json.loads((DATA / "dashboard.json").read_text())

app = FastAPI(title="Churn Radar", version="1.0")


def clean_json(obj):
    """NaN / inf are not valid JSON - send them as null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj


def rows_to_records(df: pd.DataFrame) -> list:
    out = df.reset_index().rename(columns={C.ID_COLUMN: "customer_id", "index": "customer_id"})
    return clean_json(json.loads(out.to_json(orient="records")))


def get_row(customer_id: int) -> pd.DataFrame:
    if customer_id not in customers.index:
        raise HTTPException(404, f"Customer {customer_id} is not in the demo (test) set.")
    return customers.loc[[customer_id], INPUT_COLS]


@app.get("/api/health")
def health():
    return {"status": "ok", "customers": len(customers)}


@app.get("/api/dashboard")
def get_dashboard():
    return dashboard


@app.get("/api/customers")
def list_customers(tier: str = "", q: str = "", flagged: bool = False, page: int = 1, size: int = 15,
                   sort: str = "revenue_at_risk_rs"):
    df = scored
    if tier:
        df = df[df["risk_tier"] == tier]
    if flagged:
        df = df[df["flagged"] == 1]
    if q.strip():
        df = df[df.index.astype(str).str.contains(q.strip(), regex=False)]
    if sort not in ("revenue_at_risk_rs", "churn_probability", "monthly_value_rs"):
        sort = "revenue_at_risk_rs"
    df = df.sort_values(sort, ascending=False)
    size = max(5, min(size, 100))
    page = max(page, 1)
    total = len(df)
    part = df.iloc[(page - 1) * size: page * size]
    return {"total": total, "page": page, "size": size, "rows": rows_to_records(part)}


@app.get("/api/random-customer")
def random_customer(tier: str = ""):
    df = scored if not tier else scored[scored["risk_tier"] == tier]
    if df.empty:
        raise HTTPException(404, "No customer in that tier.")
    return {"customer_id": int(df.sample(1).index[0])}


@app.get("/api/customer/{customer_id}")
def customer(customer_id: int):
    row = get_row(customer_id)
    return clean_json({"customer_id": customer_id,
                       "actual_churn": int(scored.at[customer_id, "actual_churn"]),
                       "profile": scorer.august_profile(row),
                       "result": scorer.explain_one(row)})


class WhatIf(BaseModel):
    customer_id: int
    changes: dict


@app.post("/api/what-if")
def what_if(req: WhatIf):
    row = get_row(req.customer_id)
    allowed = {"recharge_day", "recharge_amount", "incoming_min", "outgoing_min", "data_mb", "data_recharge_day"}
    changes = {k: v for k, v in req.changes.items() if k in allowed}
    try:
        new_row = scorer.apply_what_if(row, changes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return clean_json({"customer_id": req.customer_id, "changes": changes,
                       "profile": scorer.august_profile(new_row), "result": scorer.explain_one(new_row)})


@app.post("/api/score-file")
async def score_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not read the file. Please upload a CSV in the original dataset format.")
    if raw.empty:
        raise HTTPException(400, "The file has no rows.")
    if len(raw) > MAX_ROWS:
        raise HTTPException(400, f"Please upload at most {MAX_ROWS:,} rows at a time.")
    if C.ID_COLUMN not in raw.columns:
        raise HTTPException(400, f"The file needs a '{C.ID_COLUMN}' column to identify customers.")
    try:
        clean = scorer.prepare(raw)
        out = scorer.score(clean)
    except ValueError as e:
        raise HTTPException(400, str(e))
    out = out.sort_values("revenue_at_risk_rs", ascending=False)
    tiers = out["risk_tier"].value_counts().reindex(scorer.tier_names, fill_value=0)
    return {
        "rows": rows_to_records(out.drop(columns=["model_score"])),
        "summary": {"customers": int(len(out)), "flagged": int(out["flagged"].sum()),
                    "revenue_at_risk_rs": float(out["revenue_at_risk_rs"].sum()),
                    "outside_scope": int((out["high_value"] == 0).sum()),
                    "tiers": {k: int(v) for k, v in tiers.items()}},
    }


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
