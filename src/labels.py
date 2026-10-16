"""
Plain-English names for the model's features.

Shared by the SHAP report (src/explain.py) and the web app (app/), so a feature
is called the same thing everywhere. No heavy imports here on purpose.
"""
import re

from src.features import NO_RECHARGE_DAYS

MONTH = {"6": "Jun", "7": "Jul", "8": "Aug"}
BASE = {
    "spend": "recharge spend", "arpu": "ARPU", "total_rech_amt": "recharge amount",
    "total_rech_num": "number of recharges", "max_rech_amt": "largest recharge", "last_day_rch_amt": "last recharge amount",
    "total_og_mou": "outgoing minutes", "total_ic_mou": "incoming minutes", "onnet_mou": "same-network minutes",
    "offnet_mou": "other-network minutes", "loc_og_mou": "local outgoing minutes", "loc_ic_mou": "local incoming minutes",
    "std_og_mou": "STD outgoing minutes", "std_ic_mou": "STD incoming minutes", "data_mb": "mobile data (MB)",
    "total_rech_data": "data recharges", "av_rech_amt_data": "avg data recharge amount", "max_rech_data": "largest data recharge",
    "spl_ic_mou": "special incoming minutes", "roam_og_mou": "roaming outgoing minutes", "roam_ic_mou": "roaming incoming minutes",
    "loc_og_t2m_mou": "local outgoing to other mobiles", "loc_og_t2t_mou": "local outgoing same network",
    "loc_og_t2f_mou": "local outgoing to landlines", "loc_ic_t2t_mou": "local incoming same network",
    "loc_ic_t2m_mou": "local incoming from other mobiles", "vol_2g_mb": "2G data (MB)", "vol_3g_mb": "3G data (MB)",
    "monthly_2g": "monthly 2G packs", "monthly_3g": "monthly 3G packs", "sachet_2g": "sachet 2G packs",
    "sachet_3g": "sachet 3G packs", "count_rech_2g": "2G data recharges", "count_rech_3g": "3G data recharges",
}
SPECIAL = {
    "aon": "tenure on network (days)", "tenure_years": "tenure (years)", "roamed_8": "roamed in Aug (flag)",
    "no_rech_8": "no recharge in Aug (flag)", "no_data_rech_8": "no data recharge in Aug (flag)",
    "used_data_8": "used mobile data in Aug (flag)", "monthly_pack_8": "monthly data pack in Aug (flag)",
    "sachet_pack_8": "sachet pack in Aug (flag)", "no_calls_good_phase": "no calls in Jun-Jul (flag)",
    "months_with_recharge": "months with a recharge (Jun-Aug)",
}
WORDS = {"loc": "local", "std": "STD", "isd": "ISD", "og": "outgoing", "ic": "incoming", "t2t": "same network",
         "t2m": "to/from other mobiles", "t2f": "landline", "t2c": "call centre", "mou": "minutes", "spl": "special",
         "roam": "roaming", "rech": "recharge", "amt": "amount", "num": "count", "vol": "volume", "mb": "(MB)",
         "pck": "pack", "fb": "Facebook", "jun": "Jun", "jul": "Jul", "aug": "Aug", "arpu": "ARPU", "2g": "2G", "3g": "3G", "vbc": "volume-based cost", "others": "other"}
GROUP_COLORS = {"Recharge recency": "#1baf7a", "Recharge & revenue": "#eda100", "Calls": "#2a78d6",
                "Mobile data": "#d55181", "Tenure": "#6250d6", "Roaming": "#008300", "Other": "#8a8984"}


def _humanize(base: str) -> str:
    """Fallback for columns without a hand-written label, e.g. std_og_t2t_mou -> 'STD outgoing same network minutes'."""
    return " ".join(WORDS.get(w, w) for w in base.split("_"))


def pretty(name: str) -> str:
    """Technical column name -> plain-English label (for graphs and reason codes)."""
    if name in SPECIAL:
        return SPECIAL[name]
    m = re.fullmatch(r"days_since_last_(data_)?rech_([678])", name)
    if m:
        return f"days since last {'data ' if m.group(1) else ''}recharge ({MONTH[m.group(2)]})"
    m = re.fullmatch(r"(ratio|diff|good)_(.+)", name)
    if m:
        kind, base = m.groups()
        label = BASE.get(base, _humanize(base))
        return {"ratio": f"{label}: Aug / Jun-Jul", "diff": f"{label}: Aug - Jun-Jul",
                "good": f"{label}: Jun-Jul average"}[kind]
    m = re.fullmatch(r"(.+)_([678])", name)
    if m:
        base, month = m.groups()
        return f"{BASE.get(base, _humanize(base))} ({MONTH[month]})"
    return _humanize(name)


def group(name: str) -> str:
    if name.startswith("days_since") or name in ("no_rech_8", "no_data_rech_8", "months_with_recharge"):
        return "Recharge recency"
    if name in ("aon", "tenure_years"):
        return "Tenure"
    if "roam" in name:
        return "Roaming"
    if any(k in name for k in ("data", "vol_", "2g", "3g", "pack", "fb_user", "night_pck")):
        return "Mobile data"
    if any(k in name for k in ("rech", "rch", "arpu", "spend")):
        return "Recharge & revenue"
    if "mou" in name or "calls" in name or "og_" in name or "ic_" in name:
        return "Calls"
    return "Other"


def describe(name: str, value: float) -> str:
    """One reason in plain words, e.g. 'no recharge in Aug' instead of 'days_since_last_rech_8 = 32'."""
    m = re.fullmatch(r"days_since_last_(data_)?rech_([678])", name)
    if m:
        what = "data recharge" if m.group(1) else "recharge"
        if value >= NO_RECHARGE_DAYS:
            return f"no {what} in {MONTH[m.group(2)]}"
        unit = "day" if round(value) == 1 else "days"
        return f"last {what} {value:.0f} {unit} before end of {MONTH[m.group(2)]}"
    if SPECIAL.get(name, "").endswith("(flag)"):
        return f"{pretty(name).replace(' (flag)', '')}: {'yes' if value >= 0.5 else 'no'}"
    if name.startswith("ratio_"):
        return f"{pretty(name)} = {value:.2f}x"
    if name == "aon":
        return f"tenure {value / 365:.1f} years"
    return f"{pretty(name)} = {value:,.0f}"
