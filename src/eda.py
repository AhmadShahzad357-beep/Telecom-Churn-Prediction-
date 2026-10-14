"""
Step 2 - EDA helpers. Every function works on the TRAINING set only and saves
its figure to reports/figures. The notebook notebooks/01_eda.ipynb calls them.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data import monthly_spend

FIG_DIR = C.REPORTS_DIR / "figures"

# Colour roles (colour-blind checked). Churn / non-churn keep the same colour in every chart.
NON_CHURN = "#2a78d6"   # blue
CHURN = "#eb6834"       # orange
MONTH_COLORS = {"June": "#6250d6", "July": "#1baf7a", "August": "#eda100", "No month": "#8a8984"}
SINGLE = {"aqua": "#1baf7a", "violet": "#6250d6", "green": "#008300",
          "magenta": "#d55181", "yellow": "#c98500"}
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#ffffff"
MONTHS = {6: "June", 7: "July", 8: "August"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 10,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    })


def clear_old_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in FIG_DIR.glob("*.png"):
        f.unlink()


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")


def _class_legend(ax, loc="upper left", anchor=(0, 1.13)):
    handles = [plt.Line2D([], [], color=NON_CHURN, lw=2.5, marker="o", label="Stayed (non-churn)"),
               plt.Line2D([], [], color=CHURN, lw=2.5, marker="o", label="Churned")]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=2)


def ramp(base: str, n: int) -> list:
    """n distinct shades of one hue, light -> dark (every shade stays readable on white)."""
    r, g, b = to_rgb(base)
    light = tuple(c + (1 - c) * 0.55 for c in (r, g, b))
    dark = tuple(c * 0.55 for c in (r, g, b))
    cmap = LinearSegmentedColormap.from_list("ramp", [light, (r, g, b), dark])
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def _rate_bars(ax, rates: pd.Series, sizes: pd.Series, colors, overall: float, title: str) -> None:
    """Churn-rate bars - every bar has its own colour and its own legend entry."""
    x = np.arange(len(rates))
    bars = ax.bar(x, rates.values * 100, color=colors, width=0.62, zorder=2)
    line = ax.axhline(overall * 100, color=MUTED, lw=1, zorder=1)
    for i, r in enumerate(rates.values):
        ax.text(i, r * 100 + rates.max() * 100 * 0.02, f"{r:.1%}", ha="center", va="bottom",
                fontsize=9, color=TEXT, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(rates.index, sizes.values)])
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, rates.max() * 100 * 1.18)
    handles = list(bars) + [line]
    labels = [str(lab) for lab in rates.index] + [f"Overall churn {overall:.1%}"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(handles),
              fontsize=9, handlelength=1.2, columnspacing=1.2)
    ax.set_title(title, pad=30)


# ---------------------------------------------------------------- figures
def plot_target_balance(df):
    counts = df[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    labels = ["Stayed (non-churn)", "Churned"]
    colors = [NON_CHURN, CHURN]
    ax.barh(labels, counts.values, color=colors, height=0.55)
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.01, i, f"{v:,}  ({v / counts.sum():.1%})", va="center", color=TEXT)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Customers in training set")
    ax.set_title("Only 1 in 15 high-value customers churns")
    ax.invert_yaxis()
    _save(fig, "01_target_balance")
    return fig


def plot_phase_trends(df):
    tmp = df.copy()
    for m in MONTHS:
        tmp[f"spend_{m}"] = monthly_spend(tmp, m)
        tmp[f"data_mb_{m}"] = tmp[f"vol_2g_mb_{m}"] + tmp[f"vol_3g_mb_{m}"]
    panels = [("spend", "Recharge spend (Rs, median)"), ("total_og_mou", "Outgoing minutes (median)"),
              ("total_ic_mou", "Incoming minutes (median)"), ("data_mb", "Mobile data MB (mean)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (base, title) in zip(axes.ravel(), panels):
        agg = "mean" if base == "data_mb" else "median"
        for cls, color in ((0, NON_CHURN), (1, CHURN)):
            vals = [getattr(tmp.loc[tmp[C.TARGET] == cls, f"{base}_{m}"], agg)() for m in MONTHS]
            ax.plot(list(MONTHS.values()), vals, color=color, lw=2.2, marker="o", ms=7)
            ax.annotate(f"{vals[-1]:,.0f}", (2, vals[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.2, 2.45)
    _class_legend(axes[0, 0], anchor=(0, 1.22))
    fig.suptitle("Churners collapse in August - the action phase", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "02_good_vs_action_phase")
    return fig


def plot_august_spend_change(df):
    good = (monthly_spend(df, 6) + monthly_spend(df, 7)) / 2
    change = ((monthly_spend(df, 8) - good) / good.replace(0, np.nan) * 100).clip(-100, 200)
    data = [change[df[C.TARGET] == 0].dropna(), change[df[C.TARGET] == 1].dropna()]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=TEXT, lw=1.6))
    for patch, color in zip(bp["boxes"], (NON_CHURN, CHURN)):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for i, d in enumerate(data, start=1):
        ax.text(d.median() + 3, i - 0.36, f"median {d.median():+.0f}%", ha="left", va="center",
                fontsize=9, color=TEXT, zorder=3, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    ax.set_yticks([1, 2], ["Stayed (non-churn)", "Churned"])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Change in August spend vs June-July average (%)  -  clipped to [-100, 200]")
    ax.set_title("Half of churners stop recharging completely in August")
    _save(fig, "03_august_spend_change")
    return fig


def univariate_strength(df) -> pd.DataFrame:
    num = df.select_dtypes("number").drop(columns=[C.TARGET])
    rows = []
    for c in num.columns:
        auc = roc_auc_score(df[C.TARGET], num[c])
        month = next((MONTHS[m] for m in MONTHS if c.endswith(f"_{m}")), "No month")
        rows.append({"feature": c, "auc": auc, "strength": abs(auc - 0.5) * 2,
                     "direction": "lower -> more churn" if auc < 0.5 else "higher -> more churn",
                     "month": month})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def plot_univariate_strength(df, top: int = 10):
    res = univariate_strength(df)
    res["base"] = res["feature"].str.replace(r"_[678]$", "", regex=True)
    monthly = res[res["month"] != "No month"]
    bases = monthly[monthly["month"] == "August"].head(top)["base"].tolist()
    table = monthly[monthly["base"].isin(bases)].pivot(index="base", columns="month", values="strength")
    table = table.loc[bases[::-1], ["June", "July", "August"]]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    y = np.arange(len(table))
    h = 0.26
    for k, month in enumerate(["June", "July", "August"]):
        vals = table[month].values
        ax.barh(y + (k - 1) * h, vals, height=h - 0.03, color=MONTH_COLORS[month], label=month)
        if month == "August":
            for yi, v in zip(y, vals):
                ax.text(v + 0.006, yi + h, f"{v:.2f}", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y, table.index)
    ax.set_xlim(0, table.values.max() * 1.12)
    ax.set_xlabel("Separation strength = |AUC - 0.5| x 2   (0 = none, 1 = perfect)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.06), ncol=3)
    ax.set_title("Same metric, three months: August separates churners far better", pad=30)
    _save(fig, "04_univariate_strength_by_month")
    return fig, res.drop(columns="base")


def plot_last_recharge(df):
    days = (pd.Timestamp("2014-08-31") - df["date_of_last_rech_8"]).dt.days
    order = ["0-3", "4-7", "8-14", "15-21", "22-31", "No recharge"]
    bucket = pd.cut(days, [-1, 3, 7, 14, 21, 31], labels=order[:-1]).astype(object)
    bucket[days.isna()] = "No recharge"
    g = df.groupby(bucket)[C.TARGET].agg(["mean", "size"]).reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _rate_bars(ax, g["mean"], g["size"], ramp(SINGLE["aqua"], len(g)), df[C.TARGET].mean(),
               "The longer since the last recharge, the higher the churn")
    ax.set_xlabel("Days between last recharge and end of August")
    _save(fig, "05_churn_by_last_recharge")
    return fig


def plot_tenure(df):
    labels = ["< 1 yr", "1-2 yrs", "2-3 yrs", "3-5 yrs", "5+ yrs"]
    bucket = pd.cut(df["aon"] / 365, [0, 1, 2, 3, 5, 100], labels=labels)
    g = df.groupby(bucket, observed=True)[C.TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _rate_bars(ax, g["mean"], g["size"], ramp(SINGLE["violet"], len(g)), df[C.TARGET].mean(),
               "New customers churn ~4x more than loyal ones")
    ax.set_xlabel("Tenure on network (aon)")
    _save(fig, "06_churn_by_tenure")
    return fig


def plot_data_usage(df):
    used = (df["vol_2g_mb_8"] + df["vol_3g_mb_8"]) > 0
    labels = used.map({True: "Used mobile data", False: "Voice only (no data)"})
    g = df.groupby(labels)[C.TARGET].agg(["mean", "size"]).reindex(["Voice only (no data)", "Used mobile data"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _rate_bars(ax, g["mean"], g["size"], [SINGLE["magenta"], SINGLE["green"]], df[C.TARGET].mean(),
               "Voice-only customers churn ~5x more")
    ax.set_xlabel("Mobile data usage in August")
    _save(fig, "07_churn_by_data_usage")
    return fig


def plot_arpu_decile(df):
    dec = pd.qcut(df["arpu_8"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    g = df.groupby(dec, observed=True)[C.TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _rate_bars(ax, g["mean"], g["size"], ramp(SINGLE["yellow"], len(g)), df[C.TARGET].mean(),
               "Churn is concentrated in the lowest August-revenue decile")
    ax.set_xlabel("August ARPU decile (D1 = lowest revenue, D10 = highest)")
    _save(fig, "08_churn_by_arpu_decile")
    return fig


def plot_zero_share(df, top: int = 25):
    num = df.select_dtypes("number").drop(columns=[C.TARGET])
    zero = (num == 0).mean().sort_values(ascending=False)
    top_s = zero.head(top).iloc[::-1] * 100
    fig, ax = plt.subplots(figsize=(9, 7.5))
    months = [next((MONTHS[m] for m in MONTHS if c.endswith(f"_{m}")), "No month") for c in top_s.index]
    ax.barh(top_s.index, top_s.values, color=[MONTH_COLORS[m] for m in months], height=0.62)
    ax.axvline(75, color=MUTED, lw=1)
    for y, v in enumerate(top_s.values):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=8.5, color=TEXT)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Customers with value = 0 (%)")
    shown = [m for m in MONTH_COLORS if m in months]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MONTH_COLORS[m]) for m in shown]
    handles.append(plt.Line2D([], [], color=MUTED, lw=1))
    ax.legend(handles, shown + ["75% zeros - IQR becomes 0"], loc="lower left",
              bbox_to_anchor=(0, 1.01), ncol=len(handles), fontsize=9)
    n_half = int((zero > 0.5).sum())
    ax.set_title(f"{n_half} of {len(zero)} numeric columns are more than half zeros", pad=30)
    _save(fig, "09_zero_heavy_columns")
    return fig, zero


def plot_month_correlation(df):
    tmp = pd.DataFrame(index=df.index)
    names = {"spend": "Spend", "total_og_mou": "Outgoing", "total_ic_mou": "Incoming", "data": "Data MB"}
    for m in MONTHS:
        tmp[f"Spend {MONTHS[m][:3]}"] = monthly_spend(df, m)
        tmp[f"Outgoing {MONTHS[m][:3]}"] = df[f"total_og_mou_{m}"]
        tmp[f"Incoming {MONTHS[m][:3]}"] = df[f"total_ic_mou_{m}"]
        tmp[f"Data MB {MONTHS[m][:3]}"] = df[f"vol_2g_mb_{m}"] + df[f"vol_3g_mb_{m}"]
    order = [f"{n} {MONTHS[m][:3]}" for n in names.values() for m in MONTHS]
    corr = tmp[order].corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    for i in range(len(order)):
        for j in range(len(order)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.7 else TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman correlation")
    cb.outline.set_visible(False)
    ax.set_title("Usage repeats month to month; recharge spend does not")
    _save(fig, "10_month_correlation")
    return fig, corr


def plot_spend_distribution(df):
    good = (monthly_spend(df, 6) + monthly_spend(df, 7)) / 2
    bins = np.logspace(np.log10(good.min()), np.log10(good.max()), 45)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (cls, color, name) in zip(axes, ((0, NON_CHURN, "Stayed (non-churn)"), (1, CHURN, "Churned"))):
        vals = good[df[C.TARGET] == cls]
        ax.hist(vals, bins=bins, color=color, label=name)
        ax.axvline(vals.median(), color=TEXT, lw=1.2, label=f"Median Rs {vals.median():,.0f}")
        ax.set_xscale("log")
        ax.set_xlabel("Average June-July spend (Rs, log scale)")
        ax.set_ylabel("Customers")
        ax.legend(loc="upper right")
    skew = good.skew()
    fig.suptitle(f"Spend is extremely right-skewed (skewness {skew:.1f}) - log scale needed to see it",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_spend_distribution")
    return fig


def plot_recharge_count(df):
    labels = ["0", "1", "2", "3-5", "6-10", "10+"]
    bucket = pd.cut(df["total_rech_num_8"], [-1, 0, 1, 2, 5, 10, 10_000], labels=labels)
    g = df.groupby(bucket, observed=True)[C.TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _rate_bars(ax, g["mean"], g["size"], ramp(SINGLE["green"], len(g))[::-1], df[C.TARGET].mean(),
               "Fewer recharges in August, higher churn")
    ax.set_xlabel("Number of recharges in August")
    _save(fig, "12_churn_by_recharge_count")
    return fig


def plot_outgoing_change(df):
    good = (df["total_og_mou_6"] + df["total_og_mou_7"]) / 2
    change = (df["total_og_mou_8"] - good) / good.replace(0, np.nan)
    order = ["Rise >25%", "Stable +/-25%", "Drop 25-50%", "Drop 50-75%", "Drop >75%", "No calls Jun-Jul"]
    bucket = pd.cut(change, [-1.01, -0.75, -0.5, -0.25, 0.25, np.inf],
                    labels=order[4::-1]).astype(object)
    bucket[good == 0] = "No calls Jun-Jul"
    g = df.groupby(bucket)[C.TARGET].agg(["mean", "size"]).reindex(order)
    colors = ramp(SINGLE["magenta"], 5) + ["#8a8984"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _rate_bars(ax, g["mean"], g["size"], colors, df[C.TARGET].mean(),
               "A >75% fall in outgoing calls means ~18x the churn of stable users")
    ax.set_xlabel("Change in August outgoing minutes vs June-July average")
    _save(fig, "13_churn_by_outgoing_change")
    return fig


def plot_roaming(df):
    roam = ((df["roam_og_mou_8"] + df["roam_ic_mou_8"]) > 0).map({True: "Roamed in August", False: "No roaming"})
    g = df.groupby(roam)[C.TARGET].agg(["mean", "size"]).reindex(["No roaming", "Roamed in August"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _rate_bars(ax, g["mean"], g["size"], [SINGLE["aqua"], SINGLE["violet"]], df[C.TARGET].mean(),
               "Roaming customers churn ~2x more")
    ax.set_xlabel("Roaming usage in August")
    _save(fig, "14_churn_by_roaming")
    return fig


def plot_pack_type(df):
    monthly = (df["monthly_2g_8"] + df["monthly_3g_8"]) > 0
    sachet = (df["sachet_2g_8"] + df["sachet_3g_8"]) > 0
    pack = pd.Series(np.select([monthly, sachet], ["Monthly data pack", "Sachet packs only"], "No data pack"),
                     index=df.index)
    order = ["No data pack", "Sachet packs only", "Monthly data pack"]
    g = df.groupby(pack)[C.TARGET].agg(["mean", "size"]).reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _rate_bars(ax, g["mean"], g["size"], [SINGLE["yellow"], SINGLE["aqua"], SINGLE["green"]], df[C.TARGET].mean(),
               "Monthly-pack customers are the most loyal")
    ax.set_xlabel("Type of data pack bought in August")
    _save(fig, "15_churn_by_pack_type")
    return fig


def plot_revenue_share(df):
    churn = df[C.TARGET] == 1
    good_rev = (df["arpu_6"] + df["arpu_7"]) / 2
    shares = pd.Series({
        "Share of customers": churn.mean(),
        "Share of June-July revenue": good_rev[churn].sum() / good_rev.sum(),
        "Share of August revenue": df.loc[churn, "arpu_8"].sum() / df["arpu_8"].sum(),
    })
    colors = [SINGLE["violet"], CHURN, SINGLE["yellow"]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.barh(shares.index[::-1], shares.values[::-1] * 100, color=colors[::-1], height=0.55)
    for i, v in enumerate(shares.values[::-1]):
        ax.text(v * 100 + 0.12, i, f"{v:.1%}", va="center", color=TEXT)
    ax.set_xlim(0, shares.max() * 100 * 1.2)
    ax.set_xlabel("Churners as % of the training set")
    ax.legend(bars[::-1], list(shares.index), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=9)
    avg_c, avg_n = good_rev[churn].mean(), good_rev[~churn].mean()
    ax.set_title(f"Churners were worth more before leaving (Rs {avg_c:,.0f} vs Rs {avg_n:,.0f} per month)", pad=30)
    _save(fig, "16_churner_revenue_share")
    return fig