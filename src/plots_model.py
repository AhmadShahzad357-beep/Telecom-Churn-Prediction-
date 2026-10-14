"""Figures for Step 4 (same style as the EDA: no grid, own colour per bar, legend above)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import eda

eda.set_style()


def _bars(ax, labels, means, stds, colors):
    """Value printed inside the bar so markers above the bar never cover it."""
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6, capsize=5,
                  error_kw=dict(ecolor=eda.MUTED, lw=1))
    for i, m in enumerate(means):
        ax.text(i, m / 2, f"{m:.3f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white")
    ax.set_xticks(x, labels)
    return bars


def plot_family_comparison(df: pd.DataFrame):
    colors = [eda.SINGLE["violet"], eda.SINGLE["green"], eda.SINGLE["magenta"], eda.SINGLE["aqua"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, title in ((axes[0], "pr_auc", "PR-AUC (main metric)"), (axes[1], "roc_auc", "ROC-AUC")):
        bars = _bars(ax, df["model"], df[f"cv_{metric}"], df[f"cv_{metric}_std"], colors)
        ax.scatter(np.arange(len(df)), df[f"train_{metric}"], marker="D", s=40, color=eda.TEXT, zorder=3)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=11)
        handles = list(bars) + [plt.Line2D([], [], marker="D", ls="", color=eda.TEXT)]
        ax.legend(handles, list(df["model"]) + ["Train score"], loc="lower left",
                  bbox_to_anchor=(0, 1.06), ncol=3, fontsize=8.5)
    fig.suptitle("Model families - 5-fold CV on train (bar = CV mean, error bar = 1 std, diamond = train)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_strategies(df: pd.DataFrame):
    colors = [eda.SINGLE["yellow"], eda.SINGLE["aqua"], eda.SINGLE["violet"]]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = _bars(ax, df["imbalance"], df["cv_pr_auc"], df["cv_pr_auc_std"], colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("CV PR-AUC")
    ax.legend(bars, list(df["imbalance"]), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=9)
    ax.set_title(f"Imbalance handling for {df['model'].iloc[0]} (SMOTE applied inside each fold)", pad=30)
    return fig


def plot_learning_curve(curve: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for col, std, color, label in (("train_pr_auc", "train_std", eda.SINGLE["violet"], "Train"),
                                   ("cv_pr_auc", "cv_std", eda.SINGLE["aqua"], "3-fold CV")):
        ax.plot(curve["train_size"], curve[col], marker="o", lw=2.2, ms=7, color=color, label=label)
        ax.fill_between(curve["train_size"], curve[col] - curve[std], curve[col] + curve[std],
                        color=color, alpha=0.15, lw=0)
        ax.annotate(f"{curve[col].iloc[-1]:.3f}", (curve["train_size"].iloc[-1], curve[col].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Training customers used")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    ax.set_title("Learning curve - gap between train and CV shows how much the model memorises", pad=30)
    return fig