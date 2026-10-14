#==========================
# KT4 | Figure 3, transformations
#==========================
#skewness and kurtosis of the 13 pre-fire variables, before and after
#log, square root, Yeo-Johnson and z-score
#reads the table from 03_normality_transforms.R, so the figure can't drift from it
#left-skewed variables are flipped before log and square root (prep_pos() in the R script)
#run after the R script: python3 07_fig_transformations.py
#output: figures/A3_transformations.png
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t = pd.read_csv("tables/A3_normality.csv")
y = np.arange(len(t))[::-1]                       # first variable at the top

METHODS = [  # (column suffix, label, marker, colour, vertical offset)
    ("raw",  "Raw",         "o", "#3B2A22", -0.15),
    ("log",  "Log",         "^", "#B5441F", -0.08),
    ("sqrt", "Square root", "s", "#E8A03A",  0.00),
    ("yj",   "Yeo-Johnson", "D", "#2E6B3A",  0.08),
    ("z",    "Z-score",     "x", "#8A96AA",  0.15),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, stat, target, band, title in [
        (axes[0], "skew", 0, (-0.5, 0.5), "Skewness (target 0)"),
        (axes[1], "kurt", 3, (2, 4),      "Kurtosis (target 3)")]:
    ax.axvspan(*band, color="#F2E9DF", zorder=0)          # roughly normal range
    ax.axvline(target, color="#7A6A5E", ls="--", lw=0.8, zorder=1)
    for suf, label, mk, col, dy in METHODS:
        ax.scatter(t[f"{stat}_{suf}"], y + dy, marker=mk, color=col, s=36, label=label, zorder=3)
    ax.set_title(title, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(t["variable"])
axes[1].legend(loc="lower right", frameon=False)
fig.suptitle("Effect of transformations (z-score overlaps raw: it changes scale, not shape)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/A3_transformations.png", dpi=200, bbox_inches="tight")
print("saved figures/A3_transformations.png")
