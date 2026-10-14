#==========================
# KT4 | Figure 4, correlation among the 22 predictors
#==========================
#S  = pre-fire bands + NDVI, NDMI, BSI, NBR_pre
#T  = elevation, slope, northness
#Sp = spring 2016 bands + NDVI_spr, YI_spr, dNDVI_spr_sum
#same lists as 04_classification/00_common.py
#cells: core fire cells (inside_aoi == 1), same as the exploration step
#Spearman, not Pearson, because most variables are not normal (Figure 2)
#run from this folder: python3 07_predictor_correlation.py
#output: figures/fig_predictor_correlation.png, tables/predictor_correlation_spearman.csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

DATA = Path(__file__).resolve().parents[2] / "data"
SPEC = ["pre_B2", "pre_B3", "pre_B4", "pre_B5", "pre_B6", "pre_B7", "NDVI", "NDMI", "BSI", "NBR_pre"]
TERR = ["elev", "slope", "northness"]
SPR = ["spr_B2", "spr_B3", "spr_B4", "spr_B5", "spr_B6", "spr_B7", "NDVI_spr", "YI_spr", "dNDVI_spr_sum"]
VARS = SPEC + TERR + SPR


def read(path):
    with rasterio.open(path) as src:
        a = src.read().astype("float32")
        a[a == src.nodata] = np.nan
        return {name: a[i] for i, name in enumerate(src.descriptions)}


S = read(DATA / "PortHills2017_stack.tif")
P = read(DATA / "PortHills2016_spring.tif")          # same 30 m grid as the main stack
bands = {**S, **P}
core = bands["inside_aoi"] == 1
df = pd.DataFrame({v: bands[v][core] for v in VARS}).dropna()
print(f"core cells: {int(core.sum()):,}; with all 22 variables valid: {len(df):,}")

rho = df.rank().corr()                               # Spearman = Pearson on ranks
Path("tables").mkdir(exist_ok=True)
rho.round(3).to_csv("tables/predictor_correlation_spearman.csv")

# strongly redundant pairs, for the text
pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(VARS) for b in VARS[i + 1:] if abs(rho.loc[a, b]) >= 0.8]
print(f"\npairs with |rho| >= 0.8: {len(pairs)} of {len(VARS) * (len(VARS) - 1) // 2}")
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
    print(f"  {a:>14} ~ {b:<14} {r:+.2f}")

# heatmap, lower triangle only (the matrix is symmetric)
m = rho.values.copy()
m[np.triu_indices_from(m)] = np.nan
fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
n = len(VARS)
for i in range(n):
    for j in range(i):
        r = rho.iat[i, j]
        if abs(r) >= 0.8:   # label only the strongly redundant pairs; colour shows the rest
            ax.text(j, i, f"{r:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white")
ax.set_xticks(range(n)); ax.set_xticklabels(VARS, rotation=90, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(VARS, fontsize=8)
for edge in (len(SPEC) - 0.5, len(SPEC) + len(TERR) - 0.5):   # separate S / T / Sp blocks
    ax.axhline(edge, color="grey", lw=0.8); ax.axvline(edge, color="grey", lw=0.8)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.set_label("Spearman ρ")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/fig_predictor_correlation.png", dpi=200, bbox_inches="tight")
print("\nsaved figures/fig_predictor_correlation.png, tables/predictor_correlation_spearman.csv")
