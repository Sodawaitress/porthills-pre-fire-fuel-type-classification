"""
Run from python/04_classification: python3 06_shap_local.py
Inputs: repository data/ (32-band main stack, 9-band spring stack, reference CSV).
Install once:  pip install shap scikit-learn pandas matplotlib tifffile imagecodecs

What it does
  - Same samples / blocks / features as 04_rf_classification.py (main 171; 300 m blocks; no grid_row/col)
  - Out-of-fold SHAP: each point is explained by a model that did NOT see its 300 m block
    (5-fold spatial-block CV), so explanations match the reported accuracy.
  - Runs both candidate main models: S+T+Sp and S+Sp.
Outputs -> ./R5_shap/
  Fig5_6_shap_<set>_<class>.png   beeswarm per class (direction: red = high value)
  Fig5_6_shap_<set>_bar.png       mean |SHAP| per class
  Tab5_12_shap_<set>.csv          mean |SHAP| and direction (Spearman value vs SHAP) per class
"""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt, shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedGroupKFold
HERE = os.getcwd(); sys.path.insert(0, HERE)
DATA = Path(__file__).resolve().parents[2] / "data"
from A3_R4_regression_io import read_stack
OUT = os.path.join(HERE, "R5_shap"); os.makedirs(OUT, exist_ok=True)

CLS  = ["pasture", "gorse_broom", "broadleaf_scrub", "exotic_pine"]
SPEC = ["pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7","NDVI","NDMI","BSI","NBR_pre"]
TERR = ["elev","slope","northness"]
SPR  = ["spr_B2","spr_B3","spr_B4","spr_B5","spr_B6","spr_B7","NDVI_spr","YI_spr","dNDVI_spr_sum"]
SETS = {"S+T+Sp": SPEC+TERR+SPR, "S+Sp": SPEC+SPR}

S, n = read_stack(str(DATA / "PortHills2017_stack.tif"))
P, pn = read_stack(str(DATA / "PortHills2016_spring.tif"))
A = np.concatenate([S, P], -1); names = list(n) + list(pn)
df = pd.read_csv(DATA / "A3_reference_cells.csv")
m = df[df.use_A3 & (df.source == "main")].reset_index(drop=True)
r, c = m.array_row.values, m.array_col.values
for i, b in enumerate(names): m[b] = A[r, c, i]
m["block"] = (r // 10).astype(str) + "_" + (c // 10).astype(str)
y = m.FuelClass.map({k: i for i, k in enumerate(CLS)}).values
print("samples:", len(m), dict(m.FuelClass.value_counts()))

def to_array(sv, nf):            # shap returns list (old) or (n, f, classes) array (new)
    if isinstance(sv, list): return np.stack(sv, -1)
    sv = np.asarray(sv); return sv if sv.shape[1] == nf else np.moveaxis(sv, 1, -1)

for tag, cols in SETS.items():
    X = m[cols]; SV = np.zeros((len(m), len(cols), 4))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, m.block):
        mod = RF(500, max_features="sqrt", random_state=0, n_jobs=-1).fit(X.iloc[tr], y[tr])
        SV[te] = to_array(shap.TreeExplainer(mod).shap_values(X.iloc[te]), len(cols))
    rows = []
    for k, cl in enumerate(CLS):
        for j, f in enumerate(cols):
            rho = spearmanr(X[f], SV[:, j, k]).correlation
            rows.append([cl, f, np.abs(SV[:, j, k]).mean(), rho])
        plt.figure(); shap.summary_plot(SV[:, :, k], X, max_display=10, show=False)
        plt.title(f"{tag}: {cl}", fontsize=10); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"Fig5_6_shap_{tag.replace('+','_')}_{cl}.png"), dpi=250); plt.close("all")
    T = pd.DataFrame(rows, columns=["class", "var", "mean_abs_shap", "direction_rho"])
    T.to_csv(os.path.join(OUT, f"Tab5_12_shap_{tag.replace('+','_')}.csv"), index=False)
    W = T.pivot(index="var", columns="class", values="mean_abs_shap")[CLS]
    W = W.loc[W.sum(1).sort_values().index[-12:]]
    W.plot.barh(figsize=(7, 6), color=["#d4a017", "#c0392b", "#27ae60", "#1f4e79"])
    plt.xlabel("mean |SHAP| (class probability)"); plt.title(f"{tag}: per-class SHAP (out-of-fold)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"Fig5_6_shap_{tag.replace('+','_')}_bar.png"), dpi=250); plt.close("all")
    print(f"\n== {tag}: top-3 per class (mean|SHAP|, direction) ==")
    for cl in CLS:
        t = T[T["class"] == cl].nlargest(3, "mean_abs_shap")
        print(f"{cl:16s}", "; ".join(f"{v} {a:.3f} ({'+' if d > 0 else '-'})" for v, a, d in t[["var", "mean_abs_shap", "direction_rho"]].values))
print("\ndone ->", OUT)
