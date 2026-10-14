"""
Section 5 step 1b: class separability on the 171 main cells.
Per-variable Kruskal-Wallis eta^2 + best pairwise AUC; pairwise Jeffries-Matusita
distance for three band sets; boxplots of the most useful variables.
"""
import numpy as np, pandas as pd, itertools, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

exec(open("00_common.py").read())          # -> load(), CLS, COL, SPEC/TERR/SPR
d, A, names = load(); m = d[d.source == "main"]
V = SPEC + TERR + SPR

# Per variable: Kruskal-Wallis eta^2 (how strongly class separates it) + best pairwise AUC
rows = []
for v in V:
    g = [m.loc[m.FuelClass == k, v].values for k in CLS]
    H = kruskal(*g).statistic; n, kk = len(m), 4
    row = {"var": v, "eta2_KW": (H - kk + 1) / (n - kk)}
    for a, b in itertools.combinations(CLS, 2):
        s = m[m.FuelClass.isin([a, b])]; au = roc_auc_score(s.FuelClass == a, s[v])
        row[f"{a}|{b}"] = max(au, 1 - au)
    rows.append(row)
T = pd.DataFrame(rows).round(3); T.to_csv("Tab5_2_variable_separability.csv", index=False)

def jm(X1, X2):                             # Jeffries-Matusita distance (0-2); ~2 = well separated
    m1, m2 = X1.mean(0), X2.mean(0); C1, C2 = np.cov(X1.T), np.cov(X2.T); C = (C1 + C2) / 2
    dm = m1 - m2
    B = dm @ np.linalg.solve(C, dm) / 8 + 0.5 * np.log(np.linalg.det(C) / np.sqrt(np.linalg.det(C1) * np.linalg.det(C2)))
    return 2 * (1 - np.exp(-B))

# JM distance per class pair, for three band sets
jr = []
for name, cols in {"pre 6 bands": SPEC[:6], "spring 6 bands": SPR[:6], "pre+spring 12 bands": SPEC[:6] + SPR[:6]}.items():
    for a, b in itertools.combinations(CLS, 2):
        jr.append([name, f"{a}|{b}", jm(m.loc[m.FuelClass == a, cols].values, m.loc[m.FuelClass == b, cols].values)])
J = pd.DataFrame(jr, columns=["bands", "pair", "JM"]).pivot(index="pair", columns="bands", values="JM").round(2)
J.to_csv("Tab5_3_JM_distance.csv")

# Boxplots of nine informative variables
show = ["pre_B4", "pre_B5", "pre_B6", "NDVI", "NDMI", "BSI", "NDVI_spr", "YI_spr", "dNDVI_spr_sum"]
fig, ax = plt.subplots(3, 3, figsize=(10, 8.5))
for a, v in zip(ax.flat, show):
    bp = a.boxplot([m.loc[m.FuelClass == k, v] for k in CLS], patch_artist=True, widths=.6, showfliers=False)
    for p, k in zip(bp["boxes"], CLS): p.set_facecolor(COL[k]); p.set_alpha(.6)
    a.set_xticks(range(1, 5)); a.set_xticklabels(["pasture", "gorse", "broadl.", "pine"], fontsize=8)
    e = T.set_index("var").loc[v, "eta2_KW"]; a.set_title(f"{v}  (η²={e:.2f})", fontsize=9); a.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("Fig5_2_boxplots.png", dpi=300)
print(T.sort_values("eta2_KW", ascending=False).head(10).to_string()); print(J)
print("\nbest single var gorse|broadleaf:", T.loc[T["gorse_broom|broadleaf_scrub"].idxmax(), ["var", "gorse_broom|broadleaf_scrub"]].values)
