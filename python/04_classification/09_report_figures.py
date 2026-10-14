"""
Section 5 report figures:
  Fig 5.2 - accuracy by predictor set (spatial CV vs independent test): overall + gorse.
  Fig 5.4 - SHAP panels (bar + gorse beeswarm) from the local SHAP run (step 06).
Reads the tables written by step 04 and the SHAP PNGs in ./R5_shap.
"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cv = pd.read_csv("Tab5_6_cv_by_repeat.csv"); te = pd.read_csv("Tab5_8_expansion_test.csv")
order = ["S", "S+T", "S+Sp", "S+T+Sp"]; lab = ["Spectral", "+ terrain", "+ spring", "All"]
g = cv.groupby("set").agg(OA=("OA", "mean"), sd=("OA", "std"), G=("PA_gorse_broom", "mean")).loc[order]
t = te.set_index("set").loc[order]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); x = np.arange(4); w = .38
ax[0].bar(x - w/2, g.OA*100, w, yerr=g.sd*100, color="#5b7a99", capsize=3, label="Spatial CV (main 171, 20 repeats)")
ax[0].bar(x + w/2, t.OA*100, w, color="#b9c7d6", label="Independent test (expansion 88)")
for i in range(4):
    ax[0].text(i - w/2, g.OA.iloc[i]*100 + 2.2, f"{g.OA.iloc[i]*100:.1f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, t.OA.iloc[i]*100 + 1, f"{t.OA.iloc[i]*100:.1f}", ha="center", fontsize=8)
ax[0].set_ylim(40, 75); ax[0].set_ylabel("Overall accuracy (%)"); ax[0].set_title("(a) Overall accuracy", fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc="upper left")
ax[1].bar(x - w/2, g.G*100, w, color="#c0392b", alpha=.85, label="Spatial CV")
ax[1].bar(x + w/2, t["PA_gorse_broom"]*100, w, color="#e6a39b", label="Independent test")
for i in range(4):
    ax[1].text(i - w/2, g.G.iloc[i]*100 + 1, f"{g.G.iloc[i]*100:.0f}", ha="center", fontsize=8)
    ax[1].text(i + w/2, t["PA_gorse_broom"].iloc[i]*100 + 1, f"{t['PA_gorse_broom'].iloc[i]*100:.0f}", ha="center", fontsize=8)
ax[1].set_ylim(0, 60); ax[1].set_ylabel("Gorse producer's accuracy (%)"); ax[1].set_title("(b) Gorse, the hardest class", fontsize=10)
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=.3, axis="y"); a.spines[["top", "right"]].set_visible(False)
    a.axvspan(1.5, 2.5, color="#fff1a8", alpha=.5, zorder=0)
fig.tight_layout(); fig.savefig("Fig5_2_feature_sets.png", dpi=220)
UP = "./R5_shap/"
A, B = plt.imread(UP + "Fig5_6_shap_S_Sp_bar.png"), plt.imread(UP + "Fig5_6_shap_S_Sp_gorse_broom.png")
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [A.shape[1]/A.shape[0], B.shape[1]/B.shape[0]]})
for a, im, ttl in [(ax[0], A, "(a) Mean |SHAP| by class"), (ax[1], B, "(b) Gorse: direction of each variable")]:
    a.imshow(im); a.axis("off"); a.set_title(ttl, fontsize=11)
fig.tight_layout(); fig.savefig("Fig5_4_shap_SSp.png", dpi=170)
