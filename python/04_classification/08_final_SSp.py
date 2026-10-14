"""
Section 5 final model: S+Sp (pre-fire spectral + spring 2016).
Spatial-block CV (5-fold x 20) on main 171; independent test on expansion 88
(also excluding OID 10049, a 'pine' point inside a 2016 clearfell block).
Outputs: confusion figure, class + confidence map (Hansen 2016 cleared-pine mask), area table.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

exec(open("00_common.py").read())          # -> load(), SETS, CLS, COL

def wilson(k, n, z=1.96):                   # 95% CI for a proportion
    p = k/n
    c = (p + z*z/(2*n))/(1 + z*z/n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return c-h, c+h

def acc(cm):                                # OA, producer's, user's
    return np.trace(cm)/cm.sum(), np.diag(cm)/cm.sum(1), np.diag(cm)/cm.sum(0)

d, A, names = load(); cols = SETS["S+Sp"]   # main model = spectral + spring
m = d[d.source == "main"].reset_index(drop=True)
e = d[d.source != "main"].reset_index(drop=True)
mask = np.load("./cleared_pine_mask.npy")   # Hansen 2016 clearfell blocks (from step 07)

# Spatial-block CV (5-fold x 20)
cm_cv = np.zeros((4, 4), int); oas = []
for rep in range(20):
    pred = np.empty(len(m), int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(m, m.y, m.block):
        pred[te] = RF(500, max_features="sqrt", random_state=rep, n_jobs=-1).fit(m.loc[tr, cols], m.y[tr]).predict(m.loc[te, cols])
    cm = confusion_matrix(m.y, pred, labels=range(4)); cm_cv += cm; oas.append(acc(cm)[0])

# Independent test (mean prob over 10 seeds), three subsets
P = np.array([RF(500, max_features="sqrt", random_state=s, n_jobs=-1).fit(m[cols], m.y).predict_proba(e[cols]) for s in range(10)]).mean(0)
e["pred"] = P.argmax(1); rows = []
for tag, sel in [("expansion 88", np.ones(len(e), bool)), ("excl. OID 10049 (n=87)", e.OID != 10049),
                 ("excl. 17 near + 10049 (n=70)", (e.dist_to_main_m >= 90) & (e.OID != 10049))]:
    s = e[sel]; cm = confusion_matrix(s.y, s.pred, labels=range(4)); oa, pa, ua = acc(cm); lo, hi = wilson(np.trace(cm), len(s))
    rows.append([tag, len(s), oa, lo, hi, *pa, *ua])
    if tag == "expansion 88": cm_te = cm
T = pd.DataFrame(rows, columns=["test set","n","OA","lo95","hi95"]+[f"PA_{k}" for k in CLS]+[f"UA_{k}" for k in CLS]).round(3)
oa, pa, ua = acc(cm_cv)
cvt = pd.DataFrame({"PA": pa, "UA": ua}, index=CLS).round(3)
T.to_csv("Tab5_14_final_SSp_test.csv", index=False); cvt.to_csv("Tab5_15_final_SSp_cv_PA_UA.csv")
for tag, cm in [("cv_pooled20", cm_cv), ("test88", cm_te)]:
    pd.DataFrame(cm, index=[f"ref_{k}" for k in CLS], columns=[f"map_{k}" for k in CLS]).to_csv(f"Tab5_16_confusion_SSp_{tag}.csv")
print(f"CV OA {np.mean(oas):.3f} ± {np.std(oas):.3f}\n", cvt, "\n", T.to_string())

# Confusion figure (row-normalised = producer's view)
short = ["pasture", "gorse", "broadleaf", "pine"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
for a, cm, ttl in [(ax[0], cm_cv, f"(a) Spatial-block CV, main 171 (20 repeats)\nOA = {np.mean(oas)*100:.1f}%"),
                   (ax[1], cm_te, f"(b) Independent test, expansion 88\nOA = {T.OA[0]*100:.1f}% (95% CI {T.lo95[0]*100:.0f}–{T.hi95[0]*100:.0f}%)")]:
    Rn = cm / cm.sum(1, keepdims=True); a.imshow(Rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): a.text(j, i, f"{Rn[i,j]*100:.0f}%\n({cm[i,j]})", ha="center", va="center", fontsize=8, color="white" if Rn[i,j] > .5 else "black")
    a.set_xticks(range(4)); a.set_xticklabels(short); a.set_yticks(range(4)); a.set_yticklabels(short)
    a.set_xlabel("Classified"); a.set_ylabel("Reference"); a.set_title(ttl, fontsize=9)
fig.tight_layout(); fig.savefig("Fig5_7_confusion_SSp.png", dpi=300)

# Wall-to-wall map: class + confidence; cleared-pine cells set to class 4
final = RF(500, max_features="sqrt", random_state=0, n_jobs=-1).fit(m[cols], m.y)
Xa = A[..., [names.index(c) for c in cols]]
aoi = A[..., names.index("inside_aoi")] >= 0.5
ok = aoi & np.isfinite(Xa).all(-1)
Pr = final.predict_proba(Xa[ok])
cls = np.full(ok.shape, np.nan); cls[ok] = Pr.argmax(1)
conf = np.full(ok.shape, np.nan); conf[ok] = Pr.max(1)
cls[ok & mask] = 4; conf[ok & mask] = np.nan        # cleared pine -> separate class, no confidence
lab = CLS + ["cleared_pine (mask)"]; v = cls[ok]
at = pd.DataFrame({"ha": [(v == i).sum()*0.09 for i in range(5)]}, index=lab)
at["pct"] = (at.ha/at.ha.sum()*100).round(1); at = at.round(2)
at.to_csv("Tab5_17_map_area_SSp_masked.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cls, cmap=ListedColormap([COL[k] for k in CLS] + ["#9e9e9e"]), vmin=0, vmax=4, interpolation="none")
for k, c in zip(lab, [COL[k] for k in CLS] + ["#9e9e9e"]): ax[0].plot([], [], "s", c=c, label=k)
ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_title("(a) Pre-fire fuel classes, RF (spectral + spring)", fontsize=10)
im = ax[1].imshow(conf, cmap="viridis", vmin=.25, vmax=1, interpolation="none"); plt.colorbar(im, ax=ax[1], shrink=.7, label="Max class probability")
ax[1].set_title("(b) Classification confidence", fontsize=10)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("Fig5_8_class_map_SSp_masked.png", dpi=250)
print(at, "\nshare conf<0.5:", round((conf[ok & ~mask] < .5).mean(), 3))
print("cleared cells were classified as:", pd.Series(Pr.argmax(1)[mask[ok]]).map(dict(enumerate(CLS))).value_counts().to_dict())
