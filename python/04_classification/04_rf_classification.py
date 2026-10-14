"""
Section 5 steps 3-4: random forest fuel-type classification.
Spatial-block CV (5-fold x 20) + original split + independent test, 4 predictor sets.
Accuracy = OA + per-class PA/UA; no Kappa (Foody 2020).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

exec(open("00_common.py").read())          # -> load(), SETS, CLS

def wilson(k, n, z=1.96):                   # 95% CI for a proportion (stable at small n)
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h

def rf(seed):                               # 500 trees, sqrt(features) per split
    return RandomForestClassifier(500, max_features="sqrt", random_state=seed, n_jobs=-1)

def acc(cm):                                # OA, producer's, user's
    return np.trace(cm)/cm.sum(), np.diag(cm)/cm.sum(1), np.diag(cm)/cm.sum(0)

d, A, names = load()
main = d[d.source == "main"].reset_index(drop=True)      # 171 cells for training/CV
exp  = d[d.source != "main"].reset_index(drop=True)      # 88 held out as independent test

# 1. Spatial-block CV: same 300 m block -> same fold (no spatial leakage); 20 repeats averaged
cv_rows, cv_cm = [], {}
for name, cols in SETS.items():
    cm_tot = np.zeros((4, 4), int)
    for rep in range(20):
        pred = np.empty(len(main), int)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(main, main.y, main.block):
            pred[te] = rf(rep).fit(main.loc[tr, cols], main.y[tr]).predict(main.loc[te, cols])
        cm = confusion_matrix(main.y, pred, labels=range(4)); cm_tot += cm
        oa, pa, ua = acc(cm); cv_rows.append([name, rep, oa, *pa, *ua])
    cv_cm[name] = cm_tot

cv = pd.DataFrame(cv_rows, columns=["set","rep","OA"]+[f"PA_{k}" for k in CLS]+[f"UA_{k}" for k in CLS])
cv.to_csv("Tab5_6_cv_by_repeat.csv", index=False)
summary = cv.groupby("set", sort=False).agg(["mean","std"]).drop(columns="rep").round(3)
summary.to_csv("Tab5_7_cv_summary.csv")
base = cv[cv.set=="S"].OA.values
gain = {k: cv[cv.set==k].OA.values - base for k in SETS}
print("== Spatial-block CV (main 171, 5-fold x 20) ==")
print(summary[[("OA","mean"),("OA","std")]+[(f"PA_{k}","mean") for k in CLS]].to_string())
print("OA gain vs S (mean, share>0):", {k:(round(v.mean(),3), round((v>0).mean(),2)) for k,v in gain.items()})

# 2. Original 300 m train->valid split, averaged over 10 RF seeds
orig = []
for name, cols in SETS.items():
    tr, va = main[main.split=="train"], main[main.split=="valid"]
    oa = np.mean([rf(s).fit(tr[cols], tr.y).predict(va[cols]) == va.y for s in range(10)])
    orig.append([name, len(tr), len(va), oa])
print("\n== Original 300 m split (train -> valid, 10 seeds) ==")
print(pd.DataFrame(orig, columns=["set","n_tr","n_va","OA"]).round(3))

# 3. Independent test: train on 171 main, predict 88 expansion (mean prob over 10 seeds)
near = exp.dist_to_main_m < 90
rows, exp_cm = [], {}
for name, cols in SETS.items():
    proba = np.array([rf(s).fit(main[cols], main.y).predict_proba(exp[cols]) for s in range(10)]).mean(0)
    pred = proba.argmax(1)
    cm = confusion_matrix(exp.y, pred, labels=range(4)); exp_cm[name] = cm
    oa, pa, ua = acc(cm)
    lo, hi = wilson(int(np.trace(cm)), len(exp))
    oa_far = (pred[~near] == exp.y[~near]).mean()          # cells >=90 m from any training cell
    rows.append([name, oa, lo, hi, oa_far, *pa, *ua])
    if name == "S+T+Sp": exp["pred"] = pred
test = pd.DataFrame(rows, columns=["set","OA","OA_lo95","OA_hi95","OA_far(>=90m,n=71)"]+[f"PA_{k}" for k in CLS]+[f"UA_{k}" for k in CLS]).round(3)
test.to_csv("Tab5_8_expansion_test.csv", index=False)
print("\n== Independent test: expansion 88 ==")
print(test.iloc[:, :5].to_string()); print(test[["set"]+[f"PA_{k}" for k in CLS]].to_string())

for name in ["S","S+T+Sp"]:
    for tag, cm in [("cv", cv_cm[name]), ("test", exp_cm[name])]:
        pd.DataFrame(cm, index=[f"ref_{k}" for k in CLS], columns=[f"map_{k}" for k in CLS]).to_csv(f"Tab5_9_confusion_{tag}_{name.replace('+','_')}.csv")
print("\nCV confusion (pooled 20 reps), S+T+Sp:\n", pd.DataFrame(cv_cm["S+T+Sp"], index=CLS, columns=CLS))
print("Test confusion, S+T+Sp:\n", pd.DataFrame(exp_cm["S+T+Sp"], index=CLS, columns=CLS))
print("Test confusion, S:\n", pd.DataFrame(exp_cm["S"], index=CLS, columns=CLS))
