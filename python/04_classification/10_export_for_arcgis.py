"""
Section 5: export the final fuel-class map (RF spectral + spring, cleared-pine masked)
and the confidence layer as Esri ASCII grids + .prj (NZTM2000, EPSG:2193) for ArcGIS Pro.
No GDAL needed.
"""
import numpy as np, os
from sklearn.ensemble import RandomForestClassifier as RF
exec(open("00_common.py").read())
d, A, names = load(); cols = SETS["S+Sp"]; m = d[d.source == "main"].reset_index(drop=True)
mask = np.load("./cleared_pine_mask.npy")
final = RF(500, max_features="sqrt", random_state=0, n_jobs=-1).fit(m[cols], m.y)
X = A[..., [names.index(c) for c in cols]]; aoi = A[..., names.index("inside_aoi")] >= 0.5; ok = aoi & np.isfinite(X).all(-1)
Pr = final.predict_proba(X[ok])
cls = np.full(ok.shape, -9999, int); cls[ok] = Pr.argmax(1) + 1; cls[ok & mask] = 5
conf = np.full(ok.shape, -9999.0); conf[ok] = Pr.max(1); conf[ok & mask] = -9999
OUT = "./R5_arcgis/"; os.makedirs(OUT, exist_ok=True)
H, W = cls.shape; hdr = f"ncols {W}\nnrows {H}\nxllcorner 1565580\nyllcorner {5174310 - 30*H}\ncellsize 30\nNODATA_value -9999\n"
PRJ = 'PROJCS["NZGD_2000_New_Zealand_Transverse_Mercator",GEOGCS["GCS_NZGD_2000",DATUM["D_NZGD_2000",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1600000.0],PARAMETER["False_Northing",10000000.0],PARAMETER["Central_Meridian",173.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
for name, arr, fmt in [("fuel_class_SSp_masked", cls, "%d"), ("fuel_class_confidence", conf, "%.3f")]:
    with open(OUT + name + ".asc", "w") as f: f.write(hdr); np.savetxt(f, arr, fmt=fmt)
    open(OUT + name + ".prj", "w").write(PRJ)
lut = [(1, "Pasture", "#d4a017"), (2, "Gorse / broom", "#c0392b"), (3, "Broadleaved scrub", "#27ae60"), (4, "Exotic pine", "#1f4e79"), (5, "Cleared pine 2016 (masked)", "#9e9e9e")]
with open(OUT + "fuel_class_legend.csv", "w") as f:
    f.write("value,class,hex,ha,pct\n")
    for v, n, h in lut:
        c = (cls == v).sum(); f.write(f"{v},{n},{h},{c*0.09:.1f},{c/ok.sum()*100:.1f}\n")
print(open(OUT + "fuel_class_legend.csv").read())
