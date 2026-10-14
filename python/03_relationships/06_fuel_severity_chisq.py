#==========================
# KT4 | does burn severity depend on fuel class? (chi-square, Figure 5)
#==========================
#field variable: fuel class of each reference cell (main + expansion)
#image variable: LCDB-corrected severity class at the same 30 m cell, from 05_export_lcdb_severity.py
#  0 unburned (<= T), 1 low (T-270), 2 moderate (270-660), 3 high (> 660)
#  270 and 660 from Key & Benson 2006, T from the mapped unburned islands
#chi-square test of independence, Cramer's V for how strong it is
#cells with no post-fire data are dropped and counted
#run from this folder: python3 06_fuel_severity_chisq.py
#output: tables/fuel_severity_crosstab.csv, tables/fuel_severity_chisq.csv
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
CLS = ["pasture", "gorse_broom", "broadleaf_scrub", "exotic_pine"]
SEV = {0: "Unburned", 1: "Low", 2: "Moderate", 3: "High"}

d = pd.read_csv(DATA / "A3_reference_cells.csv")
d = d[d.use_A3].copy()
with rasterio.open(HERE / "outputs" / "lcdb_maps" / "PortHills2017_severity_LCDB.tif") as src:
    sev = src.read(1)
    # reference cells carry their row/col on the same 30 m grid as the stack
    with rasterio.open(DATA / "PortHills2017_stack.tif") as stack:
        assert src.transform == stack.transform and src.shape == stack.shape, "severity map not on the stack grid"
d["severity"] = sev[d.array_row.values, d.array_col.values]
dropped = int((d.severity == 255).sum())
d = d[d.severity != 255]

tab = pd.crosstab(d.FuelClass, d.severity).reindex(index=CLS, columns=list(SEV), fill_value=0)
tab.columns = [SEV[c] for c in tab.columns]
chi2, p, dof, expected = chi2_contingency(tab.values)
n = int(tab.values.sum())
V = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
low_exp = int((expected < 5).sum())

print(tab.to_string())
print(f"\nn = {n} reference cells ({dropped} dropped: outside the severity map)")
print(f"chi-square = {chi2:.2f}, df = {dof}, p = {p:.2g}, Cramer's V = {V:.2f}")
print(f"expected counts < 5: {low_exp} of {expected.size} cells "
      "(chi-square is approximate if more than 20% of cells are below 5)")

Path("tables").mkdir(exist_ok=True)
tab.to_csv("tables/fuel_severity_crosstab.csv")
pd.DataFrame([dict(n=n, dropped=dropped, chi2=chi2, df=dof, p=p, cramers_v=V,
                   expected_below_5=low_exp, table_cells=expected.size)]).round(4) \
    .to_csv("tables/fuel_severity_chisq.csv", index=False)
print("saved tables/fuel_severity_crosstab.csv, tables/fuel_severity_chisq.csv")
