#==========================
# KT4 | Table 1, reference cells per fuel class
#==========================
#main = cells for spatial-block CV and for choosing the predictor set
#expansion = held-out test cells, never used to choose anything
#only the four modelled classes are counted (use_A3)
#the classes I left out are listed below with the reason
#run from this folder: python3 05_reference_counts.py
#output: tables/reference_counts.csv
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
CLS = ["pasture", "gorse_broom", "broadleaf_scrub", "exotic_pine"]

d = pd.read_csv(DATA / "A3_reference_cells.csv")
used = d[d.use_A3]
tab = pd.crosstab(used.FuelClass, used.source).reindex(CLS)[["main", "expansion"]]
tab["total"] = tab.sum(axis=1)
tab.loc["total"] = tab.sum()

excluded = d[~d.use_A3].groupby(["FuelClass", "role"]).size()
print(tab.to_string())
print("\nexcluded (not modelled):\n" + excluded.to_string())

Path("tables").mkdir(exist_ok=True)
tab.to_csv("tables/reference_counts.csv")
print("\nsaved tables/reference_counts.csv")
