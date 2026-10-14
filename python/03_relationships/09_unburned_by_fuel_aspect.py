#==========================
# KT4 | Table 6, unburned share by fuel class and aspect
#==========================
#the chi-square only used 251 reference cells, this looks at every cell in the fire
#same as Tabulate Area in ArcGIS (zone = fuel or fuel x aspect, class = severity)
#I checked the fuel x aspect areas against my ArcGIS table on 13 Oct, identical
#no p-values: neighbouring cells are not independent,
#and the fuel map is only about 59% right on the test cells
#run from this folder: python3 09_unburned_by_fuel_aspect.py
#inputs: the fuel map from 04_classification/10_export_for_arcgis.py,
#the severity map from 05_export_lcdb_severity.py, northness from the GEE stack
#output: tables/unburned_by_fuel.csv, tables/unburned_by_fuel_aspect.csv
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
read = lambda f, band=1: rasterio.open(f).read(band)
fuel = read(ROOT / "python/04_classification/R5_arcgis/fuel_class_SSp_masked.asc")
sev = read("outputs/lcdb_maps/PortHills2017_severity_LCDB.tif")
with rasterio.open(ROOT / "data/PortHills2017_stack.tif") as s:
    nn = s.read(list(s.descriptions).index("northness") + 1)

FUEL = {1: "Pasture", 2: "Gorse / broom", 3: "Broadleaf scrub", 4: "Exotic pine", 5: "Cleared pine 2016"}
SEV = {0: "Unburned", 1: "Low", 2: "Moderate", 3: "High", 255: "No post-fire data"}
#northness = cos(aspect): <= -0.5 south-facing, > 0.5 north-facing (within 60 degrees of north), same cut as in ArcGIS
ASPECT = np.select([nn <= -0.5, nn <= 0.5], ["South", "East/West"], "North")

ok = (fuel > 0) & (nn != -9999)
df = pd.DataFrame({"fuel": pd.Series(fuel[ok]).map(FUEL), "aspect": ASPECT[ok],
                   "severity": pd.Series(sev[ok]).map(SEV)})


def summarise(keys):
    ha = df.groupby(keys + ["severity"]).size().unstack(fill_value=0).mul(0.09)   # 30 m cell = 0.09 ha
    ha = ha.reindex(columns=list(SEV.values()), fill_value=0)
    with_data = ha.drop(columns="No post-fire data").sum(axis=1)
    ha["Total ha"] = ha.sum(axis=1)
    ha["Unburned % (of cells with data)"] = 100 * ha["Unburned"] / with_data
    return ha.round(1)


Path("tables").mkdir(exist_ok=True)
by_fuel = summarise(["fuel"]).reindex(FUEL.values())
by_fuel_aspect = summarise(["fuel", "aspect"])
by_fuel.to_csv("tables/unburned_by_fuel.csv")
by_fuel_aspect.to_csv("tables/unburned_by_fuel_aspect.csv")

pd.set_option("display.width", 200)
print(by_fuel, "\n")
print(by_fuel_aspect)
all_data = df.severity != "No post-fire data"
print(f"\nWhole fire: {100 * (df.severity == 'Unburned').sum() / all_data.sum():.1f}% unburned "
      f"(cells with post-fire data)")
