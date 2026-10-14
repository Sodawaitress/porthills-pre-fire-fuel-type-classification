#==========================
# KT4 | reference points as a shapefile (Section 7)
#==========================
#built from data/A3_reference_cells.csv, NZTM2000 (EPSG:2193)
#fields:
#  OID        point ID, expansion points are OID + 10000
#  FuelClass  class name, class_id is its number code
#  source     main (CV) or expansion (held-out test)
#  in_model   1 = used by the 4-class model, 0 = left out
#  excl_note  why a point was left out
#the old 70/30 split column is dropped on purpose, spatial-block CV replaced it
#run from this folder: python3 06_export_reference_shapefile.py
#output: ../../data/shapefiles_training_points/training_validation_points.shp
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "shapefiles_training_points" / "training_validation_points.shp"

d = pd.read_csv(DATA / "A3_reference_cells.csv")
note = pd.Series("", index=d.index)
note[d.FuelClass == "bare_rock"] = "bare rock: too few (7)"
note[d.FuelClass == "cleared_pine"] = "cleared pine: few, scattered (19)"
note[~d.use_A3 & d.FuelClass.isin(["pasture", "gorse_broom", "broadleaf_scrub", "exotic_pine"])] = "straddles fire perimeter"

g = gpd.GeoDataFrame(
    {"OID": d.OID, "FuelClass": d.FuelClass, "class_id": d.class_id, "source": d.source,
     "in_model": d.use_A3.astype(int), "excl_note": note},
    geometry=gpd.points_from_xy(d.NZTM_X, d.NZTM_Y), crs="EPSG:2193")
g.to_file(OUT, encoding="utf-8")
print(g.groupby(["source", "in_model"]).size().to_string())
print("saved", OUT)
