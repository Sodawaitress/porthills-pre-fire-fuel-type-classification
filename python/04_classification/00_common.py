"""
Shared loader for Section 5.
load() returns:
  d     - reference cells (use_A3) with every feature sampled at each cell,
          a 300 m spatial-block id (`block`) and the integer class `y`
  A     - the full feature stack (2017 + 2016 spring), rows x cols x bands
  names - band names for A
Also defines the class list CLS, class colours COL, and the four predictor sets SETS.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from A3_R4_regression_io import read_stack
DATA = Path(__file__).resolve().parents[2] / "data"

CLS = ["pasture", "gorse_broom", "broadleaf_scrub", "exotic_pine"]
COL = {"pasture": "#d4a017", "gorse_broom": "#c0392b", "broadleaf_scrub": "#27ae60", "exotic_pine": "#1f4e79"}
SPEC = ["pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7","NDVI","NDMI","BSI","NBR_pre"]   # pre-fire spectral
TERR = ["elev","slope","northness"]            # terrain (no grid_row/grid_col, Meyer et al. 2019)
SPR  = ["spr_B2","spr_B3","spr_B4","spr_B5","spr_B6","spr_B7","NDVI_spr","YI_spr","dNDVI_spr_sum"]   # spring 2016
SETS = {"S": SPEC, "S+T": SPEC+TERR, "S+Sp": SPEC+SPR, "S+T+Sp": SPEC+TERR+SPR}

def load():
    S, n = read_stack(str(DATA / "PortHills2017_stack.tif")); P, pn = read_stack(str(DATA / "PortHills2016_spring.tif"))
    A = np.concatenate([S, P], -1); names = n + pn
    df = pd.read_csv(DATA / "A3_reference_cells.csv"); d = df[df.use_A3].copy()
    r, c = d.array_row.values, d.array_col.values
    for i, b in enumerate(names): d[b] = A[r, c, i]                     # sample every band at each cell
    d["block"] = (r // 10).astype(str) + "_" + (c // 10).astype(str)   # 10 cells = 300 m blocks
    d["y"] = d.FuelClass.map({k: i for i, k in enumerate(CLS)})
    return d, A, names
