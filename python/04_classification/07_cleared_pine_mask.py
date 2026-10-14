"""
Section 5: rasterise the cleared-pine blocks (Hansen lossyear == 2016, >= 6 px) onto the
30 m NZTM grid, so the final map can mask them out. Pure-Python NZTM2000 projection
(no pyproj here), checked against the NZTM columns in the reference CSV.
"""
import json, numpy as np, pandas as pd
from matplotlib.path import Path
from pathlib import Path as FilePath
DATA = FilePath(__file__).resolve().parents[2] / "data"

def nztm(lon, lat):
    """lon/lat (deg) -> NZTM2000 easting/northing (m). GRS80, TM, lon0 173, k0 0.9996."""
    a, f = 6378137.0, 1/298.257222101; k0, lon0, FE, FN = 0.9996, np.radians(173), 1600000.0, 10000000.0
    n = f/(2-f)
    lat, lon = np.radians(lat), np.radians(lon)
    A = a/(1+n)*(1 + n**2/4 + n**4/64)
    al = [n/2 - 2/3*n**2 + 5/16*n**3, 13/48*n**2 - 3/5*n**3, 61/240*n**3]
    t = np.sinh(np.arctanh(np.sin(lat)) - 2*np.sqrt(n)/(1+n)*np.arctanh(2*np.sqrt(n)/(1+n)*np.sin(lat)))
    xi = np.arctan2(t, np.cos(lon-lon0)); eta = np.arctanh(np.sin(lon-lon0)/np.sqrt(1+t**2))
    x = eta + sum(al[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(3))
    y = xi + sum(al[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(3))
    return FE + k0*A*x, FN + k0*A*y

# Check the projection reproduces the CSV's NZTM columns
df = pd.read_csv(DATA / "A3_reference_cells.csv")
X, Y = nztm(df.lon.values, df.lat.values)
print("projection check vs CSV NZTM (m): max |dx|=%.2f |dy|=%.2f" % (np.abs(X-df.NZTM_X).max(), np.abs(Y-df.NZTM_Y).max()))

# 30 m grid cell centres
g = json.load(open(DATA / "cleared_pine_blocks.geojson"))
H, W, X0, Y0 = 205, 200, 1565580, 5174310
cx = X0 + 30*(np.arange(W)+.5); cy = Y0 - 30*(np.arange(H)+.5); CX, CY = np.meshgrid(cx, cy)
pts = np.c_[CX.ravel(), CY.ravel()]; mask = np.zeros(H*W, bool); poly_ha = 0

# A cell is "cleared pine" if its centre falls inside a block polygon (minus any holes)
for ft in g["features"]:
    for ring_set in ([ft["geometry"]["coordinates"]] if ft["geometry"]["type"] == "Polygon" else ft["geometry"]["coordinates"]):
        ring = np.array(ring_set[0]); px, py = nztm(ring[:, 0], ring[:, 1])
        poly_ha += abs(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1)))/2/1e4
        inside = Path(np.c_[px, py]).contains_points(pts)
        for hole in ring_set[1:]:
            hx, hy = nztm(np.array(hole)[:, 0], np.array(hole)[:, 1]); inside &= ~Path(np.c_[hx, hy]).contains_points(pts)
        mask |= inside
mask = mask.reshape(H, W); np.save("./cleared_pine_mask.npy", mask)
print(f"blocks: {len(g['features'])} | polygon area {poly_ha:.2f} ha | rasterised {mask.sum()} cells = {mask.sum()*0.09:.2f} ha")

# Which reference points fall inside the blocks?
d = df[df.use_A3].copy(); d["in_cleared"] = mask[d.array_row, d.array_col]
print("\nreference cells inside cleared_pine blocks:\n", pd.crosstab(d.FuelClass, [d.source, d.in_cleared]))
d[d.in_cleared][["OID", "FuelClass", "source", "split", "array_row", "array_col"]].to_csv("Tab5_13_refs_in_cleared_pine.csv", index=False)
