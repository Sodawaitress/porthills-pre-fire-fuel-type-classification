"""Build LCDB-corrected regression inputs, preserving the recovered LCDB workflow.

Run from this directory before 03_regression.py and 04_figures.py.
Uses the repository data/ main stack, raw ring export and buffered LCDB2012 shapefile.
Retains the original 3x3 sub-sample majority rasterisation, class selection,
pooled fallback, official-island p90 threshold, and stack dNBR validity mask.
Grid mapping is checked against raster metadata.
"""
import os, json, hashlib
from pathlib import Path as FilePath
import numpy as np, pandas as pd
import rasterio
from matplotlib.path import Path
from A3_R4_regression_io import read_stack
from shp import read_dbf, read_shp
DATA = FilePath(__file__).resolve().parents[2] / "data"
os.makedirs("tables", exist_ok=True)

b, nm = read_stack(str(DATA / "PortHills2017_ring_LCDB.tif")); b[b==-9999]=np.nan; R={k:b[:,:,i] for i,k in enumerate(nm)}
a, sn = read_stack(str(DATA / "PortHills2017_stack.tif")); a[a==-9999]=np.nan; S={k:a[:,:,i] for i,k in enumerate(sn)}
H,W = b.shape[:2]; gr=np.nanmedian(R["grid_row"],1); gc=np.nanmedian(R["grid_col"],0)

# Fail before writing outputs if CRS, resolution or grid coordinates disagree.
with rasterio.open(DATA / "PortHills2017_ring_LCDB.tif") as ring, rasterio.open(DATA / "PortHills2017_stack.tif") as stack:
    for src in (ring, stack):
        if src.crs != rasterio.crs.CRS.from_epsg(2193):
            raise ValueError("Both rasters must use EPSG:2193")
        if not np.allclose([src.transform.a, src.transform.b, src.transform.d, src.transform.e], [30, 0, 0, -30]):
            raise ValueError("Expected a north-up 30 m grid")
    for values, expected in (
        (gc, (ring.transform.c + (np.arange(ring.width) + .5)*30)/30),
        (gr, -(ring.transform.f - (np.arange(ring.height) + .5)*30)/30),
    ):
        if not np.allclose(values, expected, rtol=0, atol=.01):
            raise ValueError("Ring grid coordinate bands disagree with georeference")
    sr = -(stack.transform.f - (np.arange(stack.height) + .5)*30)/30
    sc = (stack.transform.c + (np.arange(stack.width) + .5)*30)/30
    grid_info = dict(crs=str(stack.crs), transform=list(stack.transform), height=stack.height, width=stack.width)
with open(DATA / "lcdb/LCDB2012_PortHills_2500m.prj") as f:
    if rasterio.crs.CRS.from_wkt(f.read()).to_epsg() != 2193:
        raise ValueError("LCDB polygons must use EPSG:2193")

# rasterise LCDB (3x3 sub-samples per 30 m cell, majority)
f, recs = read_dbf(str(DATA / "lcdb/LCDB2012_PortHills_2500m.dbf")); _,_,shapes = read_shp(str(DATA / "lcdb/LCDB2012_PortHills_2500m.shp"))
o=(np.arange(3)+.5)*10-15; xs=(gc[:,None]*30+o).ravel(); ys=(-gr[:,None]*30-o).ravel()
XX,YY=np.meshgrid(xs,ys); pts=np.column_stack([XX.ravel(),YY.ravel()]); sub=np.zeros(len(pts),int)
for r,sh in zip(recs,shapes):
    v=[];c=[]
    for ring in sh: v+=ring; c+=[Path.MOVETO]+[Path.LINETO]*(len(ring)-2)+[Path.CLOSEPOLY]
    x0,y0=np.min(v,0); x1,y1=np.max(v,0)
    cand=np.where((pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1))[0]
    sub[cand[Path(v,c).contains_points(pts[cand])]]=int(r["Class_2012"])
sub=sub.reshape(H,3,W,3).transpose(0,2,1,3).reshape(H,W,9)
lc=np.apply_along_axis(lambda x:np.bincount(x).argmax(),2,sub)
names={int(r["Class_2012"]):r["Name_2012"] for r in recs}

d=R["dNBR_raw"]; base=(R["ring_base"]==1)&np.isfinite(d)
VEG=[40,41,50,51,52,54,56,69,71]; ctl=base&np.isin(lc,VEG); pooled=d[ctl].mean()
OWN=[40,41,51,52,54,71]                                   # original fixed set of classes with >=500 control cells
if not np.isfinite(pooled) or any((base & (lc == k)).sum() < 500 for k in OWN):
    raise ValueError("Control support changed; review the original fixed class-offset set")
rows=[dict(class_id=k,name=names[k],control_cells=int((base&(lc==k)).sum()),mean=d[base&(lc==k)].mean(),
           median=np.median(d[base&(lc==k)]),sd=d[base&(lc==k)].std()) for k in OWN]
rows.append(dict(class_id=0,name="pooled vegetation (all other classes)",control_cells=int(ctl.sum()),mean=pooled,median=np.median(d[ctl]),sd=d[ctl].std()))
off=np.full(d.shape,pooled)
for r_ in rows[:-1]: off[lc==r_["class_id"]]=r_["mean"]
dl=d-off
valid=(R["dnbr_data"]==1)&(R["veg_mask"]==1)&np.isfinite(d)
island = (R["official_island"]==1)&valid
if not island.any():
    raise ValueError("No valid official-island cells for the threshold")
T=np.percentile(dl[island],90); comm=(dl[ctl]>T).mean()
tab=pd.DataFrame(rows); tab.loc[len(tab)]=dict(class_id=-1,name=f"threshold T = island p90; commission = {100*comm:.1f}%",mean=T)


# move to the stack grid
ri=np.searchsorted(gr,sr); ci=np.searchsorted(gc,sc)
if (np.any(ri >= H) or np.any(ci >= W)
        or not np.allclose(gr[ri], sr, rtol=0, atol=.01)
        or not np.allclose(gc[ci], sc, rtol=0, atol=.01)):
    raise ValueError("Stack grid is not aligned with and contained in the control-ring export")
g=lambda A:A[np.ix_(ri,ci)]
dNBR_lc=np.where(np.isfinite(S["dNBR"]), g(dl), np.nan)
if np.any(np.isfinite(S["dNBR"]) & ~np.isfinite(dNBR_lc)):
    raise ValueError("Ring raw dNBR is missing where the stack response is valid")
tab.round(2).to_csv("tables/offset_LCDB.csv",index=False)
np.save("dNBR_lc.npy",dNBR_lc.astype("float32")); np.save("lcdb_stackgrid.npy",g(lc)); np.save("T_lc.npy",np.array(T))
print(tab.round(1).to_string(index=False)); print("T =",round(T,1),"commission =",round(100*comm,1),"%")

# Record exact inputs and calculation outputs for reproducibility.
inputs = [DATA / "PortHills2017_stack.tif", DATA / "PortHills2017_ring_LCDB.tif",
          *sorted((DATA / "lcdb").glob("LCDB2012_PortHills_2500m.*")),
          *[FilePath(__file__).resolve().parent / name for name in
            ("00_build_regression_inputs.py", "shp.py", "A3_R4_regression_io.py")]]
provenance = dict(
    inputs_sha256={str(p.relative_to(DATA.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    grid=grid_info, rasterisation="original 3x3 sub-sample majority; lowest-code tie break",
    own_offset_classes=OWN, pooled_vegetation_classes=VEG,
    offsets={str(row["class_id"]): float(row["mean"]) for row in rows},
    threshold=float(T), commission=float(comm), island_cells=int(island.sum()),
    finite_response_cells=int(np.isfinite(dNBR_lc).sum()),
)
FilePath("tables/lcdb_input_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
