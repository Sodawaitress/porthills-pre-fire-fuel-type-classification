from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"

with rasterio.open(DATA / "PortHills2017_stack.tif") as src:
    names = list(src.descriptions)
    shape = (src.height, src.width)
    crs = src.crs
    transform = src.transform

    required = ["dNBR", "dnbr_data", "NDVI", "inside_aoi"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing required bands in the main stack: {missing}")

    bands = {
        name: src.read(names.index(name) + 1, masked=True)
        .astype("float32").filled(np.nan)
        for name in required
    }

dnbr_lc = np.load(HERE / "dNBR_lc.npy", allow_pickle=False)
threshold = np.load(HERE / "T_lc.npy", allow_pickle=False)

if dnbr_lc.shape != shape:
    raise ValueError(
        f"Grid shape mismatch: LCDB={dnbr_lc.shape}, main stack={shape}"
    )

if threshold.shape != () or not np.isfinite(threshold).item():
    raise ValueError("T_lc.npy must contain a single finite scalar threshold")

T = float(threshold)

if crs != rasterio.crs.CRS.from_epsg(2193):
    raise ValueError(f"Expected EPSG:2193, but found: {crs}")

if not np.allclose(
    [transform.a, transform.b, transform.d, transform.e],
    [30, 0, 0, -30],
):
    raise ValueError("Expected a north-up grid with 30 m pixels")

if not np.array_equal(
    np.isfinite(dnbr_lc), np.isfinite(bands["dNBR"])
):
    raise ValueError("Finite-pixel locations differ between the LCDB response and the main-stack dNBR")

print("Input checks passed")
print(f"Grid: {shape[0]} rows x {shape[1]} columns")
print(f"CRS: {crs}")
print(f"LCDB threshold: {T:.8f}")
print(f"Finite LCDB pixels: {np.isfinite(dnbr_lc).sum():,}")
print("No output files were written.")

#same validity and pre-fire vegetation rules as the original
valid_response = np.isfinite(dnbr_lc)
valid_dates = bands["dnbr_data"] == 1
vegetated = np.isfinite(bands["NDVI"]) & (bands["NDVI"] > 0.25)

base_mask = valid_response & valid_dates & vegetated

#compare the two boundary rules before picking one
aoi = bands["inside_aoi"]
boundary_rules = {
    "AOI >= 0.5": np.isfinite(aoi) & (aoi >= 0.5),
    "AOI == 1.0": np.isfinite(aoi) & (aoi == 1.0),
}

print("\nMask checks")
print(f"Finite LCDB response: {valid_response.sum():,}")
print(f"After date-validity mask: {(valid_response & valid_dates).sum():,}")
print(f"After vegetation mask: {base_mask.sum():,}")

for label, boundary_mask in boundary_rules.items():
    selected = base_mask & boundary_mask
    unburned = selected & (dnbr_lc <= T)
    burned = selected & (dnbr_lc > T)

    assert unburned.sum() + burned.sum() == selected.sum()

    print(
        f"{label}: valid={selected.sum():,}, "
        f"unburned={unburned.sum():,}, "
        f"burned={burned.sum():,}"
    )

print("Diagnostic only: no output files written.")

#same AOI rule as the fuel map
map_mask = base_mask & boundary_rules["AOI >= 0.5"]

#keep 270 and 660 for low/moderate/high (Key & Benson 2006)
if not T < 270 < 660:
    raise ValueError("Severity thresholds must be strictly increasing")

if not map_mask.any():
    raise ValueError("The final map mask contains no valid pixels")

#0 unburned, 1 low, 2 moderate, 3 high, 255 no data
severity = np.full(shape, 255, dtype=np.uint8)
severity[map_mask] = 0
severity[map_mask & (dnbr_lc > T)] = 1
severity[map_mask & (dnbr_lc > 270)] = 2
severity[map_mask & (dnbr_lc > 660)] = 3

#write to its own file, never over the old one by accident
output_dir = HERE / "outputs" / "lcdb_maps"
output_path = output_dir / "PortHills2017_severity_LCDB.tif"

if output_path.exists():
    #re-running doubles as a check: same result -> keep the old file, different -> stop
    with rasterio.open(output_path) as old:
        if np.array_equal(old.read(1), severity):
            print(f"Saved map is identical to this run; kept: {output_path}")
            raise SystemExit(0)
    raise FileExistsError(
        f"Output already exists and DIFFERS from this run; review it before replacing: {output_path}"
    )

output_dir.mkdir(parents=True, exist_ok=True)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=shape[0],
    width=shape[1],
    count=1,
    dtype="uint8",
    crs=crs,
    transform=transform,
    nodata=255,
    compress="deflate",
) as dst:
    dst.write(severity, 1)
    dst.set_band_description(1, "LCDB-corrected dNBR severity")
    dst.update_tags(
        response="dNBR_lc.npy",
        threshold_source="T_lc.npy",
        unburned_threshold=str(T),
        moderate_boundary="270",
        high_boundary="660",
        mask_rule="finite response AND dnbr_data==1 AND NDVI>0.25 AND inside_aoi>=0.5",
        class_codes="0=candidate_unburned;1=low;2=moderate;3=high;255=nodata",
        official_islands="included where valid",
        tree_mask="not applied",
    )

#read it back to make sure values and georeferencing survived
with rasterio.open(output_path) as check:
    assert check.crs == crs
    assert check.transform == transform
    assert check.nodata == 255
    assert np.array_equal(check.read(1), severity)

print("\nLCDB raster export passed")
for code, label in enumerate(
    ["Candidate unburned", "Low", "Moderate", "High"]
):
    print(f"{code} - {label}: {(severity == code).sum():,} pixels")

print(f"NoData: {(severity == 255).sum():,} pixels")
print(f"Saved: {output_path}")
