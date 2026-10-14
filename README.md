# Port Hills Pre-Fire Fuel-Type Classification

Pre-fire fuel-type map of the 2017 Port Hills fire (Christchurch, New Zealand) from Landsat 8,
and how fuel type and aspect relate to burn severity and unburned patches.
ERST619 Kaupapa Tuhika 4 (Lincoln University, 2026). Projection throughout: **NZTM2000 / EPSG:2193**.

## Layout: one folder per platform

```
gee/          Google Earth Engine
  00_gee_PortHills2017_stack.js   run in the GEE Code Editor -> the three rasters in data/
  python/                         preprocessing check and Figure 1 through the GEE Python API
python/       local analysis (Python, plus two R scripts), one folder per report section
  02_exploration/                 Section 3.1 data exploration
  03_relationships/               Sections 3.2 / 4.1 burn severity, regression, chi-square, overlay table
  04_classification/              Sections 3.3 / 4.2 separability, random forest, SHAP, accuracy
notebook/     run_all.ipynb       runs every local script in report order (easiest)
data/         input rasters, reference cells, shapefiles (see data/README.md)
```

## How to run

1. **GEE (optional).** The three GEE exports are already in `data/` (5.5 MB), so you can skip this step.
   To rebuild them: upload `data/shapefiles_training_points/study_area_boundary.shp` (the CDEM fire extent)
   as an asset in your own Earth Engine project, edit `ASSET_ID` in `gee/00_gee_PortHills2017_stack.js`,
   run it in the Code Editor and export `PortHills2017_stack.tif` (32 bands), `PortHills2016_spring.tif` (9)
   and `PortHills2017_ring_LCDB.tif` (11) into `data/`. For the scripts in `gee/python/`, set your project
   first: `export EE_PROJECT=<your-project-id>`.
2. **Local analysis.** Open `notebook/run_all.ipynb` and run all cells. Each cell runs the original
   script from its own folder, so you can also run any script by hand:
   `cd python/03_relationships && python3 06_fuel_severity_chisq.py`.
   Needs Python 3.12 (`numpy pandas scipy scikit-learn matplotlib rasterio shap`) and
   R (`terra ggplot2 dplyr tidyr`).
3. **ArcGIS Pro.** Map 1 and Map 2 were laid out by hand in ArcGIS Pro from the fuel map that
   `python/04_classification/10_export_for_arcgis.py` writes to `R5_arcgis/` (ASCII grid + .prj, NZTM).

⚠️ Burn severity is the **LCDB-corrected** dNBR (`python/03_relationships/dNBR_lc.npy`, `T_lc.npy`).
The stack bands `dNBR_cs`, `severity` and `ephemeral_refugia` are an older Dynamic World version. Do not use them.

## Report item → script → output

| Report item | Script (run from its folder) | Output |
|---|---|---|
| 3.1a preprocessing numbers | `gee/python/01_check_preprocessing.py` | console |
| Figure 1 visual inspection | `gee/python/04_fig_visual_inspection.py` | `outputs/Fig_visual_inspection.png` |
| Table 1 reference cells | `python/02_exploration/05_reference_counts.py` | `tables/reference_counts.csv` |
| Reference-point shapefile (Section 7) | `python/02_exploration/06_export_reference_shapefile.py` | `data/shapefiles_training_points/` |
| Table 2 outliers; Figure 12 box plots | `python/02_exploration/02_outliers.R` | `tables/A3_outlier_types.csv`, `figures/` |
| Figure 2 histograms | `python/02_exploration/03_normality_transforms.R` | `figures/A3_histograms.png` |
| Figure 3 transformations | `python/02_exploration/07_fig_transformations.py` | `figures/A3_transformations.png` |
| Table 3 LCDB offsets, threshold T | `python/03_relationships/00_build_regression_inputs.py` | `tables/offset_LCDB.csv`, `dNBR_lc.npy`, `T_lc.npy` |
| Figure 4 predictor correlation | `python/03_relationships/07_predictor_correlation.py` | `figures/fig_predictor_correlation.png` |
| Figure 5 fuel × severity; χ² | `05_export_lcdb_severity.py` → `06_fuel_severity_chisq.py` → `08_fig_fuel_severity.py` | `tables/fuel_severity_*.csv`, `figures/fig_fuel_severity.png` |
| Figure 6 regression | `python/03_relationships/03_regression.py` → `04_figures.py` | `tables/reg_*.csv`, `figures/fig_regression.png` |
| Table 6 unburned share by fuel × aspect | `python/04_classification/10_export_for_arcgis.py` → `python/03_relationships/09_unburned_by_fuel_aspect.py` | `tables/unburned_by_fuel*.csv` |
| Table 4 JM; Figure 7 box plots | `python/04_classification/02_separability.py` | `Tab5_3_JM_distance.csv`, `Fig5_2_boxplots.png` |
| Table 5 predictor-set CV | `python/04_classification/04_rf_classification.py` | `Tab5_7_cv_summary.csv` |
| Figure 8 SHAP | `06_shap_local.py` → `09_report_figures.py` | `R5_shap/`, `Fig5_4_shap_SSp.png` |
| Figure 9 confidence; Figure 10 confusion; Table 7 PA/UA; areas | `07_cleared_pine_mask.py` → `08_final_SSp.py` → `11_fig_confidence.py` | `Fig5_8b_confidence.png`, `Fig5_7_confusion_SSp.png`, `Tab5_13`–`Tab5_17` |

## Data sources

Landsat 8 Collection 2 Level-2 (USGS EROS); LCDB v5.0, 2012 time step (Manaaki Whenua – Landcare Research);
SRTM 30 m terrain; Global Forest Change v1.11 (Hansen et al.); fire extent 16 Feb 2017 (Canterbury CDEM);
LINZ 2015–16 aerial photos. Full citations are in the report.

Yu Zhou · ERST619, Lincoln University, 2026.
