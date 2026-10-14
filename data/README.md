# data

All rasters are on the same NZTM2000 (EPSG:2193) 30 m grid, 205 rows x 200 columns,
upper-left corner (1565580, 5174310). Hashes are in `input_manifest.json`.

| File | What it is |
|---|---|
| `PortHills2017_stack.tif` (32 bands) | pre- and post-fire Landsat 8, indices, terrain, masks |
| `PortHills2016_spring.tif` (9 bands) | spring 2016 composite (Sep-Nov), for gorse flowering |
| `PortHills2017_ring_LCDB.tif` (11 bands) | raw control-ring inputs over a 2,500 m buffer, for the LCDB offsets |
| `lcdb/LCDB2012_PortHills_2500m.*` | LCDB v5.0 polygons (2012 time step) covering the control ring |
| `A3_reference_cells.csv` | the 259 reference cells (main 171, expansion 88) and the excluded ones |
| `cleared_pine_blocks.geojson` | pine blocks harvested in 2016, masked on the final map |
| `shapefiles_training_points/` | reference points and the CDEM fire perimeter as shapefiles |

the three rasters come from `gee/00_gee_PortHills2017_stack.js`

## main stack bands

| Bands | Names |
|---|---|
| 1-6 | `pre_B2` ... `pre_B7` pre-fire reflectance |
| 7-12 | `post_B2` ... `post_B7` post-fire reflectance (not used to predict fuel) |
| 13-18 | `NBR_pre`, `NBR_post`, `dNBR`, `NDVI`, `NDMI`, `BSI` |
| 19-20 | `severity`, `dnbr_data` |
| 21-23 | `pre_source`, `grid_row`, `grid_col` |
| 24-26 | `elev`, `slope`, `northness` |
| 27 | `ephemeral_refugia` |
| 28-31 | `pre_analysis_valid`, `inside_aoi`, `clean_pre_mask`, `post_neg_refl` |
| 32 | `dNBR_cs` |

`severity`, `ephemeral_refugia` and `dNBR_cs` are an older Dynamic World correction.
the report uses the LCDB correction instead, built by
`python/03_relationships/00_build_regression_inputs.py` (`dNBR_lc.npy`, `T_lc.npy`)

## spring stack bands

`spr_B2` ... `spr_B7`, `NDVI_spr`, `YI_spr` = (Red-Green)/(Red+Green),
`dNDVI_spr_sum` = NDVI_spr - NDVI_pre
