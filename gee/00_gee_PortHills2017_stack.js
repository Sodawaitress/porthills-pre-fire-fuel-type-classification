// ============================================================
// GEE: Port Hills 2017 feature-stack builder
// MAIN ENTRY: run this one script for the saved study input schemas.
// 01_build_stack.js / 02_phenology.js / 03_burn_severity.js / 04_land_change.js
// are optional generic variants, NOT further numbered steps required by this file.
// A1 base (dNBR, offset, severity, refugia, island validation) reused
// and EXTENDED for A3. [A3] blocks are marked below:
//   - 30 m NZTM analysis grid (grid_row / grid_col)
//   - spatial data-cleaning masks (pre_analysis_valid, inside_aoi,
//     clean_pre_mask)  -> Section 3 spatial cleaning
//   - NDMI moisture index (used in Section 4 regression)
//   - pre-fire reconstruction (Jan 23 + Jan 16) + source provenance
//   - 30 base bands + post_neg_refl + historical DW dNBR_cs (32 total)
//   - recovered 9-band spring 2016 export
//   - raw control-ring export for the CURRENT local LCDB correction
// Outputs: PortHills2017_stack.tif, PortHills2016_spring.tif,
//          PortHills2017_ring_LCDB.tif (all EPSG:2193) + island_validation.csv
// IMPORTANT: dNBR_cs below preserves the historical Dynamic World band.
// It is NOT the final LCDB response. Current regression/severity use
// python/03_relationships/00_build_regression_inputs.py -> dNBR_lc.npy / T_lc.npy.
// The LCDB input export at the end does not filter control cells with DW.
// ============================================================

//==========================
// 1. STUDY AREA (AOI)
//==========================
//to run it yourself: upload data/shapefiles_training_points/study_area_boundary.shp to your own project and change this line
var ASSET_ID = 'projects/genuine-hold-427410-f0/assets/Port_Hills_2017_Fire_Boundary';
var raw = ee.FeatureCollection(ASSET_ID);

var fireWithHoles = raw.filter(ee.Filter.eq('Type','Fire Boundary')).first().geometry();
var unburntFC     = raw.filter(ee.Filter.eq('Type','Unburnt'));
var aoi           = fireWithHoles.union(unburntFC.geometry(), 1);  //fill the two holes
var wideFrame     = aoi.buffer(4000);   //wide enough for the control ring
var mapFrame      = aoi.buffer(500);    //display extent only

var ha        = 10000;
var pixelArea = ee.Image.pixelArea();

function haIn(mask, geom){
  return pixelArea.updateMask(mask).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geom || aoi,
    scale: 30,
    maxPixels: 1e9,
    tileScale: 4
  }).getNumber('area').divide(ha);
}

Map.centerObject(aoi, 13);
Map.setOptions('SATELLITE');

//source file is Web Mercator; .area() is geodesic so it is correct
print('===== OFFICIAL MAP (ha) =====');
print('perimeter:',        aoi.area(1).divide(ha));
print('official unburnt:', unburntFC.geometry().area(1).divide(ha));
print('official burned:',  fireWithHoles.area(1).divide(ha));

//==========================
// 2. STUDY PERIOD AND OUTPUT SETTINGS
//==========================
var preStart  = '2016-12-01';   //widened back for cloud-free coverage
var preEnd    = '2017-02-11';   //fire ignited 13 Feb
var postStart = '2017-02-24';   //hotspots ran to about 25 Feb
var postEnd   = '2017-03-30';   //kept short; regrowth lifts NBR

var outputBands = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'];
//                  blue    green   red     NIR     SWIR1   SWIR2
var outputScale = 30;
var outputCRS   = 'EPSG:2193';  //NZ Transverse Mercator, for correct areas


//30 m NZTM grid, origin at 0,0, so cell edges fall on multiples of 30 m
var exportTransform = [outputScale, 0, 0, 0, -outputScale, 0];
var analysisProj    = ee.Projection(outputCRS, exportTransform);


//==========================
// 3. TERRAIN
//==========================
var dem       = ee.Image('USGS/SRTMGL1_003').rename('elev');
var slope     = ee.Terrain.slope(dem).rename('slope');
//northness, not aspect: 359 and 1 degrees are 358 apart as numbers
var northness = ee.Terrain.aspect(dem).multiply(Math.PI/180).cos().rename('northness');
var terrain   = dem.addBands(slope).addBands(northness);

var firePct = terrain.select(['elev','slope']).reduceRegion({
  reducer: ee.Reducer.percentile([2,98]),
  geometry: aoi, scale: outputScale, maxPixels: 1e9});

//==========================
// 4. LOAD LANDSAT AND MASK CLOUD
//==========================
//Collection 2 needs an offset as well as a scale factor
function scaleSR(image){
  return ee.Image(image.select('SR_B.').multiply(0.0000275).add(-0.2)
    .copyProperties(image, image.propertyNames()));  //ee.Image() wrapper is required
}

//QA_PIXEL bits: 3 = cloud, 4 = cloud shadow
//Historical note reported 242 ha of high-dNBR ground flagged as shadow.
//That area/interpretation is not verified by this script alone. Preserve the
//original rule below: remove cloud everywhere, shadow outside the fire only.
function maskL8clouds_PRE(image){
  var qa = image.select('QA_PIXEL');

  var bad = qa.bitwiseAnd(1 << 3).neq(0)   // cloud
    .or(qa.bitwiseAnd(1 << 4).neq(0));     // cloud shadow

  return scaleSR(image).updateMask(bad.not());
}

function maskL8clouds_QA(image){
  var qa = image.select('QA_PIXEL');
  var inFire = ee.Image.constant(1).clip(aoi).mask();
  var bad = qa.bitwiseAnd(1 << 3).neq(0)                    //cloud, everywhere
    .or(qa.bitwiseAnd(1 << 4).neq(0).and(inFire.not()));    //shadow, outside only
  return scaleSR(image).updateMask(bad.not());
}

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(wideFrame);

//Keep raw collections because QA_PIXEL is needed to compare scene coverage
var preRaw = l8.filterDate(preStart, preEnd);
var postRaw = l8.filterDate(postStart, postEnd);

//==========================
// PRE-FIRE SCENE CHECK
//==========================
//Calculate clear-sky coverage specifically inside the Port Hills AOI,
//rather than relying only on scene-wide CLOUD_COVER.

var preRawStats = preRaw.map(function(img){

  var qa = img.select('QA_PIXEL');

  var clear = qa.bitwiseAnd(1 << 3).eq(0)      //not cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));         //not cloud shadow

  var clearArea = ee.Image.pixelArea()
    .updateMask(clear)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: aoi,
      scale: 30,
      maxPixels: 1e9
    })
    .getNumber('area');

  return img.set({
    acquisition_date:
      ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),

    clear_percent_AOI:
      clearArea.divide(aoi.area(1)).multiply(100)
  });
});

print('===== PRE-FIRE CANDIDATE SCENES =====');
print('Number of scenes:', preRawStats.size());
print('Dates:', preRawStats.aggregate_array('acquisition_date'));
print('Product IDs:', preRawStats.aggregate_array('LANDSAT_PRODUCT_ID'));
print('Scene CLOUD_COVER:', preRawStats.aggregate_array('CLOUD_COVER'));
print('Clear % inside AOI:', preRawStats.aggregate_array('clear_percent_AOI'));


//Now apply your existing cloud-mask/scaling function
var preCol = preRaw.map(maskL8clouds_PRE);
var postCol = postRaw.map(maskL8clouds_QA);


//==========================
// 4B. PRE-FIRE RECONSTRUCTION   [A3]
//==========================
//Primary scene: 23 Jan 2017 (~96% clear over AOI)
//Gap fill:      16 Jan 2017 (near-complete coverage)
//Purpose: keep most pixels from one date and use the nearest earlier
//clear observation only where the primary scene has missing data.

var jan23_raw = ee.Image(
  preRaw.filter(
    ee.Filter.eq(
      'LANDSAT_PRODUCT_ID',
      'LC08_L2SP_074090_20170123_20200905_02_T1'
    )
  ).first()
);

var jan16_raw = ee.Image(
  preRaw.filter(
    ee.Filter.eq(
      'LANDSAT_PRODUCT_ID',
      'LC08_L2SP_073090_20170116_20200905_02_T1'
    )
  ).first()
);


//Apply the same scaling and cloud masking used elsewhere
var jan23 = maskL8clouds_PRE(jan23_raw)
  .select(outputBands);

var jan16 = maskL8clouds_PRE(jan16_raw)
  .select(outputBands);


//Primary Jan 23 image, Jan 16 fills remaining gaps
var pre = jan23
  .unmask(jan16)
  .clip(wideFrame);


//==========================
// SOURCE DATE / PROVENANCE   [A3]
//==========================
//2 = Jan 23
//1 = Jan 16 gap fill

var jan23Valid = jan23.select('SR_B5').mask();
var jan16Valid = jan16.select('SR_B5').mask();

var preSource = jan23.select('SR_B5')
  .multiply(0)
  .unmask(0)
  .where(jan16Valid, 1)
  .where(jan23Valid, 2)
  .updateMask(jan23Valid.or(jan16Valid))
  .rename('pre_source')
  .toByte();


//==========================
// 30 m ANALYSIS GRID   [A3]
//==========================

//Pixel row/column coordinates on the common 30 m NZTM grid
var gridXY = ee.Image.pixelCoordinates(analysisProj);

var gridCol = gridXY.select('x')
  .rename('grid_col');

var gridRow = gridXY.select('y')
  .rename('grid_row');

print('30 m grid projection:', analysisProj);

print('Analysis projection:', analysisProj);
print('Analysis transform:', analysisProj.transform());


//==========================
// CHECK REMAINING GAPS   [A3]
//==========================

//Valid pixels after Jan23 + Jan16 reconstruction
var preValid = pre.select('SR_B5').mask();

var reconstructedArea = ee.Image.pixelArea()
  .updateMask(preValid)
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 30,
    maxPixels: 1e9
  })
  .getNumber('area');

var totalArea = ee.Image.pixelArea()
  .clip(aoi)
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 30,
    maxPixels: 1e9
  })
  .getNumber('area');

print(
  'Reconstructed PRE coverage %:',
  reconstructedArea.divide(totalArea).multiply(100)
);


//Post-fire image: keep the existing median composite for now
var post = postCol
  .select(outputBands)
  .median()
  .clip(wideFrame);



//Valid for pre/post change analysis (dNBR)
var dnbrValid = pre.select('SR_B5').mask()
  .and(post.select('SR_B5').mask());
//dNBR needs valid pixels on both dates, so coverage is the overlap

print('===== COVERAGE =====');
print('pre-fire scenes:',  preCol.size());
print('post-fire scenes:', postCol.size());
print('dNBR computable (ha):', haIn(dnbrValid));
print('coverage %:', haIn(dnbrValid).divide(1758.88).multiply(100));
print('cloud gap (ha):', ee.Number(1758.88).subtract(haIn(dnbrValid)));

//==========================
// 5. SPECTRAL INDICES
//==========================
//--------------------------
// NBR, Key & Benson (2006)
// Formula: (NIR - SWIR2) / (NIR + SWIR2)
// Fire drops NIR and raises SWIR2, so the difference is amplified
//--------------------------
var nbrPre  = pre.normalizedDifference(['SR_B5','SR_B7']).rename('NBR_pre');
var nbrPost = post.normalizedDifference(['SR_B5','SR_B7']).rename('NBR_post');
var dnbrRaw = nbrPre.subtract(nbrPost).multiply(1000).rename('dNBR');  //x1000, USGS convention

//--------------------------
// NDVI, pre-fire only, for the vegetation mask
// Formula: (NIR - Red) / (NIR + Red)
// Rock and roads are stable in both scenes and would read as unburned
//--------------------------
var ndviPre = pre.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
var vegMask = ndviPre.gt(0.25);
// [A3] NDMI moisture index, added for A3 (used in Section 4 regression)
var ndmiPre = pre
  .normalizedDifference(['SR_B5','SR_B6'])
  .rename('NDMI');
var valid   = dnbrValid.and(vegMask);

//--------------------------
// BSI, an independent check on where bare ground is
// Formula: ((Red+SWIR)-(NIR+Blue)) / ((Red+SWIR)+(NIR+Blue))
//--------------------------
var bsi = pre.expression(
  '((RED + SWIR) - (NIR + BLUE)) / ((RED + SWIR) + (NIR + BLUE))', {
    RED:  pre.select('SR_B4'), SWIR: pre.select('SR_B6'),
    NIR:  pre.select('SR_B5'), BLUE: pre.select('SR_B2')
  }).rename('BSI');

//==========================
// PRE-FIRE CLASSIFICATION CLEANING MASK   [A3: spatial data cleaning -> Section 3]
//==========================

var prePredictors = pre
  .rename([
    'pre_B2','pre_B3','pre_B4',
    'pre_B5','pre_B6','pre_B7'
  ])
  .addBands(ndviPre)
  .addBands(ndmiPre)
  .addBands(bsi)
  .addBands(terrain);

//1 = all predictor bands have valid data
var preAnalysisValid = prePredictors
  .mask()
  .reduce(ee.Reducer.min())
  .rename('pre_analysis_valid');

var predictorCoverage = prePredictors
  .mask()
  .reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: aoi,
    scale: outputScale,
    crs: outputCRS,
    maxPixels: 1e9,
    tileScale: 4
  });

print('===== SPATIAL CLEANING =====');
print('Valid fraction by predictor:', predictorCoverage);

print('Jan 23 primary area (ha):',
      haIn(preSource.eq(2), aoi));

print('Jan 16 gap-fill area (ha):',
      haIn(preSource.eq(1), aoi));

//==========================
// SPATIAL CLEANING: AOI + VALID CELLS   [A3]
//==========================

//Keep only cells inside the fire perimeter
var aoiMask = ee.Image.constant(1)
  .clip(aoi)
  .mask()
  .rename('inside_aoi');

//Valid pre-fire analysis cells
var cleanPreMask = preAnalysisValid
  .and(aoiMask)
  .rename('clean_pre_mask');

print('Clean pre-fire area (ha):',
      haIn(cleanPreMask, aoi));

//==========================
// SPATIAL ALIGNMENT CHECK   [A3]
//==========================

print('===== SPATIAL ALIGNMENT =====');

print('Target analysis grid:', analysisProj);

print('Pre-fire Landsat B5 projection:',
      pre.select('SR_B5').projection());

print('Elevation projection:',
      dem.projection());

print('Slope projection:',
      slope.projection());

print('Northness projection:',
      northness.projection());

print('pre_source projection:',
      preSource.projection());

print('grid_row projection:',
      gridRow.projection());

//==========================
// 6. OFFSET CORRECTION
//==========================
//unburned vegetation should read zero; whatever it reads is seasonal drift
//control ring matched to the fire on elevation, slope and cover type
//crops excluded: a paddock cut between the dates looks like a burn

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2016-12-01','2017-02-12').filterBounds(wideFrame)
  .select('label').mode();
var water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent').unmask(0);

var ring = aoi.buffer(2000).difference(aoi.buffer(500), 1);
//500 m inner setback avoids scorch beyond the containment line

var controlMask = ee.Image.constant(1).clip(ring).mask()
  .and(dem.gte(ee.Number(firePct.get('elev_p2'))))
  .and(dem.lte(ee.Number(firePct.get('elev_p98'))))
  .and(slope.gte(ee.Number(firePct.get('slope_p2'))))
  .and(dw.eq(1).or(dw.eq(2)).or(dw.eq(5)))   //trees, grass, shrub only
  .and(dw.neq(4))                             //no crops
  .and(water.eq(0)).and(vegMask).selfMask();

var ctrl = dnbrRaw.updateMask(controlMask).reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
  geometry: ring, scale: outputScale, maxPixels: 1e9});

var offset = ee.Number(ctrl.get('dNBR_mean'));
var ctrlSD = ee.Number(ctrl.get('dNBR_stdDev'));
var dnbr   = dnbrRaw.subtract(offset).rename('dNBR');

print('===== OFFSET =====');
print('mean dNBR on unburned control:', offset);
print('standard deviation:', ctrlSD);
print('control area (ha):', haIn(controlMask, ring));

//==========================
// 7. THRESHOLD
//==========================
//refugia sit in gullies and shaded slopes, so they cure less than the control
//the two mapped islands are unburned ground INSIDE the fire, a better null
//90th percentile keeps 90% of known refugia unburned by construction
//a percentile, not mean+2sd, because the tail inflates the sd
//Otsu was tested and rejected: it cut inside the burned group

var islCore = unburntFC.geometry();

var islPct = dnbr.updateMask(valid).reduceRegion({
  reducer: ee.Reducer.percentile([50,90,95]),
  geometry: islCore, scale: outputScale, maxPixels: 1e9});

var T = ee.Number(islPct.get('dNBR_p90'));

function falsePositiveRate(t){
  return dnbr.gt(t).updateMask(controlMask).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: ring,
    scale: outputScale, maxPixels: 1e9}).getNumber('dNBR');
}

print('===== THRESHOLD =====');
print('island dNBR p50 / p90 / p95:',
      islPct.values(['dNBR_p50','dNBR_p90','dNBR_p95']));
print('T:', T);
print('2 x control sd would give:', ctrlSD.multiply(2));
print('commission error at T:', falsePositiveRate(T));

print(ui.Chart.image.histogram({
  image: dnbr.updateMask(vegMask), region: aoi, scale: outputScale, maxBuckets: 120
}).setOptions({title:'dNBR, offset corrected, vegetated ground',
               hAxis:{title:'dNBR x1000'}, vAxis:{title:'pixels'}}));

//==========================
// 6B. HISTORICAL DW CLASS-SPECIFIC BAND (provenance only)
//==========================
// Recovered original calculation for band 32 of the existing classification
// stack. Do not rename this to LCDB or use it in the current LCDB analysis.
// Final LCDB offsets are calculated locally from the RAW ring export below.
var dwGroups = ee.List(dnbrRaw.addBands(dw.rename('dw'))
  .updateMask(controlMask)
  .reduceRegion({
    reducer: ee.Reducer.mean().group({groupField: 1, groupName: 'dw'}),
    geometry: ring, scale: outputScale, maxPixels: 1e9
  }).get('groups'));
var dwFrom = dwGroups.map(function(d) { return ee.Dictionary(d).getNumber('dw'); });
var dwMean = dwGroups.map(function(d) { return ee.Dictionary(d).getNumber('mean'); });
var dwOffset = dw.remap(dwFrom, dwMean, offset).toFloat();
var dnbrCS = dnbrRaw.subtract(dwOffset).rename('dNBR_cs');

//==========================
// 8. CLASSIFY AND MEASURE
//==========================
//burned break from my own data; 270 and 660 kept from USGS for comparability
var severity = ee.Image(0)
  .where(dnbr.gt(T),   1)
  .where(dnbr.gt(270), 2)
  .where(dnbr.gt(660), 3)
  .updateMask(valid).rename('severity');

function burnedHa(t){ return haIn(valid.and(dnbr.gt(t))); }

var covered  = haIn(valid);
var burned   = burnedHa(T);
var unburned = haIn(valid.and(dnbr.lte(T)));

print('===== AREAS (ha), covered ground only =====');
print('covered:',   covered);
print('burned:',    burned);
print('unburned:',  unburned);
print('sensitivity T-50 / T+50:', burnedHa(T.subtract(50)), burnedHa(T.add(50)));
print('burned rate %:', burned.divide(covered).multiply(100));
//assumes cloudy ground burned at the same rate as clear ground
print('scaled to the perimeter:', burned.divide(covered).multiply(1758.88));

//how much unburned ground did the hand-drawn map miss?
var inIslands = haIn(valid.and(dnbr.lte(T)), unburntFC.geometry());
print('===== EPHEMERAL REFUGIA =====');
print('my unburned total:', unburned);
print('inside the two official islands:', inIslands);
print('NEW ephemeral (ha):', unburned.subtract(inIslands));
print('scaled to the perimeter:',
      unburned.subtract(inIslands).divide(covered.divide(1758.88)));

var ephemeralRefugia = valid
  .and(dnbr.lte(T))
  .and(
    ee.Image.constant(1)
      .clip(unburntFC.geometry())
      .mask()
      .not()
  )
  .selfMask()
  .rename('ephemeral_refugia');

//==========================
// 9. VALIDATION
//==========================
//the two islands were mapped by hand, so they are independent reference data
var diag = unburntFC.map(function(f){
  var g = f.geometry();
  return f.set({
    recall: dnbr.lte(T).updateMask(valid).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: g,
      scale: outputScale, maxPixels: 1e8}).get('dNBR'),
    mean_dnbr: dnbr.updateMask(valid).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: g,
      scale: outputScale, maxPixels: 1e8}).get('dNBR'),
    covered_ha: haIn(valid, g)
  });
});

print('===== REFERENCE AGREEMENT CHECK =====');
print('mapped area (ha):', unburntFC.aggregate_array('Area'));
print('covered (ha):',     diag.aggregate_array('covered_ha'));
print('mean dNBR inside:', diag.aggregate_array('mean_dnbr'));
print('recall:',           diag.aggregate_array('recall'));
print('officially burned ground also called burned:',
  dnbr.gt(T).updateMask(valid).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: fireWithHoles,
    scale: outputScale, maxPixels: 1e9}).get('dNBR'));

//==========================
// 10. VISUALISATION
//==========================
//reflectance is 0-1 after scaling, so max is 0.3-0.4, not 3000
//set stretches separately for pre and post; the post surface is darker
Map.addLayer(pre.clip(mapFrame),  {bands:['SR_B4','SR_B3','SR_B2'], min:0, max:0.3},
             '1a Natural colour PRE', false);
Map.addLayer(post.clip(mapFrame), {bands:['SR_B4','SR_B3','SR_B2'], min:0, max:0.3},
             '1b Natural colour POST', false);
Map.addLayer(pre.clip(mapFrame),  {bands:['SR_B5','SR_B4','SR_B3'], min:0, max:0.4},
             '2a Colour infrared PRE', false);
Map.addLayer(post.clip(mapFrame), {bands:['SR_B5','SR_B4','SR_B3'], min:0, max:0.4},
             '2b Colour infrared POST', false);

//burn composite: red = SWIR2 which rises, green = NIR which falls
Map.addLayer(pre.clip(mapFrame),  {bands:['SR_B7','SR_B5','SR_B4'], min:0, max:0.4},
             '3a Burn SWIR composite PRE', false);
Map.addLayer(post.clip(mapFrame), {bands:['SR_B7','SR_B5','SR_B4'], min:0, max:0.4},
             '3b Burn SWIR composite POST', true);

Map.addLayer(ndviPre.clip(mapFrame), {min:-0.2, max:0.8,
  palette:['blue','white','yellow','green','darkgreen']}, '4 NDVI pre-fire', false);
Map.addLayer(bsi.clip(mapFrame), {min:-0.5, max:0.5,
  palette:['darkgreen','yellow','orange','brown','white']}, '5 Bare Soil Index', false);
Map.addLayer(dnbr.updateMask(valid).clip(mapFrame), {min:-200, max:800,
  palette:['2c7bb6','ffffbf','d7191c']}, '6 dNBR corrected', false);  //diverging, centred on 0

Map.addLayer(severity.clip(mapFrame), {min:0, max:3,
  palette:['1a9641','ffffbf','fdae61','d7191c']}, '7 Severity', true);
Map.addLayer(dnbrValid.not().selfMask().clip(mapFrame), {palette:'808080'},
             '8 No data (cloud)', true, 0.75);  //never white; white reads as unburned

Map.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(aoi)]), 0, 3),
             {palette:'ffcc00'}, '9 Fire perimeter', true);
Map.addLayer(ee.Image().paint(unburntFC, 0, 2),
             {palette:'00ffff'}, '10 Official unburnt islands', true);

//unclipped: the rectangular patches west are cut paddocks, not fire
//evidence that this index detects vegetation removal, not combustion
Map.addLayer(dnbr.gt(T).updateMask(vegMask).selfMask(), {palette:'d7191c'},
             '11 Wide view, note the paddock shapes', false);

Map.addLayer(
  ephemeralRefugia,
  {palette:['00FFFF']},
  'Ephemeral refugia',
  false
);
//==========================
// EXPORT TERRAIN RESAMPLING
//==========================

//Only terrainOut was actually used in the historical export. Do not silently
//switch the spectral/index bands to unused bilinear variants: that changes data.
var terrainOut  = terrain.resample('bilinear');

//==========================
// 11. EXPORT TO GEOTIFF   [A3: recovered 32-band classification stack]
//==========================
//Band order (32):
//1-6 pre_B2..pre_B7   7-12 post_B2..post_B7
//13 NBR_pre  14 NBR_post  15 dNBR  16 NDVI  17 NDMI  18 BSI
//19 severity  20 dnbr_data  21 pre_source  22 grid_row  23 grid_col
//24 elev  25 slope  26 northness  27 ephemeral_refugia
//28 pre_analysis_valid  29 inside_aoi  30 clean_pre_mask
//31 post_neg_refl  32 dNBR_cs (historical DW; NOT current dNBR_lc)
var mainBandOrder = [
  'pre_B2','pre_B3','pre_B4','pre_B5','pre_B6','pre_B7',
  'post_B2','post_B3','post_B4','post_B5','post_B6','post_B7',
  'NBR_pre','NBR_post','dNBR','NDVI','NDMI','BSI',
  'severity','dnbr_data','pre_source','grid_row','grid_col',
  'elev','slope','northness','ephemeral_refugia',
  'pre_analysis_valid','inside_aoi','clean_pre_mask','post_neg_refl','dNBR_cs'
];

var stack = pre
  .rename([
    'pre_B2','pre_B3','pre_B4',
    'pre_B5','pre_B6','pre_B7'
  ])

  .addBands(
    post.rename([
      'post_B2','post_B3','post_B4',
      'post_B5','post_B6','post_B7'
    ])
  )

  .addBands(nbrPre)
  .addBands(nbrPost)
  .addBands(dnbr)

  .addBands(ndviPre)
  .addBands(ndmiPre)
  .addBands(bsi)

  .addBands(severity)
  .addBands(dnbrValid.rename('dnbr_data'))

  //Which date supplied each pre-fire pixel
  .addBands(preSource)

  //30 m grid identity
  .addBands(gridRow)
  .addBands(gridCol)

  //Terrain variables
  .addBands(terrainOut)
  .addBands(ephemeralRefugia)
  .addBands(preAnalysisValid)


  .addBands(aoiMask)
  .addBands(cleanPreMask)

  .toFloat();

// Recovered additions must precede Export.image.toDrive so its image includes
// both bands. The earlier stackA3 file contains 31 bands, without dNBR_cs.
var postNegRefl = post.lt(0).reduce(ee.Reducer.max()).rename('post_neg_refl');
stack = stack.addBands(postNegRefl).addBands(dnbrCS).toFloat();

//Select by name, never relabel missing bands. A missing required band causes
//a GEE error instead of silently exporting an incomplete stack.
stack = stack.select(mainBandOrder);
print('===== CANONICAL INPUT EXPORTS =====');
print('Main stack: expected 32 bands', stack.bandNames().size());
print('Main stack band order:', stack.bandNames());
print('Post-fire input product IDs:', postRaw.aggregate_array('LANDSAT_PRODUCT_ID'));

Export.image.toDrive({
  image: stack,
  description: 'PortHills2017_stack',
  folder: 'GEE_exports',
  fileNamePrefix: 'PortHills2017_stack',
  region: aoi,
  crs: outputCRS,
  crsTransform: exportTransform,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {noData: -9999}
});

Export.table.toDrive({
  //Historical pooled-offset diagnostics, NOT the final LCDB island assessment.
  collection: diag,
  description: 'PortHills2017_island_validation',
  folder: 'GEE_exports',
  fileFormat: 'CSV'
});

//==========================
// SPRING 2016: recovered original 9-band classifier input
//==========================
// Preserve the original date window, PRE cloud/shadow mask and summer baseline.
// filterDate end is exclusive: [2016-09-01, 2016-11-30).
var springRaw = l8.filterDate('2016-09-01', '2016-11-30');
var spring = springRaw.map(maskL8clouds_PRE)
  .select(outputBands).median().clip(wideFrame);
var ndviSpr = spring.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI_spr');
var yellow = spring.normalizedDifference(['SR_B4','SR_B3']).rename('YI_spr');
var dNDVI = ndviSpr.subtract(ndviPre).rename('dNDVI_spr_sum');
var seasonal = spring.rename(['spr_B2','spr_B3','spr_B4','spr_B5','spr_B6','spr_B7'])
  .addBands(ndviSpr).addBands(yellow).addBands(dNDVI).toFloat();
var springBandOrder = [
  'spr_B2','spr_B3','spr_B4','spr_B5','spr_B6','spr_B7',
  'NDVI_spr','YI_spr','dNDVI_spr_sum'
];
seasonal = seasonal.select(springBandOrder);
print('Spring stack: expected 9 bands', seasonal.bandNames().size());
print('Spring stack band order:', seasonal.bandNames());
print('Spring input product IDs:', springRaw.aggregate_array('LANDSAT_PRODUCT_ID'));
Export.image.toDrive({
  image: seasonal, description: 'PortHills2016_spring', folder: 'GEE_exports',
  fileNamePrefix: 'PortHills2016_spring', region: aoi, crs: outputCRS,
  crsTransform: exportTransform, maxPixels: 1e13, fileFormat: 'GeoTIFF',
  formatOptions: {noData: -9999}
});

//==========================
// 12. CURRENT LCDB WORKFLOW: export RAW inputs, not a DW-corrected response
//==========================
// Recovered from ERST619/LCDB_offset_GEE_section12.js. LCDB classes are added
// locally by python/03_relationships/00_build_regression_inputs.py using buffered
// LCDB 2012 polygons, 3x3 majority sampling and the original offset/fallback set.
// No DW filter is applied to lcRingBase. dw_label is an unused comparison band.
var lcFrame = aoi.buffer(2500);
var lcRingBase = ee.Image.constant(1).clip(ring).mask()
  .and(dem.gte(ee.Number(firePct.get('elev_p2'))))
  .and(dem.lte(ee.Number(firePct.get('elev_p98'))))
  .and(slope.gte(ee.Number(firePct.get('slope_p2'))))
  .and(water.eq(0)).and(vegMask).rename('ring_base');
var lcIsland = ee.Image.constant(1).clip(unburntFC.geometry()).mask()
  .rename('official_island');
var lcExport = nbrPre.addBands(nbrPost)
  .addBands(dnbrRaw.rename('dNBR_raw'))
  .addBands(dnbrValid.rename('dnbr_data'))
  .addBands(vegMask.rename('veg_mask'))
  .addBands(lcRingBase).addBands(lcIsland).addBands(aoiMask)
  .addBands(dw.rename('dw_label')).addBands(gridRow).addBands(gridCol).toFloat();
var lcdbRingBandOrder = [
  'NBR_pre','NBR_post','dNBR_raw','dnbr_data','veg_mask','ring_base',
  'official_island','inside_aoi','dw_label','grid_row','grid_col'
];
lcExport = lcExport.select(lcdbRingBandOrder);
print('LCDB raw ring: expected 11 bands', lcExport.bandNames().size());
print('LCDB raw ring band order:', lcExport.bandNames());
Export.image.toDrive({
  image: lcExport, description: 'PortHills2017_ring_LCDB', folder: 'GEE_exports',
  fileNamePrefix: 'PortHills2017_ring_LCDB', region: lcFrame,
  crs: outputCRS, crsTransform: exportTransform, maxPixels: 1e13,
  fileFormat: 'GeoTIFF', formatOptions: {noData: -9999}
});
