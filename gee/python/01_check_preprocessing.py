#==========================
# Port Hills 2017 | preprocessing check
#==========================
#Python version of sections 1-4B of 00_gee_PortHills2017_stack.js:
#study area, dates, terrain, scaling, cloud/shadow masks, scene check,
#Jan 23 + Jan 16 rebuild, post-fire median, coverage
#prints the same numbers as the Console, so I can check them outside the Code Editor
#exports nothing, doesn't touch the stack in data/
#run from this folder: python3 01_check_preprocessing.py
#output: outputs/pre_truecolour_L8.png, outputs/post_truecolour_L8.png

import os
import urllib.request

import ee

#use your own GEE project: export EE_PROJECT=<id>, otherwise it falls back to mine
PROJECT = os.environ.get('EE_PROJECT', 'genuine-hold-427410-f0')
ee.Initialize(project=PROJECT)

# ===== 1. STUDY AREA (AOI) =====
ASSET_ID = f'projects/{PROJECT}/assets/Port_Hills_2017_Fire_Boundary'
raw = ee.FeatureCollection(ASSET_ID)

fire_with_holes = raw.filter(ee.Filter.eq('Type', 'Fire Boundary')).first().geometry()
unburnt_fc = raw.filter(ee.Filter.eq('Type', 'Unburnt'))
aoi = fire_with_holes.union(unburnt_fc.geometry(), 1)   # fill the two holes
wide_frame = aoi.buffer(4000)                            # wide enough for the control ring
map_frame = aoi.buffer(500)                              # display extent only

HA = 10000
pixel_area = ee.Image.pixelArea()


def ha_in(mask, geom=None):
    return ee.Number(pixel_area.updateMask(mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom or aoi,
        scale=30, maxPixels=1e9, tileScale=4).get('area')).divide(HA)


# source file is Web Mercator; .area() is geodesic so it is correct
print('===== OFFICIAL MAP (ha) =====')
print('perimeter:       ', aoi.area(1).divide(HA).getInfo())
print('official unburnt:', unburnt_fc.geometry().area(1).divide(HA).getInfo())
print('official burned: ', fire_with_holes.area(1).divide(HA).getInfo())

# ===== 2. STUDY PERIOD AND OUTPUT SETTINGS =====
PRE_START, PRE_END = '2016-12-01', '2017-02-11'     # fire ignited 13 Feb
POST_START, POST_END = '2017-02-24', '2017-03-30'   # hotspots ran to ~25 Feb; short to avoid regrowth
OUTPUT_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
#                blue     green    red      NIR      SWIR1    SWIR2

# ===== 3. TERRAIN =====
# (not needed for the numbers below; kept so the section list matches the JS script)
dem = ee.Image('USGS/SRTMGL1_003').rename('elev')

# ===== 4. LOAD LANDSAT AND MASK CLOUD =====
def scale_sr(image):
    #Collection 2 Level-2 scaling: DN * 0.0000275 - 0.2
    return ee.Image(image.select('SR_B.').multiply(0.0000275).add(-0.2)
                    .copyProperties(image, image.propertyNames()))


def mask_l8_clouds_pre(image):
    #QA_PIXEL bit 3 = cloud, bit 4 = cloud shadow, both removed everywhere
    qa = image.select('QA_PIXEL')
    bad = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0))
    return scale_sr(image).updateMask(bad.Not())


def mask_l8_clouds_qa(image):
    # post-fire: cloud removed everywhere, shadow removed outside the fire only
    qa = image.select('QA_PIXEL')
    in_fire = ee.Image.constant(1).clip(aoi).mask()
    bad = (qa.bitwiseAnd(1 << 3).neq(0)
           .Or(qa.bitwiseAnd(1 << 4).neq(0).And(in_fire.Not())))
    return scale_sr(image).updateMask(bad.Not())


l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(wide_frame)
pre_raw = l8.filterDate(PRE_START, PRE_END)
post_raw = l8.filterDate(POST_START, POST_END)

# ===== PRE-FIRE SCENE CHECK: clear-sky % inside the AOI, not scene-wide CLOUD_COVER =====
def add_clear_stats(img):
    qa = img.select('QA_PIXEL')
    clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    clear_area = ee.Number(ee.Image.pixelArea().updateMask(clear).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9).get('area'))
    return img.set({
        'acquisition_date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
        'clear_percent_AOI': clear_area.divide(aoi.area(1)).multiply(100)})


pre_raw_stats = pre_raw.map(add_clear_stats)
info = pre_raw_stats.reduceColumns(
    ee.Reducer.toList(4),
    ['acquisition_date', 'LANDSAT_PRODUCT_ID', 'CLOUD_COVER', 'clear_percent_AOI']).get('list').getInfo()

print('\n===== PRE-FIRE CANDIDATE SCENES =====')
print('Number of scenes:', len(info))
print(f"{'date':<12}{'product ID':<44}{'scene cloud %':>14}{'clear % in AOI':>16}")
for date, pid, cc, clear in info:
    print(f"{date:<12}{pid:<44}{cc:>14.1f}{clear:>16.1f}")

pre_col = pre_raw.map(mask_l8_clouds_pre)
post_col = post_raw.map(mask_l8_clouds_qa)

# ===== 4B. PRE-FIRE RECONSTRUCTION: Jan 23 primary, Jan 16 fills its gaps =====
def by_id(pid):
    return ee.Image(pre_raw.filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', pid)).first())


jan23 = mask_l8_clouds_pre(by_id('LC08_L2SP_074090_20170123_20200905_02_T1')).select(OUTPUT_BANDS)
jan16 = mask_l8_clouds_pre(by_id('LC08_L2SP_073090_20170116_20200905_02_T1')).select(OUTPUT_BANDS)
pre = jan23.unmask(jan16).clip(wide_frame)

jan23_valid = jan23.select('SR_B5').mask()
jan16_valid = jan16.select('SR_B5').mask()

# ===== CHECK REMAINING GAPS =====
pre_valid = pre.select('SR_B5').mask()
reconstructed = ee.Number(ee.Image.pixelArea().updateMask(pre_valid).reduceRegion(
    reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9).get('area'))
total = ee.Number(ee.Image.pixelArea().clip(aoi).reduceRegion(
    reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9).get('area'))

print('\n===== PRE-FIRE RECONSTRUCTION =====')
print('Jan 23 valid inside AOI (ha):', ha_in(jan23_valid).getInfo())
print('Jan 16 used for gap fill (ha):', ha_in(jan16_valid.And(jan23_valid.Not())).getInfo())
print('Reconstructed PRE coverage %:', reconstructed.divide(total).multiply(100).getInfo())

# post-fire: median composite of the masked scenes
post = post_col.select(OUTPUT_BANDS).median().clip(wide_frame)

# dNBR needs valid pixels on both dates, so coverage is the overlap
dnbr_valid = pre.select('SR_B5').mask().And(post.select('SR_B5').mask())
post_dates = post_raw.aggregate_array('system:time_start').map(
    lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo()

#1758.88 ha is the perimeter area hard-coded in the JS
print('\n===== COVERAGE =====')
print('pre-fire scenes: ', pre_col.size().getInfo())
print('post-fire scenes:', post_col.size().getInfo(), post_dates)
print('dNBR computable (ha):', ha_in(dnbr_valid).getInfo())
print('coverage %:', ha_in(dnbr_valid).divide(1758.88).multiply(100).getInfo())
print('cloud gap (ha):', ee.Number(1758.88).subtract(ha_in(dnbr_valid)).getInfo())

# ===== VISUAL CHECK: same stretch as the JS Map layer (B4/B3/B2, 0–0.3) =====
os.makedirs('outputs', exist_ok=True)
outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(aoi)]), 1, 2) \
    .visualize(palette=['yellow'])
for name, img in [('pre', pre), ('post', post)]:
    rgb = img.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min=0, max=0.3).blend(outline)
    url = rgb.clip(map_frame).getThumbURL(
        {'region': map_frame, 'dimensions': 1600, 'crs': 'EPSG:2193', 'format': 'png'})
    path = f'outputs/{name}_truecolour_L8.png'
    urllib.request.urlretrieve(url, path)
    print('saved', path)
