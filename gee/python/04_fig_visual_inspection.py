#==========================
# KT4 | Figure 1, visual inspection
#==========================
#three panels, same extent, fire perimeter in yellow
#  (a) pre-fire true colour B4/B3/B2, Jan 23 2017 with gaps filled from Jan 16
#  (b) pre-fire false colour B5/B4/B3, NIR shown as red
#  (c) spring 2016 true colour, median of Sep-Nov 2016, when gorse flowers
#masks and dates follow 00_gee_PortHills2017_stack.js (maskL8clouds_PRE)
#exports nothing, doesn't touch data/
#run from this folder: python3 04_fig_visual_inspection.py
#output: outputs/Fig_visual_inspection.png

import io
import os
import urllib.request

import ee
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#use your own GEE project: export EE_PROJECT=<id>, otherwise it falls back to mine
PROJECT = os.environ.get('EE_PROJECT', 'genuine-hold-427410-f0')
ee.Initialize(project=PROJECT)

raw = ee.FeatureCollection(f'projects/{PROJECT}/assets/Port_Hills_2017_Fire_Boundary')
fire_with_holes = raw.filter(ee.Filter.eq('Type', 'Fire Boundary')).first().geometry()
aoi = fire_with_holes.union(raw.filter(ee.Filter.eq('Type', 'Unburnt')).geometry(), 1)
wide_frame = aoi.buffer(4000)
map_frame = aoi.buffer(500)

BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(wide_frame)


def mask_pre(image):
    #Collection 2 scaling, QA_PIXEL bit 3 = cloud, bit 4 = cloud shadow, both removed
    qa = image.select('QA_PIXEL')
    bad = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0))
    return image.select('SR_B.').multiply(0.0000275).add(-0.2).updateMask(bad.Not())


def by_id(pid):
    return mask_pre(ee.Image(l8.filter(ee.Filter.eq('LANDSAT_PRODUCT_ID', pid)).first())).select(BANDS)


pre = by_id('LC08_L2SP_074090_20170123_20200905_02_T1').unmask(
    by_id('LC08_L2SP_073090_20170116_20200905_02_T1'))
spring = l8.filterDate('2016-09-01', '2016-11-30').map(mask_pre).select(BANDS).median()

panels = [
    ('(a) Pre-fire true colour (B4/B3/B2)\n23 Jan 2017', pre, ['SR_B4', 'SR_B3', 'SR_B2'], 0.3),
    ('(b) Pre-fire false colour (B5/B4/B3)\n23 Jan 2017, NIR as red', pre, ['SR_B5', 'SR_B4', 'SR_B3'], 0.4),
    ('(c) Spring true colour (B4/B3/B2)\nmedian Sep–Nov 2016', spring, ['SR_B4', 'SR_B3', 'SR_B2'], 0.3),
]

# extent in NZTM metres, so the scale bar is true to ground distance
corners = np.array(map_frame.bounds(1, 'EPSG:2193').coordinates().getInfo()[0])
(x0, y0), (x1, y1) = corners.min(0), corners.max(0)
outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(aoi)]), 1, 2).visualize(palette=['ffff00'])


def fetch(img, bands, vmax):
    rgb = img.visualize(bands=bands, min=0, max=vmax).blend(outline).clip(map_frame)
    url = rgb.getThumbURL({'region': map_frame, 'dimensions': 1200, 'crs': 'EPSG:2193', 'format': 'png'})
    return np.array(Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert('RGBA'))


fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
for k, (ax, (title, img, bands, vmax)) in enumerate(zip(axes, panels)):
    ax.imshow(fetch(img, bands, vmax), extent=(x0, x1, y0, y1))
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if k == 0:   # scale bar and north arrow on the first panel only
        sx, sy = x0 + 0.06 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([sx, sx + 1000], [sy, sy], color='black', lw=4, solid_capstyle='butt')   # frame corners are white (outside the clip)
        ax.text(sx + 500, sy + 0.025 * (y1 - y0), '1 km', color='black', ha='center', fontsize=10)
        ax.annotate('N', xy=(0.92, 0.95), xytext=(0.92, 0.82), xycoords='axes fraction', color='black',
                    ha='center', fontsize=12, fontweight='bold',
                    arrowprops=dict(color='black', width=2.5, headwidth=10))
plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/Fig_visual_inspection.png', dpi=200, bbox_inches='tight')
print('saved outputs/Fig_visual_inspection.png')
