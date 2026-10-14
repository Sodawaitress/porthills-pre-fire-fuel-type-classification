#==========================
# KT4 | Figure 9, classification confidence
#==========================
#the fuel map itself is Map 1 (ArcGIS), so I only keep the confidence panel
#crops the right half of Fig5_8_class_map_SSp_masked.png from 08_final_SSp.py
#run after 08: python3 11_fig_confidence.py
#output: Fig5_8b_confidence.png
from PIL import Image

im = Image.open("Fig5_8_class_map_SSp_masked.png")
w, h = im.size
im.crop((int(w * 0.49), int(h * 0.105), w, h)).save("Fig5_8b_confidence.png")   # right half = panel (b), without its "(b)" title
print("saved Fig5_8b_confidence.png", im.size, "->", (w - int(w * 0.49), h))
