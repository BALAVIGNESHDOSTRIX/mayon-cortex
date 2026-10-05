# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# This software and associated documentation files contain proprietary and
# confidential information of INFIDOS LLP and BALAVIGNESH M. Unauthorized
# copying, modification, distribution, transmission, or reproduction of this
# material, via any medium, is strictly prohibited without prior written
# permission from INFIDOS LLP.
# ==============================================================================

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from pathlib import Path

img_dir = Path("demos/data/images")
cat_img = np.array(Image.open(img_dir / "cat.jpg").convert("RGB"))
dog_img = np.array(Image.open(img_dir / "dog.jpg").convert("RGB"))

print("Cat shape:", cat_img.shape, "Dog shape:", dog_img.shape)

# Let's inspect dog segmentation:
# Dog: Golden fur (R > G, R > B), tongue (pink: R high, G, B low), eyes/nose (dark: R, G, B low).
# Lawn/trees: G > R, or dark green.
# Sky: B > R.
dog_f = dog_img.astype(np.float32)
R_d, G_d, B_d = dog_f[:, :, 0], dog_f[:, :, 1], dog_f[:, :, 2]

# Dog body/fur condition:
is_fur = (R_d > G_d + 5) & (R_d > B_d + 20) & (R_d > 60)
is_dark_feature = (R_d < 70) & (G_d < 70) & (B_d < 70) & (dog_f.sum(axis=2) < 180) # nose, eyes, collar
is_mouth = (R_d > 120) & (G_d < 90) & (B_d < 110) # tongue
dog_fg_raw = is_fur | is_dark_feature | is_mouth

# Restrict to dog bounding region (dog is centered from y ~ 10% to 98%, x ~ 25% to 78%)
h_d, w_d = dog_img.shape[:2]
dog_fg_raw[:int(h_d*0.10), :] = False
dog_fg_raw[:, :int(w_d*0.22)] = False
dog_fg_raw[:, int(w_d*0.82):] = False

# Morphological clean up
dog_filled = ndi.binary_fill_holes(dog_fg_raw)
dog_closed = ndi.binary_closing(dog_filled, structure=np.ones((15, 15)))
dog_opened = ndi.binary_opening(dog_closed, structure=np.ones((7, 7)))

# Find largest connected component (the dog)
labeled, num_features = ndi.label(dog_opened)
if num_features > 0:
    sizes = ndi.sum(dog_opened, labeled, range(1, num_features + 1))
    max_label = np.argmax(sizes) + 1
    dog_mask_binary = (labeled == max_label)
else:
    dog_mask_binary = dog_opened

dog_mask_filled = ndi.binary_fill_holes(dog_mask_binary)
dog_feathered = ndi.gaussian_filter(dog_mask_filled.astype(np.float32), sigma=2.5)

# Save dog cutout
dog_cutout = (dog_f * dog_feathered[:, :, None]).clip(0, 255).astype(np.uint8)
Image.fromarray(dog_cutout).save("scratch/dog_cutout.png")
print("Saved scratch/dog_cutout.png")

# Now Cat:
cat_f = cat_img.astype(np.float32)
R_c, G_c, B_c = cat_f[:, :, 0], cat_f[:, :, 1], cat_f[:, :, 2]
# Cat is orange tabby: warm orange fur, ears at top, sitting on knitted blanket
# Blanket is near bottom (y > 65%) and has whitish/cream color (R, G, B all > 170 and close to each other, saturation low)
# Sofa in background has low saturation or muted brown/grey
# Cat fur has strong warmth: (R_c - B_c > 35) & (R_c - G_c > 15) & (R_c > 90)
is_cat_fur = (R_c > G_c + 12) & (R_c > B_c + 30) & (R_c > 75)
is_cat_dark = (R_c < 60) & (G_c < 60) & (B_c < 60) # eyes outline, nose
is_cat_eyes = (R_c > 140) & (G_c > 110) & (B_c < 80) # amber/yellow eyes
is_cat_white = (R_c > 160) & (G_c > 150) & (B_c > 140) & (np.abs(R_c - G_c) < 25) # white muzzle/bib
# Cat is centered
h_c, w_c = cat_img.shape[:2]
cat_region = np.zeros((h_c, w_c), dtype=bool)
# Cat body: y from 0.08 to 0.95, x from 0.03 to 0.85
cat_region[int(h_c*0.08):int(h_c*0.95), int(w_c*0.03):int(w_c*0.88)] = True

cat_fg_raw = (is_cat_fur | is_cat_dark | is_cat_eyes) & cat_region

# Fill and connect
cat_filled = ndi.binary_fill_holes(cat_fg_raw)
cat_closed = ndi.binary_closing(cat_filled, structure=np.ones((17, 17)))
cat_opened = ndi.binary_opening(cat_closed, structure=np.ones((7, 7)))

labeled_c, num_c = ndi.label(cat_opened)
if num_c > 0:
    sizes_c = ndi.sum(cat_opened, labeled_c, range(1, num_c + 1))
    max_lbl_c = np.argmax(sizes_c) + 1
    cat_mask_binary = (labeled_c == max_lbl_c)
else:
    cat_mask_binary = cat_opened

cat_mask_filled = ndi.binary_fill_holes(cat_mask_binary)
cat_feathered = ndi.gaussian_filter(cat_mask_filled.astype(np.float32), sigma=2.5)

cat_cutout = (cat_f * cat_feathered[:, :, None]).clip(0, 255).astype(np.uint8)
Image.fromarray(cat_cutout).save("scratch/cat_cutout.png")
print("Saved scratch/cat_cutout.png")
