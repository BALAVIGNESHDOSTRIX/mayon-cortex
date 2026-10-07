# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# DUAL LICENSED: GNU Affero General Public License v3.0 (AGPL-3.0)
# or Commercial License.
#
# Open-source use is governed by the AGPL-3.0 license. For proprietary
# or commercial applications seeking exemption from copyleft requirements,
# a commercial license must be obtained from INFIDOS LLP.
# Contact: balavignesh@infidos.com | https://infidos.com
# ==============================================================================

import numpy as np
from PIL import Image
from pathlib import Path
import scipy.ndimage as ndi

images_dir = Path("demos/data/images")
out_dir = Path("demos/demo_vision_output/test_fix")

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

def extract_solid_subject_mask(img_arr, label="cat"):
    h, w, _ = img_arr.shape
    
    # 1. Background color estimate from corners
    corners = np.concatenate([
        img_arr[:45, :45].reshape(-1, 3),
        img_arr[:45, -45:].reshape(-1, 3),
        img_arr[-45:, :45].reshape(-1, 3) if label != "cat" else img_arr[:45, :45].reshape(-1, 3),
    ], axis=0)
    bg_mean = np.median(corners, axis=0)
    
    # 2. Color difference
    diff = np.linalg.norm(img_arr.astype(np.float32) - bg_mean[None, None, :], axis=2)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    
    # 3. Spatial central prior
    y, x = np.ogrid[:h, :w]
    if label == "cat":
        cy, cx = h * 0.55, w * 0.50
        ry, rx = h * 0.42, w * 0.46
    else:  # dog
        cy, cx = h * 0.52, w * 0.50
        ry, rx = h * 0.46, w * 0.40
        
    dist_sq = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    in_subject_ellipse = dist_sq <= 1.0
    
    # Threshold within ellipse
    threshold = np.percentile(diff[in_subject_ellipse], 15)
    binary = (diff > threshold) & in_subject_ellipse
    
    # Always include core bounding box of face/chest
    core_box = np.zeros((h, w), dtype=bool)
    if label == "cat":
        core_box[int(h*0.25):int(h*0.88), int(w*0.18):int(w*0.82)] = True
    else:
        core_box[int(h*0.18):int(h*0.92), int(w*0.25):int(w*0.75)] = True
    binary = binary | (core_box & in_subject_ellipse)
    
    # Fill all interior holes so eyes/chest/nose are 100% solid
    binary_filled = ndi.binary_fill_holes(binary)
    binary_closed = ndi.binary_closing(binary_filled, structure=np.ones((11, 11)))
    
    # Smooth outer feather margin
    mask_float = binary_closed.astype(np.float32)
    mask_feathered = ndi.gaussian_filter(mask_float, sigma=4.0)
    
    # Ensure core remains exactly 1.0
    core_solid = ndi.binary_erosion(binary_closed, structure=np.ones((7, 7)))
    mask_feathered[core_solid] = 1.0
    
    return np.clip(mask_feathered, 0.0, 1.0)

cat_mask = extract_solid_subject_mask(cat_img, "cat")
dog_mask = extract_solid_subject_mask(dog_img, "dog")

# Re-compose
def compose_scene(bg_img, fg1_img, fg1_mask, fg2_img, fg2_mask):
    H, W, _ = bg_img.shape
    composite = bg_img.astype(np.float32).copy()
    
    # Cat on left foreground
    cat_target_h = int(H * 0.46)
    cat_target_w = int(W * 0.46)
    cat_pil = Image.fromarray(fg1_img).resize((cat_target_w, cat_target_h), Image.Resampling.LANCZOS)
    cat_mask_pil = Image.fromarray((fg1_mask * 255).astype(np.uint8)).resize((cat_target_w, cat_target_h), Image.Resampling.LANCZOS)
    
    cat_resized = np.array(cat_pil).astype(np.float32)
    cat_mask_resized = np.array(cat_mask_pil).astype(np.float32) / 255.0
    
    # Dog on right foreground
    dog_target_h = int(H * 0.58)
    dog_target_w = int(W * 0.58)
    dog_pil = Image.fromarray(fg2_img).resize((dog_target_w, dog_target_h), Image.Resampling.LANCZOS)
    dog_mask_pil = Image.fromarray((fg2_mask * 255).astype(np.uint8)).resize((dog_target_w, dog_target_h), Image.Resampling.LANCZOS)
    
    dog_resized = np.array(dog_pil).astype(np.float32)
    dog_mask_resized = np.array(dog_mask_pil).astype(np.float32) / 255.0
    
    # Place cat at bottom left
    y_cat = H - cat_target_h - 20
    x_cat = 25
    alpha_cat = cat_mask_resized[:, :, None]
    composite[y_cat:y_cat + cat_target_h, x_cat:x_cat + cat_target_w] = (
        alpha_cat * cat_resized + (1.0 - alpha_cat) * composite[y_cat:y_cat + cat_target_h, x_cat:x_cat + cat_target_w]
    )
    
    # Place dog at bottom right
    y_dog = H - dog_target_h - 15
    x_dog = W - dog_target_w - 20
    alpha_dog = dog_mask_resized[:, :, None]
    composite[y_dog:y_dog + dog_target_h, x_dog:x_dog + dog_target_w] = (
        alpha_dog * dog_resized + (1.0 - alpha_dog) * composite[y_dog:y_dog + dog_target_h, x_dog:x_dog + dog_target_w]
    )
    
    return np.clip(composite, 0, 255).astype(np.uint8)

comp = compose_scene(nat_img, cat_img, cat_mask, dog_img, dog_mask)
Image.fromarray(comp).save(out_dir / "test_composed_solid.png")
print("Saved solid composed image!")
