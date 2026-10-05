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

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

sys.path.insert(0, ".")

images_dir = Path("demos/data/images")
out_dir = Path("demos/demo_vision_output/test_fix")
out_dir.mkdir(parents=True, exist_ok=True)

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

def extract_subject_mask(img_arr):
    """Extract foreground subject mask from image via color contrast & central prior."""
    h, w, _ = img_arr.shape
    corners = np.concatenate([
        img_arr[:40, :40].reshape(-1, 3),
        img_arr[:40, -40:].reshape(-1, 3),
    ], axis=0)
    bg_mean = np.median(corners, axis=0)
    
    diff = np.linalg.norm(img_arr.astype(np.float32) - bg_mean[None, None, :], axis=2)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    
    y, x = np.ogrid[:h, :w]
    cy, cx = h * 0.52, w * 0.50
    dist_center = np.sqrt(((y - cy)/(h*0.5))**2 + ((x - cx)/(w*0.5))**2)
    center_weight = np.clip(1.0 - 0.45 * dist_center, 0.0, 1.0)
    
    saliency = diff * center_weight
    threshold = np.percentile(saliency, 35)
    raw_mask = (saliency > threshold).astype(np.float32)
    
    # Clean mask using morphological closing & gaussian feather
    mask_closed = ndi.binary_closing(raw_mask > 0.5, structure=np.ones((9, 9)))
    mask_smooth = ndi.gaussian_filter(mask_closed.astype(np.float32), sigma=6.0)
    return np.clip(mask_smooth, 0.0, 1.0)

print("Computing subject masks...")
cat_mask = extract_subject_mask(cat_img)
dog_mask = extract_subject_mask(dog_img)

# Test Composition of cat and dog into nature
def compose_scene(bg_img, fg1_img, fg1_mask, fg2_img, fg2_mask):
    """
    Composite two foreground entities seamlessly into a background landscape.
    Place fg1 (cat) on left foreground and fg2 (dog) on right foreground.
    """
    H, W, _ = bg_img.shape
    composite = bg_img.astype(np.float32).copy()
    
    # Resize cat to ~45% of scene height, placed at bottom-left
    cat_target_h = int(H * 0.48)
    cat_target_w = int(W * 0.48)
    cat_pil = Image.fromarray(fg1_img).resize((cat_target_w, cat_target_h), Image.Resampling.LANCZOS)
    cat_mask_pil = Image.fromarray((fg1_mask * 255).astype(np.uint8)).resize((cat_target_w, cat_target_h), Image.Resampling.LANCZOS)
    
    cat_resized = np.array(cat_pil).astype(np.float32)
    cat_mask_resized = np.array(cat_mask_pil).astype(np.float32) / 255.0
    
    # Resize dog to ~56% of scene height, placed at bottom-right
    dog_target_h = int(H * 0.58)
    dog_target_w = int(W * 0.58)
    dog_pil = Image.fromarray(fg2_img).resize((dog_target_w, dog_target_h), Image.Resampling.LANCZOS)
    dog_mask_pil = Image.fromarray((fg2_mask * 255).astype(np.uint8)).resize((dog_target_w, dog_target_h), Image.Resampling.LANCZOS)
    
    dog_resized = np.array(dog_pil).astype(np.float32)
    dog_mask_resized = np.array(dog_mask_pil).astype(np.float32) / 255.0
    
    # Place cat at (y=H - cat_target_h - 20, x=30)
    y_cat = H - cat_target_h - 25
    x_cat = 25
    alpha_cat = cat_mask_resized[:, :, None]
    composite[y_cat:y_cat + cat_target_h, x_cat:x_cat + cat_target_w] = (
        alpha_cat * cat_resized + (1.0 - alpha_cat) * composite[y_cat:y_cat + cat_target_h, x_cat:x_cat + cat_target_w]
    )
    
    # Place dog at (y=H - dog_target_h - 10, x=W - dog_target_w - 20)
    y_dog = H - dog_target_h - 15
    x_dog = W - dog_target_w - 20
    alpha_dog = dog_mask_resized[:, :, None]
    composite[y_dog:y_dog + dog_target_h, x_dog:x_dog + dog_target_w] = (
        alpha_dog * dog_resized + (1.0 - alpha_dog) * composite[y_dog:y_dog + dog_target_h, x_dog:x_dog + dog_target_w]
    )
    
    return np.clip(composite, 0, 255).astype(np.uint8)

comp_img = compose_scene(nat_img, cat_img, cat_mask, dog_img, dog_mask)
Image.fromarray(comp_img).save(out_dir / "test_composed_prompt_4.png")
print("Saved composed prompt 4 image!")
