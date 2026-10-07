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
import scipy.ndimage as ndi
from pathlib import Path

img_dir = Path("demos/data/images")
cat_img = np.array(Image.open(img_dir / "cat.jpg").convert("RGB"))
dog_img = np.array(Image.open(img_dir / "dog.jpg").convert("RGB"))

def segment_subject_ml(img_arr: np.ndarray, label: str) -> np.ndarray:
    h, w, _ = img_arr.shape
    # Feature vector: R, G, B, and relative coordinates y, x
    y_coords, x_coords = np.mgrid[:h, :w]
    y_norm = (y_coords / h).astype(np.float32)
    x_norm = (x_coords / w).astype(np.float32)
    
    # RGB normalized
    rgb_f = img_arr.astype(np.float32) / 255.0
    
    # Definite background samples: top borders and far left/right borders
    bg_mask = np.zeros((h, w), dtype=bool)
    bg_mask[:int(h*0.12), :] = True
    bg_mask[:, :int(w*0.08)] = True
    bg_mask[:, int(w*0.92):] = True
    
    # Definite foreground samples: center core of the subject
    fg_mask = np.zeros((h, w), dtype=bool)
    if label == "cat":
        fg_mask[int(h*0.28):int(h*0.65), int(w*0.35):int(w*0.65)] = True
    else: # dog
        fg_mask[int(h*0.25):int(h*0.65), int(w*0.38):int(w*0.62)] = True
        
    c_max = np.max(rgb_f, axis=2)
    c_min = np.min(rgb_f, axis=2)
    sat = ((c_max - c_min) / (c_max + 1e-4))[:, :, None]
    feat_img = np.concatenate([rgb_f, sat], axis=2)

    bg_feats = feat_img[bg_mask]
    fg_feats = feat_img[fg_mask]
    
    fg_mean = np.mean(fg_feats, axis=0)
    bg_mean = np.mean(bg_feats, axis=0)
    
    fg_cov = np.cov(fg_feats.T) + np.eye(4) * 1e-4
    bg_cov = np.cov(bg_feats.T) + np.eye(4) * 1e-4
    
    inv_fg = np.linalg.inv(fg_cov)
    inv_bg = np.linalg.inv(bg_cov)
    
    flat_feats = feat_img.reshape(-1, 4)
    diff_fg = flat_feats - fg_mean
    mahal_fg = np.sum((diff_fg @ inv_fg) * diff_fg, axis=1)
    
    diff_bg = flat_feats - bg_mean
    mahal_bg = np.sum((diff_bg @ inv_bg) * diff_bg, axis=1)
    
    log_det_fg = np.linalg.slogdet(fg_cov)[1]
    log_det_bg = np.linalg.slogdet(bg_cov)[1]
    
    score = (mahal_bg + log_det_bg) - (mahal_fg + log_det_fg)
    score_map = score.reshape(h, w)
    
    # Spatial prior: subjects are located within central area
    cx, cy = 0.5, 0.48 if label == "cat" else 0.50
    rx, ry = 0.46, 0.52 if label == "cat" else 0.48
    spatial_dist = np.sqrt(((x_norm - cx)/rx)**2 + ((y_norm - cy)/ry)**2)
    spatial_weight = np.clip(1.0 - (spatial_dist / 1.05)**2, 0.0, 1.0)
    
    prob_map = 1.0 / (1.0 + np.exp(-score_map * 0.15))
    final_prob = prob_map * spatial_weight
    
    # Binary threshold and morphology
    binary = final_prob > 0.4
    binary = ndi.binary_fill_holes(binary)
    binary = ndi.binary_closing(binary, structure=np.ones((15, 15)))
    binary = ndi.binary_opening(binary, structure=np.ones((7, 7)))
    
    # Largest connected component
    labeled, num_features = ndi.label(binary)
    if num_features > 0:
        sizes = ndi.sum(binary, labeled, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        binary = (labeled == max_label)
        
    binary = ndi.binary_fill_holes(binary)
    feathered = ndi.gaussian_filter(binary.astype(np.float32), sigma=3.0)
    core = ndi.binary_erosion(binary, structure=np.ones((9, 9)))
    feathered[core] = 1.0
    return np.clip(feathered, 0.0, 1.0)

cat_mask = segment_subject_ml(cat_img, "cat")
dog_mask = segment_subject_ml(dog_img, "dog")

# Cutouts
cat_cut = (cat_img.astype(np.float32) * cat_mask[:, :, None]).clip(0, 255).astype(np.uint8)
dog_cut = (dog_img.astype(np.float32) * dog_mask[:, :, None]).clip(0, 255).astype(np.uint8)

Image.fromarray(cat_cut).save("scratch/cat_ml_cutout.png")
Image.fromarray(dog_cut).save("scratch/dog_ml_cutout.png")
print("Saved ML cutouts!")
