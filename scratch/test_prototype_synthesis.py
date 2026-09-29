import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

# Load images
images_dir = Path("demos/data/images")
cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

print("Input image shapes:", cat_img.shape, dog_img.shape, nat_img.shape)

# Let's inspect segmentation / saliency for cat and dog
def extract_subject_mask(img_arr, method="color_contrast"):
    """Extract foreground subject mask from image."""
    h, w, _ = img_arr.shape
    # Corner samples as background estimate
    corners = np.concatenate([
        img_arr[:50, :50].reshape(-1, 3),
        img_arr[:50, -50:].reshape(-1, 3),
    ], axis=0)
    bg_mean = np.median(corners, axis=0)
    
    # Color distance from background
    diff = np.linalg.norm(img_arr.astype(np.float32) - bg_mean[None, None, :], axis=2)
    # Normalized [0, 1]
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    
    # Distance from center prior (subjects are usually centered)
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist_center = np.sqrt(((y - cy)/(h/2))**2 + ((x - cx)/(w/2))**2)
    center_weight = np.clip(1.0 - 0.5 * dist_center, 0.0, 1.0)
    
    saliency = diff * center_weight
    mask = (saliency > np.percentile(saliency, 40)).astype(np.float32)
    
    # Smooth mask
    import scipy.ndimage as ndi
    mask_smooth = ndi.gaussian_filter(mask, sigma=8.0)
    return mask_smooth

cat_mask = extract_subject_mask(cat_img)
dog_mask = extract_subject_mask(dog_img)
print("Masks computed: cat mask mean =", cat_mask.mean(), "dog mask mean =", dog_mask.mean())
