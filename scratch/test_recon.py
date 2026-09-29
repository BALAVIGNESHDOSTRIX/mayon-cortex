import numpy as np
from PIL import Image
from pathlib import Path

images_dir = Path("demos/data/images")
out_dir = Path("demos/demo_vision_output/test_fix")
out_dir.mkdir(parents=True, exist_ok=True)

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
h, w, _ = cat_img.shape
ps = (32, 32)
ph, pw = ps

# Extract non-overlapping patches
rows = h // ph
cols = w // pw
patches = []
for r in range(rows):
    row_patches = []
    for c in range(cols):
        patch = cat_img[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
        row_patches.append(patch)
    patches.append(row_patches)

# Decode with old Hann window (stride_ratio=0.5)
def decode_hann(patches, stride_ratio=0.5):
    rows = len(patches)
    cols = len(patches[0])
    stride_y = int(ph * stride_ratio)
    stride_x = int(pw * stride_ratio)
    out_h = (rows - 1) * stride_y + ph
    out_w = (cols - 1) * stride_x + pw
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
    wy = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(ph) + 0.5) / ph)
    wx = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(pw) + 0.5) / pw)
    window_2d = (wy[:, None] * wx[None, :])[:, :, None].astype(np.float32)
    window_2d = np.maximum(window_2d, 0.05)
    for r in range(rows):
        for c in range(cols):
            patch = patches[r][c].astype(np.float32)
            y_start = r * stride_y
            x_start = c * stride_x
            canvas[y_start:y_start + ph, x_start:x_start + pw] += patch * window_2d
            weight_map[y_start:y_start + ph, x_start:x_start + pw] += window_2d
    reconstructed = canvas / (weight_map + 1e-6)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

# Decode with new feathered seam (retaining full patch crispness)
def decode_feathered(patches, overlap=4):
    rows = len(patches)
    cols = len(patches[0])
    stride_y = ph - overlap
    stride_x = pw - overlap
    out_h = (rows - 1) * stride_y + ph
    out_w = (cols - 1) * stride_x + pw
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
    
    wy = np.ones(ph, dtype=np.float32)
    t = np.arange(overlap, dtype=np.float32) / overlap
    ramp = 0.5 - 0.5 * np.cos(np.pi * t)
    wy[:overlap] = ramp
    wy[-overlap:] = ramp[::-1]
    
    wx = np.ones(pw, dtype=np.float32)
    wx[:overlap] = ramp
    wx[-overlap:] = ramp[::-1]
    
    w2d = (wy[:, None] * wx[None, :])[:, :, None]
    w2d = np.maximum(w2d, 1e-4)
    
    for r in range(rows):
        for c in range(cols):
            patch = patches[r][c].astype(np.float32)
            y_start = r * stride_y
            x_start = c * stride_x
            canvas[y_start:y_start + ph, x_start:x_start + pw] += patch * w2d
            weight_map[y_start:y_start + ph, x_start:x_start + pw] += w2d
    reconstructed = canvas / (weight_map + 1e-6)
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

img_hann = decode_hann(patches)
img_feather = decode_feathered(patches)

Image.fromarray(img_hann).save(out_dir / "recon_hann.png")
Image.fromarray(img_feather).save(out_dir / "recon_feather.png")
print("Saved comparison recon images!")
