import numpy as np
from PIL import Image

def create_feathered_window(patch_h, patch_w, overlap_h, overlap_w):
    """
    Create a 2D weight window with 1.0 in the center and smooth cosine
    ramps ONLY in the overlap margin.
    """
    wy = np.ones(patch_h, dtype=np.float32)
    if overlap_h > 0:
        # Half cosine ramp from 0 to 1 in [0, overlap_h]
        t = np.arange(overlap_h, dtype=np.float32) / overlap_h
        ramp_up = 0.5 - 0.5 * np.cos(np.pi * t)
        wy[:overlap_h] = ramp_up
        wy[-overlap_h:] = ramp_up[::-1]
        
    wx = np.ones(patch_w, dtype=np.float32)
    if overlap_w > 0:
        t = np.arange(overlap_w, dtype=np.float32) / overlap_w
        ramp_up = 0.5 - 0.5 * np.cos(np.pi * t)
        wx[:overlap_w] = ramp_up
        wx[-overlap_w:] = ramp_up[::-1]
        
    w2d = (wy[:, None] * wx[None, :])[:, :, None]
    return np.maximum(w2d, 1e-4)

# Test overlap
w = create_feathered_window(32, 32, 6, 6)
print("Center weight:", w[16, 16, 0])
print("Corner weight:", w[0, 0, 0])
print("Edge weight at margin:", w[6, 16, 0])
