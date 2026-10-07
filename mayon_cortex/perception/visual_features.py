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

"""
Visual Feature Extractor — Pure Mathematical Computer Vision Features
======================================================================
Extracts 256-dimensional descriptive feature vectors from image patches
without neural networks or GPUs.

Feature Decomposition:
1. HOG (Histogram of Oriented Gradients):          36-dim (Shape, edges, contours)
2. Color Histogram (HSV Space):                  48-dim (Color distribution)
3. LBP (Local Binary Patterns):                  28-dim (Micro-textures, grain)
4. Gabor Filter Bank Energy:                     16-dim (Multi-scale orientation)
5. Spatial Edge Density (Harris Response 4x4):   16-dim (Sub-cell corner/edge density)
6. Dominant Color Palette (Mini K-Means LAB):    24-dim (Top-3 color centroids & stats)
7. DCT Frequency Energy (4 Quadrants x 4 Bands): 16-dim (Multi-scale spatial frequency)
8. Gradient Magnitude Distribution (Percentiles): 16-dim (Channel contrast dynamics)
9. Edge Orientation Coherence (Circular Var):    16-dim (Angular uniformity in 4x4 cells)
10. Fractal Dimension Estimate (Box-Counting):   16-dim (Structural self-similarity)
11. Multi-Scale Saliency & Contrast:             24-dim (Center-surround feature dynamics)
----------------------------------------------------------------------------------------
Total Feature Vector:                           256-dim (L2 normalized)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB or RGBA image to grayscale [0, 255] float32."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] >= 3:
        # Standard ITU-R BT.601 luma transform
        return (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        ).astype(np.float32)
    return image[:, :, 0].astype(np.float32)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image [0, 255] to HSV [0, 1] float32 using vectorized NumPy.
    """
    img = image[..., :3].astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue calculation
    h = np.zeros_like(cmax)
    mask_delta = delta > 1e-7

    # When r is max
    mask_r = (cmax == r) & mask_delta
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0

    # When g is max
    mask_g = (cmax == g) & mask_delta
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0

    # When b is max
    mask_b = (cmax == b) & mask_delta
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0

    h = (h / 6.0) % 1.0

    # Saturation calculation
    s = np.zeros_like(cmax)
    mask_max = cmax > 1e-7
    s[mask_max] = delta[mask_max] / cmax[mask_max]

    # Value calculation
    v = cmax

    return np.stack([h, s, v], axis=-1)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image [0, 255] to CIELAB space using vectorized NumPy.
    Standard D65 illuminant reference.
    """
    rgb = image[..., :3].astype(np.float32) / 255.0

    # Gamma correction to linear sRGB
    mask = rgb > 0.04045
    rgb_linear = np.zeros_like(rgb)
    rgb_linear[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_linear[~mask] = rgb[~mask] / 12.92

    # sRGB to XYZ (D65)
    r = rgb_linear[..., 0]
    g = rgb_linear[..., 1]
    b = rgb_linear[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x /= xn
    y /= yn
    z /= zn

    # Non-linear transform
    delta = 6.0 / 29.0
    fx = np.where(x > delta**3, x ** (1.0 / 3.0), (x / (3.0 * delta**2)) + (4.0 / 29.0))
    fy = np.where(y > delta**3, y ** (1.0 / 3.0), (y / (3.0 * delta**2)) + (4.0 / 29.0))
    fz = np.where(z > delta**3, z ** (1.0 / 3.0), (z / (3.0 * delta**2)) + (4.0 / 29.0))

    L = 116.0 * fy - 16.0
    A = 500.0 * (fx - fy)
    B = 200.0 * (fy - fz)

    return np.stack([L, A, B], axis=-1)


@dataclass
class PatchInfo:
    """Metadata and content for a single extracted image patch."""
    row_idx: int
    col_idx: int
    y_start: int
    x_start: int
    patch: np.ndarray
    features: Optional[np.ndarray] = None


class VisualFeatureExtractor:
    """
    Pure CPU-native visual feature extraction engine.
    Computes invariant 256-dim descriptors capturing shape, color, texture, scale,
    geometry, frequency dynamics, and structural complexity.
    """

    def __init__(self, hog_bins: int = 9, color_bins: int = 16):
        self.hog_bins = hog_bins
        self.color_bins = color_bins
        self._gabor_kernels = self._create_gabor_bank()
        self._dct_matrices = {}

    def _create_gabor_bank(self) -> List[np.ndarray]:
        """Pre-compute 16 Gabor filter kernels (4 orientations x 4 scales)."""
        kernels = []
        orientations = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
        wavelengths = [3.0, 6.0, 10.0, 16.0]
        gamma = 0.5   # Spatial aspect ratio
        psi = 0.0     # Phase offset

        for theta in orientations:
            for lambd in wavelengths:
                sigma = 0.56 * lambd
                kernel_size = int(max(7, int(3 * sigma) * 2 + 1))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = min(kernel_size, 21)

                half = kernel_size // 2
                y, x = np.mgrid[-half:half + 1, -half:half + 1]

                x_theta = x * np.cos(theta) + y * np.sin(theta)
                y_theta = -x * np.sin(theta) + y * np.cos(theta)

                gb = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta)**2) / (sigma**2)) * np.cos(2 * np.pi * x_theta / lambd + psi)
                # Zero-mean kernel
                gb -= gb.mean()
                norm = np.linalg.norm(gb)
                if norm > 1e-7:
                    gb /= norm
                kernels.append(gb.astype(np.float32))

        return kernels

    def hog_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 36-dimensional HOG feature vector for a patch.
        Uses 2x2 cells with 9 orientation bins each (0 - 180 deg).
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape

        if h < 4 or w < 4:
            return np.zeros(36, dtype=np.float32)

        # Gradient via Sobel / central difference
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gx[:, 0] = gray[:, 1] - gray[:, 0]
        gx[:, -1] = gray[:, -1] - gray[:, -2]

        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        gy[0, :] = gray[1, :] - gray[0, :]
        gy[-1, :] = gray[-1, :] - gray[-2, :]

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.arctan2(gy, gx) % np.pi) * (180.0 / np.pi)  # [0, 180)

        mid_y, mid_x = h // 2, w // 2
        cells = [
            (slice(0, mid_y), slice(0, mid_x)),
            (slice(0, mid_y), slice(mid_x, w)),
            (slice(mid_y, h), slice(0, mid_x)),
            (slice(mid_y, h), slice(mid_x, w)),
        ]

        bin_width = 180.0 / self.hog_bins
        histograms = []

        for y_slice, x_slice in cells:
            cell_mag = magnitude[y_slice, x_slice]
            cell_ori = orientation[y_slice, x_slice]

            hist = np.zeros(self.hog_bins, dtype=np.float32)
            if cell_mag.size > 0:
                bin_indices = (cell_ori / bin_width).astype(np.int32) % self.hog_bins
                for b in range(self.hog_bins):
                    mask = (bin_indices == b)
                    if np.any(mask):
                        hist[b] = np.sum(cell_mag[mask])
            histograms.append(hist)

        hog_raw = np.concatenate(histograms)  # 36 dims

        norm = np.linalg.norm(hog_raw) + 1e-6
        hog_norm = hog_raw / norm
        hog_clipped = np.clip(hog_norm, 0.0, 0.2)
        final_norm = np.linalg.norm(hog_clipped) + 1e-6
        return (hog_clipped / final_norm).astype(np.float32)

    def color_histogram(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 48-dimensional normalized HSV color histogram (16 bins per channel).
        """
        if patch.ndim == 2:
            v_hist, _ = np.histogram(patch / 255.0, bins=self.color_bins, range=(0.0, 1.0))
            v_hist = v_hist.astype(np.float32)
            norm = np.linalg.norm(v_hist) + 1e-6
            v_norm = v_hist / norm
            return np.concatenate([np.zeros(32, dtype=np.float32), v_norm])

        hsv = rgb_to_hsv(patch)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        h_hist, _ = np.histogram(h_channel, bins=self.color_bins, range=(0.0, 1.0))
        s_hist, _ = np.histogram(s_channel, bins=self.color_bins, range=(0.0, 1.0))
        v_hist, _ = np.histogram(v_channel, bins=self.color_bins, range=(0.0, 1.0))

        h_norm = h_hist.astype(np.float32) / (np.linalg.norm(h_hist) + 1e-6)
        s_norm = s_hist.astype(np.float32) / (np.linalg.norm(s_hist) + 1e-6)
        v_norm = v_hist.astype(np.float32) / (np.linalg.norm(v_hist) + 1e-6)

        color_feat = np.concatenate([h_norm, s_norm, v_norm])
        total_norm = np.linalg.norm(color_feat) + 1e-6
        return (color_feat / total_norm).astype(np.float32)

    def lbp_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 28-dimensional Local Binary Pattern histogram for micro-texture.
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape

        if h < 3 or w < 3:
            return np.zeros(28, dtype=np.float32)

        center = gray[1:-1, 1:-1]

        n0 = gray[1:-1, 2:] >= center
        n1 = gray[:-2, 2:] >= center
        n2 = gray[:-2, 1:-1] >= center
        n3 = gray[:-2, :-2] >= center
        n4 = gray[1:-1, :-2] >= center
        n5 = gray[2:, :-2] >= center
        n6 = gray[2:, 1:-1] >= center
        n7 = gray[2:, 2:] >= center

        pattern = (
            (n0.astype(np.int32) << 0) |
            (n1.astype(np.int32) << 1) |
            (n2.astype(np.int32) << 2) |
            (n3.astype(np.int32) << 3) |
            (n4.astype(np.int32) << 4) |
            (n5.astype(np.int32) << 5) |
            (n6.astype(np.int32) << 6) |
            (n7.astype(np.int32) << 7)
        )

        hist, _ = np.histogram(pattern, bins=28, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist) + 1e-6
        return (hist / norm).astype(np.float32)

    def gabor_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional Gabor filter bank response energy.
        """
        gray = rgb_to_gray(patch)
        sub_gray = gray[::2, ::2] if gray.shape[0] >= 16 else gray
        energies = np.zeros(len(self._gabor_kernels), dtype=np.float32)

        try:
            import scipy.ndimage as ndi
            for i, kernel in enumerate(self._gabor_kernels):
                sub_k = kernel[::2, ::2] if kernel.shape[0] >= 7 else kernel
                resp = ndi.correlate(sub_gray, sub_k, mode="reflect")
                energies[i] = np.sqrt(np.mean(resp**2))
        except ImportError:
            for i, kernel in enumerate(self._gabor_kernels):
                sub_k = kernel[::2, ::2]
                min_h = min(sub_gray.shape[0], sub_k.shape[0])
                min_w = min(sub_gray.shape[1], sub_k.shape[1])
                resp = np.sum(sub_gray[:min_h, :min_w] * sub_k[:min_h, :min_w])
                energies[i] = np.abs(resp)

        norm = np.linalg.norm(energies) + 1e-6
        return (energies / norm).astype(np.float32)

    def spatial_edge_density(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional Spatial Edge Density (Harris corner response / edge strength)
        in a 4x4 sub-grid.
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape
        if h < 4 or w < 4:
            return np.zeros(16, dtype=np.float32)

        # Gradients
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        ix2 = gx ** 2
        iy2 = gy ** 2
        ixy = gx * gy

        # 4x4 sub-grid cells
        row_bounds = np.linspace(0, h, 5, dtype=int)
        col_bounds = np.linspace(0, w, 5, dtype=int)

        responses = np.zeros(16, dtype=np.float32)
        k = 0.04
        idx = 0
        for i in range(4):
            r_start, r_end = row_bounds[i], row_bounds[i + 1]
            for j in range(4):
                c_start, c_end = col_bounds[j], col_bounds[j + 1]
                sx2 = np.mean(ix2[r_start:r_end, c_start:c_end]) if r_end > r_start and c_end > c_start else 0.0
                sy2 = np.mean(iy2[r_start:r_end, c_start:c_end]) if r_end > r_start and c_end > c_start else 0.0
                sxy = np.mean(ixy[r_start:r_end, c_start:c_end]) if r_end > r_start and c_end > c_start else 0.0

                det = (sx2 * sy2) - (sxy ** 2)
                trace = sx2 + sy2
                harris_resp = det - k * (trace ** 2)
                responses[idx] = max(0.0, float(harris_resp))
                idx += 1

        norm = np.linalg.norm(responses) + 1e-6
        return (responses / norm).astype(np.float32)

    def dominant_color_palette(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 24-dimensional Dominant Color Palette in LAB color space via mini k-means.
        Top-3 clusters x 8 descriptors = 24 dimensions.
        Descriptors per cluster: [L, A, B, weight, std_L, std_A, std_B, radius]
        """
        if patch.ndim == 2 or patch.shape[-1] < 3:
            p_rgb = np.stack([patch, patch, patch], axis=-1) if patch.ndim == 2 else patch
        else:
            p_rgb = patch[..., :3]

        lab = rgb_to_lab(p_rgb)
        h, w, _ = lab.shape
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # Normalize LAB channels: L in [0, 100], A in [-128, 127], B in [-128, 127]
        pixels[:, 0] /= 100.0
        pixels[:, 1] = (pixels[:, 1] + 128.0) / 255.0
        pixels[:, 2] = (pixels[:, 2] + 128.0) / 255.0

        n_pixels = len(pixels)
        k = 3

        # Subsample if large patch for speed
        if n_pixels > 256:
            step = n_pixels // 256
            samples = pixels[::step]
        else:
            samples = pixels

        # Fast mini k-means
        rng = np.random.RandomState(42)
        n_samples = len(samples)
        if n_samples < k:
            centroids = np.zeros((k, 3), dtype=np.float32)
            centroids[:n_samples] = samples
            weights = np.ones(k, dtype=np.float32) / k
            stds = np.zeros((k, 4), dtype=np.float32)
        else:
            # Deterministic init from percentiles
            centroids = np.array([
                np.percentile(samples, 20, axis=0),
                np.percentile(samples, 50, axis=0),
                np.percentile(samples, 80, axis=0),
            ], dtype=np.float32)

            for _ in range(5):
                dists = np.sum((samples[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                assigns = np.argmin(dists, axis=1)
                for ci in range(k):
                    c_mask = (assigns == ci)
                    if np.any(c_mask):
                        centroids[ci] = np.mean(samples[c_mask], axis=0)

            # Assign all pixels
            dists_all = np.sum((pixels[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assigns_all = np.argmin(dists_all, axis=1)

            # Sort clusters by size descending
            counts = np.bincount(assigns_all, minlength=k)
            sorted_indices = np.argsort(-counts)

            cluster_descriptors = []
            for ci in sorted_indices:
                weight = float(counts[ci]) / float(n_pixels)
                c_mask = (assigns_all == ci)
                if np.any(c_mask):
                    c_pixels = pixels[c_mask]
                    c_std = np.std(c_pixels, axis=0)
                    spread = float(np.mean(np.linalg.norm(c_pixels - centroids[ci], axis=1)))
                else:
                    c_std = np.zeros(3, dtype=np.float32)
                    spread = 0.0

                desc = [
                    centroids[ci, 0], centroids[ci, 1], centroids[ci, 2],
                    weight,
                    c_std[0], c_std[1], c_std[2],
                    spread,
                ]
                cluster_descriptors.extend(desc)

            palette_feat = np.array(cluster_descriptors, dtype=np.float32)
            norm = np.linalg.norm(palette_feat) + 1e-6
            return (palette_feat / norm).astype(np.float32)

        feat = np.zeros(24, dtype=np.float32)
        return feat

    def dct_frequency_energy(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional DCT frequency band energy.
        4 quadrants x 4 frequency bands (DC/low, mid-low, mid-high, high) = 16 dimensions.
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape

        if h < 8 or w < 8:
            return np.zeros(16, dtype=np.float32)

        mid_y, mid_x = h // 2, w // 2
        quadrants = [
            gray[:mid_y, :mid_x],
            gray[:mid_y, mid_x:],
            gray[mid_y:, :mid_x],
            gray[mid_y:, mid_x:],
        ]

        def _get_dct_matrix(n: int) -> np.ndarray:
            if n not in self._dct_matrices:
                i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
                c = np.cos(np.pi * (2 * j + 1) * i / (2.0 * n))
                c[0, :] *= 1.0 / np.sqrt(n)
                c[1:, :] *= np.sqrt(2.0 / n)
                self._dct_matrices[n] = c.astype(np.float32)
            return self._dct_matrices[n]

        energies = []
        for quad in quadrants:
            qh, qw = quad.shape
            min_dim = min(qh, qw)
            q_sq = quad[:min_dim, :min_dim]
            c_mat = _get_dct_matrix(min_dim)
            # 2D DCT = C * q * C.T
            dct_2d = np.abs(np.dot(np.dot(c_mat, q_sq), c_mat.T))

            # 4 radial frequency bands
            r, c = np.indices((min_dim, min_dim))
            radius = np.sqrt(r**2 + c**2)
            max_r = np.sqrt(2.0) * min_dim

            b1 = (radius <= 0.15 * max_r)
            b2 = (radius > 0.15 * max_r) & (radius <= 0.35 * max_r)
            b3 = (radius > 0.35 * max_r) & (radius <= 0.65 * max_r)
            b4 = (radius > 0.65 * max_r)

            e1 = np.log1p(np.mean(dct_2d[b1])) if np.any(b1) else 0.0
            e2 = np.log1p(np.mean(dct_2d[b2])) if np.any(b2) else 0.0
            e3 = np.log1p(np.mean(dct_2d[b3])) if np.any(b3) else 0.0
            e4 = np.log1p(np.mean(dct_2d[b4])) if np.any(b4) else 0.0

            energies.extend([e1, e2, e3, e4])

        feat = np.array(energies, dtype=np.float32)
        norm = np.linalg.norm(feat) + 1e-6
        return (feat / norm).astype(np.float32)

    def gradient_magnitude_distribution(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional Gradient Magnitude Distribution.
        Percentiles (10th, 25th, 50th, 75th, 90th) x 3 channels (15 dims) + global stat (1 dim) = 16 dimensions.
        """
        if patch.ndim == 2:
            img = np.stack([patch, patch, patch], axis=-1)
        elif patch.shape[-1] < 3:
            img = np.repeat(patch, 3, axis=-1)
        else:
            img = patch[..., :3].astype(np.float32)

        channel_stats = []
        all_mags = []
        percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]

        for ch in range(3):
            plane = img[:, :, ch]
            gx = np.zeros_like(plane)
            gy = np.zeros_like(plane)
            gx[:, 1:-1] = plane[:, 2:] - plane[:, :-2]
            gy[1:-1, :] = plane[2:, :] - plane[:-2, :]
            mag = np.sqrt(gx**2 + gy**2)
            all_mags.append(mag.ravel())
            pcts = np.percentile(mag, percentiles)
            channel_stats.extend(pcts)

        combined_mag = np.concatenate(all_mags)
        global_std = float(np.std(combined_mag))
        channel_stats.append(global_std)

        feat = np.array(channel_stats, dtype=np.float32)
        norm = np.linalg.norm(feat) + 1e-6
        return (feat / norm).astype(np.float32)

    def edge_orientation_coherence(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional Edge Orientation Coherence across 4x4 sub-regions.
        Circular variance / orientation alignment metric per cell.
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape
        if h < 4 or w < 4:
            return np.zeros(16, dtype=np.float32)

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        mag = np.sqrt(gx**2 + gy**2)
        theta = np.arctan2(gy, gx)  # [-pi, pi]

        # Double angle representation for orientation invariance
        c2 = np.cos(2.0 * theta) * mag
        s2 = np.sin(2.0 * theta) * mag

        row_bounds = np.linspace(0, h, 5, dtype=int)
        col_bounds = np.linspace(0, w, 5, dtype=int)

        coherences = np.zeros(16, dtype=np.float32)
        idx = 0
        for i in range(4):
            r_start, r_end = row_bounds[i], row_bounds[i + 1]
            for j in range(4):
                c_start, c_end = col_bounds[j], col_bounds[j + 1]
                sub_c2 = c2[r_start:r_end, c_start:c_end]
                sub_s2 = s2[r_start:r_end, c_start:c_end]
                sub_mag = mag[r_start:r_end, c_start:c_end]
                total_mag = np.sum(sub_mag)

                if total_mag > 1e-5:
                    r_len = np.sqrt(np.sum(sub_c2)**2 + np.sum(sub_s2)**2) / total_mag
                    coherences[idx] = float(np.clip(r_len, 0.0, 1.0))
                else:
                    coherences[idx] = 0.0
                idx += 1

        norm = np.linalg.norm(coherences) + 1e-6
        return (coherences / norm).astype(np.float32)

    def fractal_dimension_estimate(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 16-dimensional Fractal Dimension Estimate via multi-scale box-counting.
        4 binary threshold levels x 4 box sizes = 16 dimensions.
        """
        gray = rgb_to_gray(patch)
        h, w = gray.shape
        if h < 8 or w < 8:
            return np.zeros(16, dtype=np.float32)

        min_val, max_val = float(np.min(gray)), float(np.max(gray))
        if max_val - min_val < 1e-4:
            return np.zeros(16, dtype=np.float32)

        thresholds = [
            min_val + 0.20 * (max_val - min_val),
            min_val + 0.40 * (max_val - min_val),
            min_val + 0.60 * (max_val - min_val),
            min_val + 0.80 * (max_val - min_val),
        ]

        box_sizes = [2, 4, 8, 16]
        counts = []

        for th in thresholds:
            binary = (gray >= th)
            for bs in box_sizes:
                bh = (h // bs) * bs
                bw = (w // bs) * bs
                if bh < bs or bw < bs:
                    counts.append(0.0)
                    continue

                sub_b = binary[:bh, :bw]
                grid_cells = sub_b.reshape(bh // bs, bs, bw // bs, bs)
                # Count non-empty boxes
                has_pixels = np.any(grid_cells, axis=(1, 3))
                box_count = float(np.sum(has_pixels))
                max_boxes = float((bh // bs) * (bw // bs))
                ratio = box_count / max(1.0, max_boxes)
                counts.append(ratio)

        feat = np.array(counts, dtype=np.float32)
        norm = np.linalg.norm(feat) + 1e-6
        return (feat / norm).astype(np.float32)

    def multi_scale_saliency(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute 24-dimensional Multi-Scale Saliency & Center-Surround Contrast.
        3 perceptual channels (Intensity, Red-Green, Blue-Yellow) x 8 directional contrasts = 24 dimensions.
        """
        if patch.ndim == 2:
            p_rgb = np.stack([patch, patch, patch], axis=-1).astype(np.float32)
        else:
            p_rgb = patch[..., :3].astype(np.float32)

        r = p_rgb[..., 0] / 255.0
        g = p_rgb[..., 1] / 255.0
        b = p_rgb[..., 2] / 255.0

        intensity = (r + g + b) / 3.0
        rg = np.abs(r - g)
        by = np.abs(b - 0.5 * (r + g))

        h, w = intensity.shape
        channels = [intensity, rg, by]
        descriptors = []

        # Center vs 8 boundary segments
        cy_start, cy_end = h // 4, 3 * (h // 4)
        cx_start, cx_end = w // 4, 3 * (w // 4)

        for ch in channels:
            center_val = np.mean(ch[cy_start:cy_end, cx_start:cx_end]) if cy_end > cy_start and cx_end > cx_start else np.mean(ch)
            
            # 8 outer sectors
            s_top = np.mean(ch[:h//4, :])
            s_bottom = np.mean(ch[3*h//4:, :])
            s_left = np.mean(ch[:, :w//4])
            s_right = np.mean(ch[:, 3*w//4:])
            s_tl = np.mean(ch[:h//4, :w//4])
            s_tr = np.mean(ch[:h//4, 3*w//4:])
            s_bl = np.mean(ch[3*h//4:, :w//4])
            s_br = np.mean(ch[3*h//4:, 3*w//4:])

            surrounds = [s_top, s_bottom, s_left, s_right, s_tl, s_tr, s_bl, s_br]
            diffs = [float(abs(center_val - s)) for s in surrounds]
            descriptors.extend(diffs)

        feat = np.array(descriptors, dtype=np.float32)
        norm = np.linalg.norm(feat) + 1e-6
        return (feat / norm).astype(np.float32)

    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract complete 256-dim normalized feature vector from a single patch.
        [36 HOG + 48 Color + 28 LBP + 16 Gabor + 16 EdgeDensity + 24 ColorPalette +
         16 DCT + 16 GradDist + 16 OrientationCoherence + 16 Fractal + 24 Saliency] -> 256-dim
        """
        f_hog = self.hog_features(patch)
        f_color = self.color_histogram(patch)
        f_lbp = self.lbp_features(patch)
        f_gabor = self.gabor_features(patch)
        f_edge_density = self.spatial_edge_density(patch)
        f_color_palette = self.dominant_color_palette(patch)
        f_dct = self.dct_frequency_energy(patch)
        f_grad_dist = self.gradient_magnitude_distribution(patch)
        f_orientation_coherence = self.edge_orientation_coherence(patch)
        f_fractal = self.fractal_dimension_estimate(patch)
        f_saliency = self.multi_scale_saliency(patch)

        feat = np.concatenate([
            f_hog,
            f_color,
            f_lbp,
            f_gabor,
            f_edge_density,
            f_color_palette,
            f_dct,
            f_grad_dist,
            f_orientation_coherence,
            f_fractal,
            f_saliency,
        ]).astype(np.float32)

        norm = np.linalg.norm(feat) + 1e-7
        return feat / norm

    def decompose_image(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (32, 32),
        stride: Optional[Tuple[int, int]] = None,
    ) -> List[PatchInfo]:
        """
        Decompose an input image into a regular grid of patches.
        Extracts coordinates and prepares patches for feature processing.
        """
        if stride is None:
            stride = patch_size

        img = image
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        h, w = img.shape[:2]
        ph, pw = patch_size
        sh, sw = stride

        # Compute necessary padding to fit complete patches
        pad_bottom = (sh - ((h - ph) % sh)) % sh if h > ph else max(0, ph - h)
        pad_right = (sw - ((w - pw) % sw)) % sw if w > pw else max(0, pw - w)

        if pad_bottom > 0 or pad_right > 0:
            img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
            h, w = img.shape[:2]

        patches: List[PatchInfo] = []
        row_idx = 0
        for y in range(0, h - ph + 1, sh):
            col_idx = 0
            for x in range(0, w - pw + 1, sw):
                p_arr = img[y:y + ph, x:x + pw].copy()
                patches.append(
                    PatchInfo(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        y_start=y,
                        x_start=x,
                        patch=p_arr,
                    )
                )
                col_idx += 1
            row_idx += 1

        return patches
