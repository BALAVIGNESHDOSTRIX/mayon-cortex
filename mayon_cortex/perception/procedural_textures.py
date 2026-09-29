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

"""
Procedural Textures — Mathematical Texture Synthesis & Surface Material Generation
===================================================================================
Pure NumPy procedural generation of photorealistic textures and surface patterns.

Implements:
1. Perlin & Gradient Noise
2. Voronoi / Worley Cellular Noise
3. Fractional Brownian Motion (fBm) & Turbulence
4. Natural Materials: Marble, Wood, Clouds, Brushed Metal, Brick, Fabric
"""

import math
from typing import Optional, Tuple
import numpy as np


class ProceduralTextureEngine:
    """
    Mathematical texture synthesis engine generating high-resolution realistic materials.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Precomputed permutation table for Perlin noise
        p = np.arange(256, dtype=np.int32)
        self.rng.shuffle(p)
        self.perm = np.tile(p, 2)

    def perlin_noise_2d(self, width: int, height: int, scale: float = 10.0) -> np.ndarray:
        """Generate 2D continuous Perlin noise in range [0, 1]."""
        x = np.linspace(0, scale, width, endpoint=False)
        y = np.linspace(0, scale, height, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        xi = xx.astype(np.int32) & 255
        yi = yy.astype(np.int32) & 255

        xf = xx - xx.astype(np.int32)
        yf = yy - yy.astype(np.int32)

        # Fade curves (quintic polynomial: 6t^5 - 15t^4 + 10t^3)
        u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        # Hash coordinates of the 4 square corners
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        # Dot product with pseudo-random gradients
        def grad(hash_val, dx, dy):
            h = hash_val & 3
            gx = np.where(h & 1, -dx, dx)
            gy = np.where(h & 2, -dy, dy)
            return gx + gy

        g_aa = grad(aa, xf, yf)
        g_ba = grad(ba, xf - 1.0, yf)
        g_ab = grad(ab, xf, yf - 1.0)
        g_bb = grad(bb, xf - 1.0, yf - 1.0)

        # Bilinear interpolation
        x1 = g_aa + u * (g_ba - g_aa)
        x2 = g_ab + u * (g_bb - g_ab)
        val = x1 + v * (x2 - x1)

        # Normalize to [0, 1]
        return np.clip((val + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    def fbm_noise_2d(
        self,
        width: int,
        height: int,
        scale: float = 6.0,
        octaves: int = 5,
        lacunarity: float = 2.0,
        gain: float = 0.5,
    ) -> np.ndarray:
        """Fractional Brownian Motion multi-octave noise."""
        total = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_val = 0.0

        for _ in range(octaves):
            total += amplitude * self.perlin_noise_2d(width, height, scale=frequency)
            max_val += amplitude
            amplitude *= gain
            frequency *= lacunarity

        return (total / max(1e-6, max_val)).astype(np.float32)

    def voronoi_cells(self, width: int, height: int, num_cells: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Voronoi / Worley cellular distance field (F1 nearest, F2 second-nearest).
        """
        pts_x = self.rng.uniform(0, width, num_cells).astype(np.float32)
        pts_y = self.rng.uniform(0, height, num_cells).astype(np.float32)

        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)

        f1 = np.full((height, width), 1e6, dtype=np.float32)
        f2 = np.full((height, width), 1e6, dtype=np.float32)

        for i in range(num_cells):
            d = np.sqrt((x_grid - pts_x[i]) ** 2 + (y_grid - pts_y[i]) ** 2)
            mask1 = d < f1
            f2 = np.where(mask1, f1, np.minimum(f2, d))
            f1 = np.where(mask1, d, f1)

        f1_norm = f1 / (np.max(f1) + 1e-6)
        border_field = (f2 - f1) / (np.max(f2 - f1) + 1e-6)
        return f1_norm, border_field

    def marble_texture(
        self,
        width: int,
        height: int,
        base_color: Tuple[int, int, int] = (240, 235, 230),
        vein_color: Tuple[int, int, int] = (60, 50, 45),
        scale: float = 4.0,
        turbulence_power: float = 6.0,
    ) -> np.ndarray:
        """Generate photorealistic marble surface with organic chaotic veins."""
        x = np.linspace(0, scale, width, endpoint=False)
        y = np.linspace(0, scale, height, endpoint=False)
        xx, yy = np.meshgrid(x, y)

        turb = self.fbm_noise_2d(width, height, scale=scale * 2, octaves=5)
        pattern = np.sin((xx + yy) * np.pi + turbulence_power * turb)
        pattern = (pattern + 1.0) * 0.5  # [0, 1]

        # Non-linear sharp vein falloff
        veins = np.power(pattern, 4.0)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            ch = base_color[c] * (1.0 - veins) + vein_color[c] * veins
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return img

    def wood_texture(
        self,
        width: int,
        height: int,
        wood_light: Tuple[int, int, int] = (205, 133, 63),
        wood_dark: Tuple[int, int, int] = (101, 45, 15),
        rings_scale: float = 12.0,
        turbulence: float = 2.5,
    ) -> np.ndarray:
        """Generate organic grain and tree ring cross-section wood texture."""
        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width * 0.3, height * 0.3
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / float(max(width, height))

        turb = self.fbm_noise_2d(width, height, scale=5.0, octaves=4)
        grain = np.sin(dist * rings_scale * 2 * np.pi + turbulence * turb)
        grain = (grain + 1.0) * 0.5

        img = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            ch = wood_light[c] * (1.0 - grain) + wood_dark[c] * grain
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return img

    def clouds_sky(
        self,
        width: int,
        height: int,
        sky_blue: Tuple[int, int, int] = (70, 140, 230),
        cloud_white: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Generate soft realistic atmospheric cloudscape."""
        cloud_density = self.fbm_noise_2d(width, height, scale=3.5, octaves=6, gain=0.55)
        cloud_density = np.clip((cloud_density - 0.35) / 0.5, 0.0, 1.0)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            ch = sky_blue[c] * (1.0 - cloud_density) + cloud_white[c] * cloud_density
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        return img
