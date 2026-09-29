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
Raymarcher Engine — Signed Distance Field (SDF) 3D Mathematical Photorealism
============================================================================
Pure CPU NumPy raymarcher rendering 3D geometric scenes, PBR materials,
soft shadows, ambient occlusion, and lighting.

Enables Mayon-Cortex to construct geometrically perfect 3D objects,
depth maps, and realistic physical lighting from semantic graph primitives.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class Material:
    """Surface material properties."""
    albedo: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.3
    specular: float = 0.5
    metallic: float = 0.0


@dataclass
class SDFObject:
    """An analytical 3D primitive defined by its Signed Distance Function."""
    name: str
    sdf_fn: Callable[[np.ndarray], float]
    material: Material = field(default_factory=Material)


class RaymarchRenderer:
    """
    High-performance pure NumPy sphere tracing raymarcher.
    """

    def __init__(self, max_steps: int = 64, max_dist: float = 20.0, eps: float = 0.002):
        self.max_steps = max_steps
        self.max_dist = max_dist
        self.eps = eps

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical Signed Distance Functions (SDFs)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def sdf_sphere(p: np.ndarray, center: np.ndarray, radius: float) -> float:
        """Signed distance to a 3D sphere."""
        return float(np.linalg.norm(p - center) - radius)

    @staticmethod
    def sdf_box(p: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
        """Signed distance to a 3D axis-aligned box."""
        d = np.abs(p - center) - size
        inside = min(max(d[0], max(d[1], d[2])), 0.0)
        outside = np.linalg.norm(np.maximum(d, 0.0))
        return float(inside + outside)

    @staticmethod
    def sdf_plane(p: np.ndarray, normal: np.ndarray, height: float) -> float:
        """Signed distance to an infinite plane."""
        n_norm = normal / (np.linalg.norm(normal) + 1e-7)
        return float(np.dot(p, n_norm) + height)

    @staticmethod
    def sdf_torus(p: np.ndarray, center: np.ndarray, r_major: float, r_minor: float) -> float:
        """Signed distance to a 3D torus."""
        q = p - center
        q_xz_len = np.sqrt(q[0] * q[0] + q[2] * q[2]) - r_major
        return float(np.sqrt(q_xz_len * q_xz_len + q[1] * q[1]) - r_minor)

    @staticmethod
    def smin(a: float, b: float, k: float = 0.1) -> float:
        """Smooth minimum operator for organic blending of shapes."""
        h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * (1.0 / 4.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Scene Distance & Normal Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def scene_sdf(self, p: np.ndarray, objects: List[SDFObject]) -> Tuple[float, Material]:
        """Compute the minimum signed distance to all objects in the scene."""
        min_dist = 1e6
        best_mat = Material()

        for obj in objects:
            d = obj.sdf_fn(p)
            if d < min_dist:
                min_dist = d
                best_mat = obj.material

        return min_dist, best_mat

    def calc_normal(self, p: np.ndarray, objects: List[SDFObject]) -> np.ndarray:
        """Estimate surface normal vector via central finite differences."""
        e = self.eps
        dx = np.array([e, 0.0, 0.0])
        dy = np.array([0.0, e, 0.0])
        dz = np.array([0.0, 0.0, e])

        nx = self.scene_sdf(p + dx, objects)[0] - self.scene_sdf(p - dx, objects)[0]
        ny = self.scene_sdf(p + dy, objects)[0] - self.scene_sdf(p - dy, objects)[0]
        nz = self.scene_sdf(p + dz, objects)[0] - self.scene_sdf(p - dz, objects)[0]

        n = np.array([nx, ny, nz], dtype=np.float32)
        norm = np.linalg.norm(n) + 1e-7
        return n / norm

    def calc_soft_shadow(
        self,
        ro: np.ndarray,
        rd: np.ndarray,
        objects: List[SDFObject],
        mint: float = 0.02,
        maxt: float = 10.0,
        k: float = 16.0,
    ) -> float:
        """Compute raymarched soft penumbra shadow factor [0, 1]."""
        res = 1.0
        t = mint
        for _ in range(24):
            if t > maxt:
                break
            d, _ = self.scene_sdf(ro + rd * t, objects)
            if d < 0.001:
                return 0.0
            res = min(res, k * d / t)
            t += max(0.02, d)
        return max(0.0, min(1.0, res))

    def calc_ao(self, pos: np.ndarray, nor: np.ndarray, objects: List[SDFObject]) -> float:
        """Compute ambient occlusion contact shadowing factor."""
        occ = 0.0
        sca = 1.0
        for i in range(4):
            hr = 0.02 + 0.05 * float(i)
            aop = pos + nor * hr
            dd, _ = self.scene_sdf(aop, objects)
            occ += -(dd - hr) * sca
            sca *= 0.75
        return float(np.clip(1.0 - 2.5 * occ, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Image Rendering Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def render_scene(
        self,
        objects: List[SDFObject],
        width: int = 128,
        height: int = 128,
        cam_pos: np.ndarray = np.array([0.0, 1.5, -3.5]),
        cam_target: np.ndarray = np.array([0.0, 0.5, 0.0]),
        fov_deg: float = 60.0,
        light_dir: np.ndarray = np.array([0.6, 0.8, -0.5]),
    ) -> np.ndarray:
        """
        Render the 3D scene from camera viewpoint into an RGB uint8 image array.
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Normalize light vector
        l_dir = light_dir / (np.linalg.norm(light_dir) + 1e-7)

        # Camera coordinate frame (LookAt)
        forward = cam_target - cam_pos
        forward /= (np.linalg.norm(forward) + 1e-7)
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        right /= (np.linalg.norm(right) + 1e-7)
        up = np.cross(forward, right)

        aspect = width / float(height)
        fov_rad = np.radians(fov_deg)
        tan_half_fov = np.tan(fov_rad / 2.0)

        for y in range(height):
            ndc_y = (1.0 - 2.0 * ((y + 0.5) / height)) * tan_half_fov
            for x in range(width):
                ndc_x = (2.0 * ((x + 0.5) / width) - 1.0) * tan_half_fov * aspect

                rd = forward + ndc_x * right + ndc_y * up
                rd /= (np.linalg.norm(rd) + 1e-7)

                # Sphere tracing march
                t = 0.0
                hit = False
                hit_mat = Material()

                for _ in range(self.max_steps):
                    p = cam_pos + rd * t
                    dist, mat = self.scene_sdf(p, objects)
                    if dist < self.eps:
                        hit = True
                        hit_mat = mat
                        break
                    t += dist
                    if t > self.max_dist:
                        break

                if hit:
                    hit_pos = cam_pos + rd * t
                    normal = self.calc_normal(hit_pos, objects)

                    # Ambient + Diffuse (Lambert)
                    ndotl = max(0.0, float(np.dot(normal, l_dir)))
                    shadow = self.calc_soft_shadow(hit_pos + normal * 0.01, l_dir, objects)
                    ao = self.calc_ao(hit_pos, normal, objects)

                    # Specular (Blinn-Phong)
                    half_vec = (l_dir - rd)
                    half_vec /= (np.linalg.norm(half_vec) + 1e-7)
                    ndoth = max(0.0, float(np.dot(normal, half_vec)))
                    spec_power = (1.0 - hit_mat.roughness) * 64.0 + 4.0
                    spec = hit_mat.specular * (ndoth ** spec_power) * shadow

                    diffuse_color = np.array(hit_mat.albedo, dtype=np.float32)
                    ambient_color = diffuse_color * 0.2 * ao
                    direct_color = diffuse_color * (ndotl * shadow) + spec

                    final_color = np.clip((ambient_color + direct_color) * 255.0, 0.0, 255.0)
                    img[y, x] = final_color.astype(np.uint8)
                else:
                    # Dynamic sky gradient background
                    sky_t = (rd[1] + 1.0) * 0.5
                    sky_rgb = np.array([120.0, 170.0, 240.0]) * sky_t + np.array([210.0, 230.0, 255.0]) * (1.0 - sky_t)
                    img[y, x] = sky_rgb.astype(np.uint8)

        return img
