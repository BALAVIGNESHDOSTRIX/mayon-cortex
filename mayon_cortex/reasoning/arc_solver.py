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
Mayon-Cortex ARC-AGI 2D Grid Perception & DSL Program Synthesis Engine
========================================================================
Pure CPU, zero-GPU neuro-symbolic engine for solving François Chollet's
Abstraction and Reasoning Corpus (ARC-AGI) tasks.

Key Components:
1. ARCGrid & ARCObject: 2D matrix representations, connected-component object segmentation,
   bounding boxes, topological features, and ASCII visualization.
2. ARCPerception: Comprehensive cognitive visual perception module:
   - Object extraction (4-way/8-way connectivity)
   - Background detection & color histograms
   - Multi-axis symmetry detection (horizontal, vertical, rotational, diagonal)
   - Boundary outline extraction & topological hole/enclosure detection
   - Sub-grid partitioning & periodic pattern discovery
   - Input-output demonstration comparative analysis (Phases 1-3)
3. ARCDSL: Comprehensive domain-specific language containing 60+ primitives:
   - Geometric: rot90/180/270, flip_h/v, transpose, diagonal_flip_anti, half crops
   - Morphological: dilate, erode, extract_outline, fill_interior, hollow_objects, flood_fill_corners
   - Grid Composition: overlay_nonzero, xor_grids, and_grids, subtract_grid, mask_by_color
   - Symmetry: mirror_left/right/top/bottom, complete_rotational/horizontal/vertical_symmetry
   - Object-Level: keep/remove smallest/largest object, sort_by_size, recolor_by_size, area filters
   - Color: swap_colors, recolor, fill_background, invert, map_most_to_least, binarize, unique_mask
   - Physics: cellular gravity (4 directions)
   - Lines & Topology: connect_same_colors_with_lines, fill_enclosed_holes
   - Pattern & Scaling: tile_2x2/3x3, repeat_h/v, scale_2x/3x, downscale_majority, pattern extension
   - Frames & Borders: add_border, remove_border, extract_border_ring, fill_border
4. ARCProgramSynthesizer & IntelligentSynthesizer: Heuristic-guided beam search program synthesizer
   utilizing IO-delta signal pruning to explore deeper search depths with minimal CPU latency.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import math
import time
import json
import os


# Standard ARC Color Map (0-9)
ARC_COLOR_NAMES = {
    0: "Black",
    1: "Blue",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Grey",
    6: "Magenta",
    7: "Orange",
    8: "Azure",
    9: "Maroon",
}

ARC_ASCII_CHARS = {
    0: " . ",  # Black / Empty
    1: " 1 ",  # Blue
    2: " 2 ",  # Red
    3: " 3 ",  # Green
    4: " 4 ",  # Yellow
    5: " 5 ",  # Grey
    6: " 6 ",  # Magenta
    7: " 7 ",  # Orange
    8: " 8 ",  # Azure
    9: " 9 ",  # Maroon
}


@dataclass
class ARCObject:
    """A discrete 2D spatial entity segmented from an ARCGrid."""
    color: int
    mask: np.ndarray  # Binary 2D mask matching parent grid shape
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c) inclusive
    area: int
    center_of_mass: Tuple[float, float]
    parent_shape: Tuple[int, int]

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    def crop_subgrid(self) -> np.ndarray:
        """Returns the isolated subgrid bounding box of this object."""
        r1, c1, r2, c2 = self.bbox
        cropped_mask = self.mask[r1:r2 + 1, c1:c2 + 1]
        subgrid = np.zeros_like(cropped_mask, dtype=int)
        subgrid[cropped_mask] = self.color
        return subgrid

    def is_solid_rectangle(self) -> bool:
        """Checks if this object forms a solid filled rectangle."""
        return self.area == (self.height * self.width)

    def translate(self, dr: int, dc: int, target_shape: Optional[Tuple[int, int]] = None) -> "ARCObject":
        """Shifts object by (dr, dc) within target_shape."""
        H, W = target_shape if target_shape is not None else self.parent_shape
        new_mask = np.zeros((H, W), dtype=bool)
        coords = np.argwhere(self.mask)
        new_coords = []
        for r, c in coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                new_mask[nr, nc] = True
                new_coords.append((nr, nc))
        if not new_coords:
            return ARCObject(
                color=self.color,
                mask=new_mask,
                bbox=(0, 0, 0, 0),
                area=0,
                center_of_mass=(0.0, 0.0),
                parent_shape=(H, W),
            )
        rows = [r for r, _ in new_coords]
        cols = [c for _, c in new_coords]
        return ARCObject(
            color=self.color,
            mask=new_mask,
            bbox=(min(rows), min(cols), max(rows), max(cols)),
            area=len(new_coords),
            center_of_mass=(float(np.mean(rows)), float(np.mean(cols))),
            parent_shape=(H, W),
        )

    def recolor(self, new_color: int) -> "ARCObject":
        """Returns a copy of the object with a new color."""
        return ARCObject(
            color=new_color,
            mask=self.mask.copy(),
            bbox=self.bbox,
            area=self.area,
            center_of_mass=self.center_of_mass,
            parent_shape=self.parent_shape,
        )

    def local_rotate(self, k: int = 1) -> "ARCObject":
        """Rotates object content locally inside its bounding box."""
        subgrid = self.crop_subgrid()
        rot_sub = np.rot90(subgrid, k=k)
        sh, sw = rot_sub.shape
        H, W = self.parent_shape
        r1, c1, _, _ = self.bbox
        new_mask = np.zeros((H, W), dtype=bool)
        end_r = min(H, r1 + sh)
        end_c = min(W, c1 + sw)
        sub_h = end_r - r1
        sub_w = end_c - c1
        if sub_h > 0 and sub_w > 0:
            new_mask[r1:end_r, c1:end_c] = (rot_sub[:sub_h, :sub_w] != 0)
        coords = np.argwhere(new_mask)
        if len(coords) == 0:
            return self
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        return ARCObject(
            color=self.color,
            mask=new_mask,
            bbox=(min(rows), min(cols), max(rows), max(cols)),
            area=len(coords),
            center_of_mass=(float(np.mean(rows)), float(np.mean(cols))),
            parent_shape=(H, W),
        )

    def local_flip_h(self) -> "ARCObject":
        """Flips object horizontally within its bounding box."""
        subgrid = self.crop_subgrid()
        flipped = np.fliplr(subgrid)
        sh, sw = flipped.shape
        H, W = self.parent_shape
        r1, c1, _, _ = self.bbox
        new_mask = np.zeros((H, W), dtype=bool)
        new_mask[r1:r1 + sh, c1:c1 + sw] = (flipped != 0)
        return ARCObject(
            color=self.color,
            mask=new_mask,
            bbox=self.bbox,
            area=self.area,
            center_of_mass=self.center_of_mass,
            parent_shape=(H, W),
        )

    def local_flip_v(self) -> "ARCObject":
        """Flips object vertically within its bounding box."""
        subgrid = self.crop_subgrid()
        flipped = np.flipud(subgrid)
        sh, sw = flipped.shape
        H, W = self.parent_shape
        r1, c1, _, _ = self.bbox
        new_mask = np.zeros((H, W), dtype=bool)
        new_mask[r1:r1 + sh, c1:c1 + sw] = (flipped != 0)
        return ARCObject(
            color=self.color,
            mask=new_mask,
            bbox=self.bbox,
            area=self.area,
            center_of_mass=self.center_of_mass,
            parent_shape=(H, W),
        )

    def render_onto(self, canvas: np.ndarray) -> None:
        """Paints object pixels onto a 2D numpy matrix canvas."""
        canvas[self.mask] = self.color



class ARCGrid:
    """
    2D Matrix representation of an ARC grid with spatial & topological helper methods.
    """
    def __init__(self, data: Any):
        if isinstance(data, list):
            self.matrix = np.array(data, dtype=int)
        elif isinstance(data, np.ndarray):
            self.matrix = data.astype(int)
        elif isinstance(data, ARCGrid):
            self.matrix = data.matrix.copy()
        else:
            raise ValueError(f"Unsupported grid data type: {type(data)}")

        if self.matrix.ndim != 2:
            raise ValueError(f"ARCGrid must be 2-dimensional, got shape {self.matrix.shape}")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape

    @property
    def height(self) -> int:
        return self.matrix.shape[0]

    @property
    def width(self) -> int:
        return self.matrix.shape[1]

    @property
    def colors(self) -> Set[int]:
        return set(np.unique(self.matrix))

    def copy(self) -> "ARCGrid":
        return ARCGrid(self.matrix.copy())

    def to_list(self) -> List[List[int]]:
        return self.matrix.tolist()

    def equals(self, other: Any) -> bool:
        if isinstance(other, ARCGrid):
            return np.array_equal(self.matrix, other.matrix)
        elif isinstance(other, (list, np.ndarray)):
            return np.array_equal(self.matrix, np.array(other))
        return False

    def __eq__(self, other: Any) -> bool:
        return self.equals(other)

    def mismatch_count(self, other: "ARCGrid") -> int:
        """Count mismatching pixels between two grids."""
        if self.shape != other.shape:
            return 999999 + abs(self.height - other.height) * 100 + abs(self.width - other.width) * 100
        return int(np.sum(self.matrix != other.matrix))

    def to_ascii(self, title: Optional[str] = None) -> str:
        """Returns clean ASCII visualization of the grid."""
        lines = []
        if title:
            lines.append(f"[{title} ({self.height}x{self.width})]")
        lines.append("┌" + "───" * self.width + "┐")
        for row in self.matrix:
            row_str = "".join(ARC_ASCII_CHARS.get(int(val), f"{val:2d} ") for val in row)
            lines.append("│" + row_str + "│")
        lines.append("└" + "───" * self.width + "┘")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 1. 2D SPATIAL PERCEPTION & OBJECT SEGMENTATION (PHASE 1)
# ─────────────────────────────────────────────────────────────

class ARCPerception:
    """
    Cognitive visual perception module: extracts discrete objects, topological holes,
    background colors, multi-axis symmetries, outlines, partitions, and IO relationships.
    """

    @staticmethod
    def extract_objects(
        grid: ARCGrid,
        connectivity: int = 4,
        background_color: int = 0,
        monochromatic: bool = True,
    ) -> List[ARCObject]:
        """
        Segment connected components of the grid into distinct ARCObjects.
        connectivity: 4 (orthogonal) or 8 (orthogonal + diagonal).
        """
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        objects = []

        neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors_8 = neighbors_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        deltas = neighbors_8 if connectivity == 8 else neighbors_4

        for r in range(H):
            for c in range(W):
                val = grid.matrix[r, c]
                if val == background_color or visited[r, c]:
                    continue

                # BFS / Flood-fill to extract connected component
                component_coords = []
                queue = [(r, c)]
                visited[r, c] = True
                target_color = val

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    component_coords.append((curr_r, curr_c))

                    for dr, dc in deltas:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            nval = grid.matrix[nr, nc]
                            if (monochromatic and nval == target_color) or (not monochromatic and nval != background_color):
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                # Build ARCObject
                mask = np.zeros((H, W), dtype=bool)
                for cr, cc in component_coords:
                    mask[cr, cc] = True

                rows = [cr for cr, _ in component_coords]
                cols = [cc for _, cc in component_coords]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                area = len(component_coords)
                com = (float(np.mean(rows)), float(np.mean(cols)))

                obj = ARCObject(
                    color=target_color,
                    mask=mask,
                    bbox=(min_r, min_c, max_r, max_c),
                    area=area,
                    center_of_mass=com,
                    parent_shape=(H, W),
                )
                objects.append(obj)

        return objects

    @staticmethod
    def detect_enclosed_holes(grid: ARCGrid, background_color: int = 0) -> List[Tuple[int, int]]:
        """
        Detects coordinates of background cells completely enclosed by non-background boundaries.
        Uses reverse flood-fill from grid borders.
        """
        H, W = grid.shape
        is_bg = (grid.matrix == background_color)
        reach_border = np.zeros((H, W), dtype=bool)

        # Queue border background pixels
        queue = []
        for r in range(H):
            for c in [0, W - 1]:
                if is_bg[r, c] and not reach_border[r, c]:
                    reach_border[r, c] = True
                    queue.append((r, c))
        for c in range(W):
            for r in [0, H - 1]:
                if is_bg[r, c] and not reach_border[r, c]:
                    reach_border[r, c] = True
                    queue.append((r, c))

        # Flood fill to find all border-connected background
        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and is_bg[nr, nc] and not reach_border[nr, nc]:
                    reach_border[nr, nc] = True
                    queue.append((nr, nc))

        # Enclosed holes = background pixels that cannot reach any border
        holes = []
        for r in range(H):
            for c in range(W):
                if is_bg[r, c] and not reach_border[r, c]:
                    holes.append((r, c))
        return holes

    @staticmethod
    def detect_enclosed_chambers(grid: ARCGrid, background_color: int = 0) -> List[List[Tuple[int, int]]]:
        """
        Groups enclosed background holes into discrete connected topological chambers.
        """
        holes = ARCPerception.detect_enclosed_holes(grid, background_color=background_color)
        if not holes:
            return []
        hole_set = set(holes)
        visited = set()
        chambers = []
        for r, c in holes:
            if (r, c) not in visited:
                comp = []
                stack = [(r, c)]
                visited.add((r, c))
                while stack:
                    curr_r, curr_c = stack.pop()
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (nr, nc) in hole_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                chambers.append(comp)
        return chambers

    @staticmethod
    def detect_background_color(grid: ARCGrid) -> int:
        """
        Auto-detect the most plausible background color.
        Border pixels are heavily weighted since background typically surrounds content.
        """
        H, W = grid.shape
        border_pixels = []
        border_pixels.extend(grid.matrix[0, :].tolist())
        border_pixels.extend(grid.matrix[H - 1, :].tolist())
        border_pixels.extend(grid.matrix[:, 0].tolist())
        border_pixels.extend(grid.matrix[:, W - 1].tolist())

        if border_pixels:
            border_counts = np.bincount(border_pixels, minlength=10)
            top_border_color = int(np.argmax(border_counts))
            if border_counts[top_border_color] / len(border_pixels) >= 0.35:
                return top_border_color

        flat = grid.matrix.flatten()
        counts = np.bincount(flat, minlength=10)
        return int(np.argmax(counts))

    @staticmethod
    def count_objects(
        grid: ARCGrid,
        connectivity: int = 4,
        background_color: Optional[int] = None,
    ) -> int:
        """Counts distinct connected components in the grid."""
        bg = background_color if background_color is not None else ARCPerception.detect_background_color(grid)
        objs = ARCPerception.extract_objects(grid, connectivity=connectivity, background_color=bg)
        return len(objs)

    @staticmethod
    def color_histogram(grid: ARCGrid) -> Dict[int, int]:
        """Returns frequency count of each color present in the grid."""
        unique, counts = np.unique(grid.matrix, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    @staticmethod
    def detect_symmetry(grid: ARCGrid) -> Dict[str, bool]:
        """Detects horizontal, vertical, rotational, and diagonal symmetries."""
        mat = grid.matrix
        H, W = mat.shape
        symmetries = {
            "horizontal": bool(np.array_equal(mat, np.flipud(mat))),
            "vertical": bool(np.array_equal(mat, np.fliplr(mat))),
            "rotational_180": bool(np.array_equal(mat, np.rot90(mat, 2))),
            "diagonal_main": False,
            "diagonal_anti": False,
        }
        if H == W:
            symmetries["diagonal_main"] = bool(np.array_equal(mat, mat.T))
            symmetries["diagonal_anti"] = bool(np.array_equal(mat, mat[::-1, ::-1].T))
        return symmetries

    @staticmethod
    def extract_outline_mask(grid: ARCGrid, background_color: int = 0) -> np.ndarray:
        """Returns boolean mask of boundary pixels of non-background objects."""
        H, W = grid.shape
        is_fg = (grid.matrix != background_color)
        outline = np.zeros((H, W), dtype=bool)

        for r in range(H):
            for c in range(W):
                if not is_fg[r, c]:
                    continue
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    outline[r, c] = True
                else:
                    if (
                        not is_fg[r - 1, c]
                        or not is_fg[r + 1, c]
                        or not is_fg[r, c - 1]
                        or not is_fg[r, c + 1]
                    ):
                        outline[r, c] = True
        return outline

    @staticmethod
    def detect_grid_partition(grid: ARCGrid) -> Optional[Dict[str, Any]]:
        """
        Detects if grid is divided by uniform horizontal/vertical separator lines.
        """
        H, W = grid.shape
        divider_rows = []
        for r in range(H):
            vals = np.unique(grid.matrix[r, :])
            if len(vals) == 1:
                divider_rows.append((r, int(vals[0])))

        divider_cols = []
        for c in range(W):
            vals = np.unique(grid.matrix[:, c])
            if len(vals) == 1:
                divider_cols.append((c, int(vals[0])))

        if divider_rows or divider_cols:
            return {
                "divider_rows": [r for r, _ in divider_rows],
                "divider_cols": [c for c, _ in divider_cols],
                "has_grid_lines": True,
            }
        return None

    @staticmethod
    def find_repeating_pattern(grid: ARCGrid) -> Optional[Tuple[int, int]]:
        """
        Checks if the grid is composed of a periodically tiled sub-matrix of size (ph, pw).
        """
        H, W = grid.shape
        mat = grid.matrix
        for ph in range(1, H + 1):
            if H % ph != 0:
                continue
            for pw in range(1, W + 1):
                if W % pw != 0:
                    continue
                if ph == H and pw == W:
                    continue
                sub = mat[:ph, :pw]
                tiled = np.tile(sub, (H // ph, W // pw))
                if np.array_equal(mat, tiled):
                    return (ph, pw)
        return None

    @staticmethod
    def analyze_io_relationship(train_pairs: List[Tuple[ARCGrid, ARCGrid]]) -> Dict[str, Any]:
        """
        Comprehensive comparative analysis of input -> output demonstration pairs.
        Extracts dimensional deltas, color transfers, symmetry changes, and topological cues.
        """
        if not train_pairs:
            return {
                "same_shape": True,
                "output_is_smaller": False,
                "output_is_larger": False,
                "colors_in": set(),
                "colors_out": set(),
                "colors_added": set(),
                "colors_removed": set(),
                "colors_preserved": True,
                "has_holes_in": False,
                "symmetry_gained": False,
                "scale_factor": None,
            }

        same_shape = True
        shape_ratios = []
        in_shapes = []
        out_shapes = []
        colors_in_all: Set[int] = set()
        colors_out_all: Set[int] = set()
        colors_added_all: Set[int] = set()
        colors_removed_all: Set[int] = set()
        has_holes_in = False
        symmetry_gained = False
        fixed_output_shape = True
        first_out_shape = train_pairs[0][1].shape

        for in_g, out_g in train_pairs:
            in_shapes.append(in_g.shape)
            out_shapes.append(out_g.shape)
            if in_g.shape != out_g.shape:
                same_shape = False
            if out_g.shape != first_out_shape:
                fixed_output_shape = False

            rh = out_g.height / max(1, in_g.height)
            rw = out_g.width / max(1, in_g.width)
            shape_ratios.append((rh, rw))

            c_in = in_g.colors
            c_out = out_g.colors
            colors_in_all.update(c_in)
            colors_out_all.update(c_out)
            colors_added_all.update(c_out - c_in)
            colors_removed_all.update(c_in - c_out)

            if len(ARCPerception.detect_enclosed_holes(in_g)) > 0:
                has_holes_in = True

            sym_in = ARCPerception.detect_symmetry(in_g)
            sym_out = ARCPerception.detect_symmetry(out_g)
            for k in sym_out:
                if sym_out[k] and not sym_in.get(k, False):
                    symmetry_gained = True

        scale_factor = None
        if shape_ratios and all(r == shape_ratios[0] for r in shape_ratios):
            rh, rw = shape_ratios[0]
            if rh == rw and rh in [2.0, 3.0, 4.0]:
                scale_factor = int(rh)

        output_is_smaller = all(
            o[0] <= i[0] and o[1] <= i[1] and (o[0] < i[0] or o[1] < i[1])
            for i, o in zip(in_shapes, out_shapes)
        )
        output_is_larger = all(
            o[0] >= i[0] and o[1] >= i[1] and (o[0] > i[0] or o[1] > i[1])
            for i, o in zip(in_shapes, out_shapes)
        )

        return {
            "same_shape": same_shape,
            "fixed_output_shape": first_out_shape if fixed_output_shape else None,
            "scale_factor": scale_factor,
            "output_is_smaller": output_is_smaller,
            "output_is_larger": output_is_larger,
            "colors_in": colors_in_all,
            "colors_out": colors_out_all,
            "colors_added": colors_added_all,
            "colors_removed": colors_removed_all,
            "colors_preserved": (colors_in_all == colors_out_all),
            "has_holes_in": has_holes_in,
            "symmetry_gained": symmetry_gained,
        }


# ─────────────────────────────────────────────────────────────
# 2. ARC DOMAIN-SPECIFIC LANGUAGE (DSL) OPERATORS (PHASE 2)
# ─────────────────────────────────────────────────────────────

class ARCDSL:
    """
    Comprehensive primitive operations over 2D grids (Geometric, Morphological, Symmetry,
    Composition, Object-level, Color, Physics, Lines, and Patterns).
    """

    # ── Geometric Transforms ──
    @staticmethod
    def rot90(grid: ARCGrid) -> ARCGrid:
        """Rotate grid 90 degrees clockwise."""
        return ARCGrid(np.rot90(grid.matrix, k=-1))

    @staticmethod
    def rot180(grid: ARCGrid) -> ARCGrid:
        """Rotate grid 180 degrees."""
        return ARCGrid(np.rot90(grid.matrix, k=2))

    @staticmethod
    def rot270(grid: ARCGrid) -> ARCGrid:
        """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
        return ARCGrid(np.rot90(grid.matrix, k=1))

    @staticmethod
    def flip_h(grid: ARCGrid) -> ARCGrid:
        """Flip grid horizontally (left-right)."""
        return ARCGrid(np.fliplr(grid.matrix))

    @staticmethod
    def flip_v(grid: ARCGrid) -> ARCGrid:
        """Flip grid vertically (top-down)."""
        return ARCGrid(np.flipud(grid.matrix))

    @staticmethod
    def transpose(grid: ARCGrid) -> ARCGrid:
        """Transpose grid (swap rows and columns along main diagonal)."""
        return ARCGrid(grid.matrix.T)

    @staticmethod
    def diagonal_flip_anti(grid: ARCGrid) -> ARCGrid:
        """Reflects grid diagonally across the anti-diagonal."""
        return ARCGrid(grid.matrix[::-1, ::-1].T)

    # ── Morphological Operations ──
    @staticmethod
    def dilate(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Expands non-background colored pixels by 1 cell in 4 orthogonal directions."""
        H, W = grid.shape
        res = grid.matrix.copy()
        for r in range(H):
            for c in range(W):
                val = grid.matrix[r, c]
                if val != background_color:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid.matrix[nr, nc] == background_color:
                            res[nr, nc] = val
        return ARCGrid(res)

    @staticmethod
    def erode(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Shrinks non-background shapes by removing pixels touching the background or border."""
        H, W = grid.shape
        res = grid.matrix.copy()
        for r in range(H):
            for c in range(W):
                if grid.matrix[r, c] != background_color:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= H or nc < 0 or nc >= W or grid.matrix[nr, nc] == background_color:
                            res[r, c] = background_color
                            break
        return ARCGrid(res)

    @staticmethod
    def extract_outline(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Keeps only the outer boundary pixels of non-background objects, setting interior to background."""
        outline_mask = ARCPerception.extract_outline_mask(grid, background_color=background_color)
        res = np.full_like(grid.matrix, background_color)
        res[outline_mask] = grid.matrix[outline_mask]
        return ARCGrid(res)

    @staticmethod
    def fill_interior(grid: ARCGrid, fill_color: int = 4, background_color: int = 0) -> ARCGrid:
        """Fills enclosed hollow chambers within non-background shapes."""
        return ARCDSL.fill_enclosed_holes(grid, fill_color=fill_color, background_color=background_color)

    @staticmethod
    def flood_fill_from_corners(grid: ARCGrid, fill_color: int = 1, background_color: int = 0) -> ARCGrid:
        """Flood fills background cells starting from the 4 outer corners of the grid."""
        H, W = grid.shape
        res = grid.matrix.copy()
        corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
        visited = np.zeros((H, W), dtype=bool)
        queue = []
        for cr, cc in corners:
            if res[cr, cc] == background_color and not visited[cr, cc]:
                visited[cr, cc] = True
                queue.append((cr, cc))

        while queue:
            r, c = queue.pop(0)
            res[r, c] = fill_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and res[nr, nc] == background_color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return ARCGrid(res)

    @staticmethod
    def hollow_objects(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Turns the interior of solid objects to background, keeping only their perimeter shell."""
        outline_mask = ARCPerception.extract_outline_mask(grid, background_color=background_color)
        res = np.full_like(grid.matrix, background_color)
        res[outline_mask] = grid.matrix[outline_mask]
        return ARCGrid(res)

    # ── Grid Composition & Subtraction ──
    @staticmethod
    def overlay_nonzero(base: ARCGrid, overlay: ARCGrid) -> ARCGrid:
        """Pastes non-zero pixels of overlay onto base grid (same dimensions required)."""
        if base.shape != overlay.shape:
            return base.copy()
        res = base.matrix.copy()
        mask = (overlay.matrix != 0)
        res[mask] = overlay.matrix[mask]
        return ARCGrid(res)

    @staticmethod
    def xor_grids(g1: ARCGrid, g2: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Keeps pixels that are non-background in exactly one of the two grids."""
        if g1.shape != g2.shape:
            return g1.copy()
        res = np.full_like(g1.matrix, background_color)
        m1 = (g1.matrix != background_color)
        m2 = (g2.matrix != background_color)
        res[m1 & ~m2] = g1.matrix[m1 & ~m2]
        res[~m1 & m2] = g2.matrix[~m1 & m2]
        return ARCGrid(res)

    @staticmethod
    def and_grids(g1: ARCGrid, g2: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retains pixels where both grids share the exact same non-background color."""
        if g1.shape != g2.shape:
            return g1.copy()
        res = np.full_like(g1.matrix, background_color)
        mask = (g1.matrix == g2.matrix) & (g1.matrix != background_color)
        res[mask] = g1.matrix[mask]
        return ARCGrid(res)

    @staticmethod
    def subtract_grid(g1: ARCGrid, g2: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Clears pixels in g1 that are non-background in g2."""
        if g1.shape != g2.shape:
            return g1.copy()
        res = g1.matrix.copy()
        res[g2.matrix != background_color] = background_color
        return ARCGrid(res)

    @staticmethod
    def mask_by_color(
        grid: ARCGrid,
        mask_grid: ARCGrid,
        mask_color: int,
        background_color: int = 0,
    ) -> ARCGrid:
        """Retains pixels in grid only where mask_grid matches mask_color."""
        if grid.shape != mask_grid.shape:
            return grid.copy()
        res = np.full_like(grid.matrix, background_color)
        mask = (mask_grid.matrix == mask_color)
        res[mask] = grid.matrix[mask]
        return ARCGrid(res)

    # ── Symmetry & Reflection ──
    @staticmethod
    def mirror_left_to_right(grid: ARCGrid) -> ARCGrid:
        """Mirrors the left half of the grid onto the right half."""
        H, W = grid.shape
        res = grid.matrix.copy()
        half = W // 2
        for c in range(half):
            res[:, W - 1 - c] = res[:, c]
        return ARCGrid(res)

    @staticmethod
    def mirror_right_to_left(grid: ARCGrid) -> ARCGrid:
        """Mirrors the right half of the grid onto the left half."""
        H, W = grid.shape
        res = grid.matrix.copy()
        half = W // 2
        for c in range(half):
            res[:, c] = res[:, W - 1 - c]
        return ARCGrid(res)

    @staticmethod
    def mirror_top_to_bottom(grid: ARCGrid) -> ARCGrid:
        """Mirrors the top half of the grid onto the bottom half."""
        H, W = grid.shape
        res = grid.matrix.copy()
        half = H // 2
        for r in range(half):
            res[H - 1 - r, :] = res[r, :]
        return ARCGrid(res)

    @staticmethod
    def mirror_bottom_to_top(grid: ARCGrid) -> ARCGrid:
        """Mirrors the bottom half of the grid onto the top half."""
        H, W = grid.shape
        res = grid.matrix.copy()
        half = H // 2
        for r in range(half):
            res[r, :] = res[H - 1 - r, :]
        return ARCGrid(res)

    @staticmethod
    def complete_rotational_symmetry(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Completes 180-degree point reflection symmetry by transferring non-bg pixels to opposite cells."""
        res = grid.matrix.copy()
        rot = np.rot90(grid.matrix, 2)
        bg_mask = (res == background_color)
        res[bg_mask] = rot[bg_mask]
        return ARCGrid(res)

    @staticmethod
    def complete_horizontal_symmetry(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Completes vertical axis (horizontal flip) symmetry by filling empty cells from mirrored counterparts."""
        res = grid.matrix.copy()
        flipped = np.fliplr(grid.matrix)
        bg_mask = (res == background_color)
        res[bg_mask] = flipped[bg_mask]
        return ARCGrid(res)

    @staticmethod
    def complete_vertical_symmetry(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Completes horizontal axis (vertical flip) symmetry by filling empty cells from mirrored counterparts."""
        res = grid.matrix.copy()
        flipped = np.flipud(grid.matrix)
        bg_mask = (res == background_color)
        res[bg_mask] = flipped[bg_mask]
        return ARCGrid(res)

    # ── Object-Level Operations ──
    @staticmethod
    def crop_to_content(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Crop grid to bounding box of all non-background content."""
        non_bg = np.argwhere(grid.matrix != background_color)
        if len(non_bg) == 0:
            return grid.copy()
        min_r, min_c = non_bg.min(axis=0)
        max_r, max_c = non_bg.max(axis=0)
        return ARCGrid(grid.matrix[min_r:max_r + 1, min_c:max_c + 1])

    @staticmethod
    def extract_largest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Extract the largest connected object as an isolated cropped subgrid."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if not objs:
            return grid.copy()
        largest = max(objs, key=lambda o: o.area)
        return ARCGrid(largest.crop_subgrid())

    @staticmethod
    def extract_smallest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Extract the smallest connected object as an isolated cropped subgrid."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if not objs:
            return grid.copy()
        smallest = min(objs, key=lambda o: o.area)
        return ARCGrid(smallest.crop_subgrid())

    @staticmethod
    def remove_smallest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Removes the smallest connected foreground object, setting its pixels to background."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if len(objs) <= 1:
            return grid.copy()
        smallest = min(objs, key=lambda o: o.area)
        res = grid.matrix.copy()
        res[smallest.mask] = background_color
        return ARCGrid(res)

    @staticmethod
    def remove_largest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Removes the largest connected foreground object, setting its pixels to background."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if len(objs) <= 1:
            return grid.copy()
        largest = max(objs, key=lambda o: o.area)
        res = grid.matrix.copy()
        res[largest.mask] = background_color
        return ARCGrid(res)

    @staticmethod
    def keep_largest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retains only the largest connected object in place, clearing all others to background."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if not objs:
            return grid.copy()
        largest = max(objs, key=lambda o: o.area)
        res = np.full_like(grid.matrix, background_color)
        res[largest.mask] = largest.color
        return ARCGrid(res)

    @staticmethod
    def keep_smallest_object(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retains only the smallest connected object in place, clearing all others to background."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if not objs:
            return grid.copy()
        smallest = min(objs, key=lambda o: o.area)
        res = np.full_like(grid.matrix, background_color)
        res[smallest.mask] = smallest.color
        return ARCGrid(res)

    @staticmethod
    def sort_objects_by_size(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Packs connected objects from left to right along the bottom edge, ordered by area ascending."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if len(objs) <= 1:
            return grid.copy()
        res = np.full_like(grid.matrix, background_color)
        sorted_objs = sorted(objs, key=lambda o: o.area)
        curr_c = 0
        H, W = grid.shape
        for obj in sorted_objs:
            sub = obj.crop_subgrid()
            sh, sw = sub.shape
            if curr_c + sw <= W and sh <= H:
                dest = res[H - sh:H, curr_c:curr_c + sw]
                dest[sub != 0] = sub[sub != 0]
                curr_c += sw + 1
        return ARCGrid(res)

    @staticmethod
    def replace_color_by_object_size(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Recolors each object according to its rank-order size (smallest=1, next=2, etc.)."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        if not objs:
            return grid.copy()
        res = np.full_like(grid.matrix, background_color)
        sorted_objs = sorted(objs, key=lambda o: o.area)
        for rank, obj in enumerate(sorted_objs):
            color = min(9, rank + 1)
            res[obj.mask] = color
        return ARCGrid(res)

    @staticmethod
    def filter_objects_by_min_area(grid: ARCGrid, min_area: int = 2, background_color: int = 0) -> ARCGrid:
        """Removes all connected objects with area strictly smaller than min_area."""
        objs = ARCPerception.extract_objects(grid, connectivity=4, background_color=background_color)
        res = grid.matrix.copy()
        for obj in objs:
            if obj.area < min_area:
                res[obj.mask] = background_color
        return ARCGrid(res)

    # ── Additional Geometric & Cropping Transforms ──
    @staticmethod
    def upscale_nearest(grid: ARCGrid, factor: int = 2) -> ARCGrid:
        """Upscales grid dimensions by integer factor using nearest-neighbor repetition."""
        return ARCGrid(np.repeat(np.repeat(grid.matrix, factor, axis=0), factor, axis=1))

    @staticmethod
    def downscale_majority(grid: ARCGrid, factor: int = 2) -> ARCGrid:
        """Downscales grid by factor using majority voting within each block."""
        H, W = grid.shape
        if H < factor or W < factor:
            return grid.copy()
        new_h = H // factor
        new_w = W // factor
        res = np.zeros((new_h, new_w), dtype=int)
        for r in range(new_h):
            for c in range(new_w):
                block = grid.matrix[r * factor:(r + 1) * factor, c * factor:(c + 1) * factor].flatten()
                counts = np.bincount(block, minlength=10)
                res[r, c] = int(np.argmax(counts))
        return ARCGrid(res)

    @staticmethod
    def crop_half_left(grid: ARCGrid) -> ARCGrid:
        """Crops to the left half of the grid."""
        W = grid.width
        return ARCGrid(grid.matrix[:, :max(1, W // 2)])

    @staticmethod
    def crop_half_right(grid: ARCGrid) -> ARCGrid:
        """Crops to the right half of the grid."""
        W = grid.width
        return ARCGrid(grid.matrix[:, W // 2:])

    @staticmethod
    def crop_half_top(grid: ARCGrid) -> ARCGrid:
        """Crops to the top half of the grid."""
        H = grid.height
        return ARCGrid(grid.matrix[:max(1, H // 2), :])

    @staticmethod
    def crop_half_bottom(grid: ARCGrid) -> ARCGrid:
        """Crops to the bottom half of the grid."""
        H = grid.height
        return ARCGrid(grid.matrix[H // 2:, :])

    # ── Color Transforms ──
    @staticmethod
    def recolor(grid: ARCGrid, old_color: int, new_color: int) -> ARCGrid:
        """Recolors all cells of old_color to new_color."""
        res = grid.matrix.copy()
        res[res == old_color] = new_color
        return ARCGrid(res)

    @staticmethod
    def swap_colors(grid: ARCGrid, c1: int, c2: int) -> ARCGrid:
        """Swaps all occurrences of color c1 and color c2."""
        res = grid.matrix.copy()
        m1 = (grid.matrix == c1)
        m2 = (grid.matrix == c2)
        res[m1] = c2
        res[m2] = c1
        return ARCGrid(res)

    @staticmethod
    def fill_background(grid: ARCGrid, new_color: int) -> ARCGrid:
        """Fills background color 0 with new_color."""
        return ARCDSL.recolor(grid, 0, new_color)

    @staticmethod
    def invert_colors(grid: ARCGrid) -> ARCGrid:
        """Swap background and foreground or invert non-zero colors."""
        res = grid.matrix.copy()
        non_zeros = res[res != 0]
        if len(non_zeros) > 0:
            primary_fg = int(np.bincount(non_zeros).argmax())
            res[res == 0] = primary_fg
            res[res != primary_fg] = 0
        return ARCGrid(res)

    @staticmethod
    def map_most_to_least(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Swaps the most frequent and least frequent foreground colors."""
        fg = grid.matrix[grid.matrix != background_color]
        if len(fg) == 0:
            return grid.copy()
        counts = np.bincount(fg, minlength=10)
        present = [c for c in range(10) if counts[c] > 0]
        if len(present) < 2:
            return grid.copy()
        most_c = max(present, key=lambda c: counts[c])
        least_c = min(present, key=lambda c: counts[c])
        return ARCDSL.swap_colors(grid, most_c, least_c)

    @staticmethod
    def filter_by_color(grid: ARCGrid, keep_color: int, background_color: int = 0) -> ARCGrid:
        """Retain only pixels of keep_color, setting everything else to background."""
        res = np.full_like(grid.matrix, background_color)
        res[grid.matrix == keep_color] = keep_color
        return ARCGrid(res)

    @staticmethod
    def filter_most_frequent_color(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retain only the most common foreground color."""
        fg = grid.matrix[grid.matrix != background_color]
        if len(fg) == 0:
            return grid.copy()
        most_common = int(np.bincount(fg).argmax())
        return ARCDSL.filter_by_color(grid, most_common, background_color)

    @staticmethod
    def filter_least_frequent_color(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retain only the least common foreground color."""
        fg = grid.matrix[grid.matrix != background_color]
        if len(fg) == 0:
            return grid.copy()
        counts = np.bincount(fg)
        unique_colors = [c for c in range(len(counts)) if counts[c] > 0]
        least_common = min(unique_colors, key=lambda c: counts[c])
        return ARCDSL.filter_by_color(grid, least_common, background_color)

    @staticmethod
    def unique_color_mask(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Keeps only pixels belonging to colors that appear exactly once in the grid."""
        flat = grid.matrix.flatten()
        counts = np.bincount(flat, minlength=10)
        single_colors = {c for c in range(10) if counts[c] == 1}
        res = np.full_like(grid.matrix, background_color)
        for c in single_colors:
            res[grid.matrix == c] = c
        return ARCGrid(res)

    @staticmethod
    def binarize(grid: ARCGrid, fg_color: int = 1, background_color: int = 0) -> ARCGrid:
        """Converts all non-background pixels to a single uniform foreground color."""
        res = np.full_like(grid.matrix, background_color)
        res[grid.matrix != background_color] = fg_color
        return ARCGrid(res)

    @staticmethod
    def replace_background_with_dominant_fg(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Replaces the background color with the most prevalent foreground color."""
        fg = grid.matrix[grid.matrix != background_color]
        if len(fg) == 0:
            return grid.copy()
        dominant_fg = int(np.bincount(fg).argmax())
        res = grid.matrix.copy()
        res[res == background_color] = dominant_fg
        return ARCGrid(res)

    # ── Physics & Cellular Mechanics ──
    @staticmethod
    def gravity(grid: ARCGrid, direction: str = "down", background_color: int = 0) -> ARCGrid:
        """Simulate physical gravity pulling all colored pixels in a direction."""
        res = np.full_like(grid.matrix, background_color)
        H, W = grid.shape

        if direction == "down":
            for c in range(W):
                col_pixels = [grid.matrix[r, c] for r in range(H) if grid.matrix[r, c] != background_color]
                start_r = H - len(col_pixels)
                for idx, val in enumerate(col_pixels):
                    res[start_r + idx, c] = val
        elif direction == "up":
            for c in range(W):
                col_pixels = [grid.matrix[r, c] for r in range(H) if grid.matrix[r, c] != background_color]
                for idx, val in enumerate(col_pixels):
                    res[idx, c] = val
        elif direction == "right":
            for r in range(H):
                row_pixels = [grid.matrix[r, c] for c in range(W) if grid.matrix[r, c] != background_color]
                start_c = W - len(row_pixels)
                for idx, val in enumerate(row_pixels):
                    res[r, start_c + idx] = val
        elif direction == "left":
            for r in range(H):
                row_pixels = [grid.matrix[r, c] for c in range(W) if grid.matrix[r, c] != background_color]
                for idx, val in enumerate(row_pixels):
                    res[r, idx] = val

        return ARCGrid(res)

    # ── Topology & Hole Enclosure ──
    @staticmethod
    def fill_enclosed_holes(grid: ARCGrid, fill_color: int = 4, background_color: int = 0) -> ARCGrid:
        """Fills enclosed hollow chambers with fill_color."""
        holes = ARCPerception.detect_enclosed_holes(grid, background_color=background_color)
        res = grid.matrix.copy()
        for r, c in holes:
            res[r, c] = fill_color
        return ARCGrid(res)

    # ── Line Connecting & Spatial Alignment ──
    @staticmethod
    def connect_same_colors_with_lines(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Connects pairs of identical colored points along orthogonal axes in O(N) time."""
        res = grid.matrix.copy()
        colors = [c for c in np.unique(grid.matrix) if c != background_color]

        for color in colors:
            coords = np.argwhere(grid.matrix == color)
            row_groups: Dict[int, List[int]] = {}
            col_groups: Dict[int, List[int]] = {}
            for r, col in coords:
                row_groups.setdefault(int(r), []).append(int(col))
                col_groups.setdefault(int(col), []).append(int(r))

            for r, cols in row_groups.items():
                if len(cols) >= 2:
                    res[r, min(cols):max(cols) + 1] = color
            for col, rows in col_groups.items():
                if len(rows) >= 2:
                    res[min(rows):max(rows) + 1, col] = color
        return ARCGrid(res)

    @staticmethod
    def connect_lines_h_then_v(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """
        Connects identical color points along orthogonal axes,
        drawing all horizontal lines first across all colors,
        then drawing all vertical lines across all colors (vertical priority at intersections).
        Runs in O(N) time.
        """
        res = grid.matrix.copy()
        colors = [c for c in np.unique(grid.matrix) if c != background_color]
        # Horizontal lines first across all colors
        for c in colors:
            coords = np.argwhere(grid.matrix == c)
            row_groups: Dict[int, List[int]] = {}
            for r, col in coords:
                row_groups.setdefault(int(r), []).append(int(col))
            for r, cols in row_groups.items():
                if len(cols) >= 2:
                    res[r, min(cols):max(cols) + 1] = c

        # Vertical lines second across all colors
        for c in colors:
            coords = np.argwhere(grid.matrix == c)
            col_groups: Dict[int, List[int]] = {}
            for r, col in coords:
                col_groups.setdefault(int(col), []).append(int(r))
            for col, rows in col_groups.items():
                if len(rows) >= 2:
                    res[min(rows):max(rows) + 1, col] = c
        return ARCGrid(res)

    @staticmethod
    def connect_lines_v_then_h(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """
        Connects identical color points along orthogonal axes,
        drawing all vertical lines first across all colors,
        then drawing all horizontal lines across all colors (horizontal priority at intersections).
        Runs in O(N) time.
        """
        res = grid.matrix.copy()
        colors = [c for c in np.unique(grid.matrix) if c != background_color]
        # Vertical lines first across all colors
        for c in colors:
            coords = np.argwhere(grid.matrix == c)
            col_groups: Dict[int, List[int]] = {}
            for r, col in coords:
                col_groups.setdefault(int(col), []).append(int(r))
            for col, rows in col_groups.items():
                if len(rows) >= 2:
                    res[min(rows):max(rows) + 1, col] = c

        # Horizontal lines second across all colors
        for c in colors:
            coords = np.argwhere(grid.matrix == c)
            row_groups: Dict[int, List[int]] = {}
            for r, col in coords:
                row_groups.setdefault(int(r), []).append(int(col))
            for r, cols in row_groups.items():
                if len(cols) >= 2:
                    res[r, min(cols):max(cols) + 1] = c
        return ARCGrid(res)

    @staticmethod
    def extend_diagonal_arithmetic(grid: ARCGrid, ext_color: int = 2, background_color: int = 0) -> ARCGrid:
        """
        Detects pairs/sequences of identical colored pixels forming an arithmetic ray progression
        (r_0 + k*dr, c_0 + k*dc) and extends the ray across the grid using ext_color.
        """
        res = grid.matrix.copy()
        H, W = res.shape
        colors = [c for c in np.unique(grid.matrix) if c != background_color]
        for c in colors:
            coords = np.argwhere(grid.matrix == c)
            if len(coords) < 2:
                continue
            coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
            dr = coords[1, 0] - coords[0, 0]
            dc = coords[1, 1] - coords[0, 1]
            if dr == 0 and dc == 0:
                continue
            is_prog = True
            for k in range(2, len(coords)):
                if coords[k, 0] - coords[k - 1, 0] != dr or coords[k, 1] - coords[k - 1, 1] != dc:
                    is_prog = False
                    break
            if is_prog:
                r, col = coords[0, 0] - dr, coords[0, 1] - dc
                while 0 <= r < H and 0 <= col < W:
                    if res[r, col] == background_color:
                        res[r, col] = ext_color
                    r -= dr
                    col -= dc
                r, col = coords[-1, 0] + dr, coords[-1, 1] + dc
                while 0 <= r < H and 0 <= col < W:
                    if res[r, col] == background_color:
                        res[r, col] = ext_color
                    r += dr
                    col += dc
        return ARCGrid(res)

    @staticmethod
    def fractal_expand(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Kronecker product expansion using the non-background mask and the original grid motif."""
        mask = (grid.matrix != background_color).astype(int)
        return ARCGrid(np.kron(mask, grid.matrix))

    @staticmethod
    def fractal_complement_expand(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """
        Kronecker expansion where non-background pixels are replaced with the complement/inverted motif.
        """
        fg_colors = [c for c in np.unique(grid.matrix) if c != background_color]
        fg_color = fg_colors[0] if fg_colors else 1
        motif = np.where(grid.matrix == background_color, fg_color, background_color)
        mask = (grid.matrix != background_color).astype(int)
        return ARCGrid(np.kron(mask, motif))

    @staticmethod
    def project_rays_from_border_emitters(
        grid: ARCGrid,
        emitter_border: str = "bottom",
        background_color: int = 0,
    ) -> ARCGrid:
        """
        Border pixels act as laser emitters, firing rays through their lines.
        Each encountered pixel extends its color forward until the next obstacle or opposite edge.
        """
        res = grid.matrix.copy()
        H, W = grid.shape
        if emitter_border == "bottom":
            emitter_cols = np.where(grid.matrix[-1, :] != background_color)[0]
            for c in emitter_cols:
                seeds = [(r, grid.matrix[r, c]) for r in range(H) if grid.matrix[r, c] != background_color]
                for idx, (r, color) in enumerate(seeds):
                    stop_r = seeds[idx - 1][0] if idx > 0 else -1
                    for fill_r in range(r - 1, stop_r, -1):
                        res[fill_r, c] = color
        elif emitter_border == "top":
            emitter_cols = np.where(grid.matrix[0, :] != background_color)[0]
            for c in emitter_cols:
                seeds = [(r, grid.matrix[r, c]) for r in range(H) if grid.matrix[r, c] != background_color]
                for idx, (r, color) in enumerate(seeds):
                    stop_r = seeds[idx + 1][0] if idx + 1 < len(seeds) else H
                    for fill_r in range(r + 1, stop_r):
                        res[fill_r, c] = color
        elif emitter_border == "left":
            emitter_rows = np.where(grid.matrix[:, 0] != background_color)[0]
            for r in emitter_rows:
                seeds = [(c, grid.matrix[r, c]) for c in range(W) if grid.matrix[r, c] != background_color]
                for idx, (c, color) in enumerate(seeds):
                    stop_c = seeds[idx + 1][0] if idx + 1 < len(seeds) else W
                    for fill_c in range(c + 1, stop_c):
                        res[r, fill_c] = color
        elif emitter_border == "right":
            emitter_rows = np.where(grid.matrix[:, -1] != background_color)[0]
            for r in emitter_rows:
                seeds = [(c, grid.matrix[r, c]) for c in range(W) if grid.matrix[r, c] != background_color]
                for idx, (c, color) in enumerate(seeds):
                    stop_c = seeds[idx - 1][0] if idx > 0 else -1
                    for fill_c in range(c - 1, stop_c, -1):
                        res[r, fill_c] = color
        return ARCGrid(res)

    @staticmethod
    def draw_crosshairs_with_diagonal_reticle(
        grid: ARCGrid,
        seed_color: int = 1,
        center_color: int = 2,
        diag_color: int = 3,
        background_color: Optional[int] = None,
    ) -> ARCGrid:
        """
        Draws full orthogonal crosshair rays across the grid through every seed pixel,
        places diagonal reticles at (r±1, c±1), and recolors the center seed point.
        """
        res = grid.matrix.copy()
        H, W = grid.shape
        seeds = np.argwhere(grid.matrix == seed_color)
        if len(seeds) == 0:
            return grid.copy()
        for r, c in seeds:
            res[r, :] = seed_color
            res[:, c] = seed_color
        for r, c in seeds:
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        res[nr, nc] = diag_color
        for r, c in seeds:
            res[r, c] = center_color
        return ARCGrid(res)

    @staticmethod
    def project_monochromatic_stripe(grid: ARCGrid) -> ARCGrid:
        """
        Identifies a monochromatic row or column within the input grid,
        and projects the pattern across a 3x3 block expansion.
        """
        mat = grid.matrix
        H, W = mat.shape
        mono_r = None
        for r in range(H):
            if len(np.unique(mat[r, :])) == 1:
                mono_r = r
                break
        mono_c = None
        for c in range(W):
            if len(np.unique(mat[:, c])) == 1:
                mono_c = c
                break

        res = np.zeros((H * 3, W * 3), dtype=int)
        if mono_r is not None:
            row_start = mono_r * H
            tiled_row = np.tile(mat, (1, 3))
            res[row_start:row_start + H, :] = tiled_row
        elif mono_c is not None:
            col_start = mono_c * W
            tiled_col = np.tile(mat, (3, 1))
            res[:, col_start:col_start + W] = tiled_col
        return ARCGrid(res)

    @staticmethod
    def fill_periodic_from_indicator_bar(
        grid: ARCGrid,
        indicator_col: int = 0,
        indicator_color: int = 5,
    ) -> ARCGrid:
        """
        Measures the length K of an indicator bar in indicator_col,
        extracts the motif of the first K rows, and periodically tiles it
        into the remaining empty space at the bottom of the grid.
        """
        mat = grid.matrix
        res = mat.copy()
        H, W = mat.shape
        K = int(np.sum(mat[:, indicator_col] == indicator_color))
        if K <= 0 or K >= H:
            return grid.copy()
        motif = mat[:K, 1:].copy()
        first_empty_r = H
        for r in range(H):
            if np.all(mat[r, 1:] == 0):
                first_empty_r = r
                break
        curr_r = first_empty_r
        offset = 0
        while curr_r < H:
            res[curr_r, 1:] = motif[offset % K, :]
            curr_r += 1
            offset += 1
        return ARCGrid(res)

    # ── Pattern Tiling, Scaling & Repetition ──
    @staticmethod
    def tile_pattern(grid: ARCGrid, reps_h: int = 2, reps_w: int = 2) -> ARCGrid:
        """Replicates the grid as a tiled periodic wallpaper pattern."""
        return ARCGrid(np.tile(grid.matrix, (reps_h, reps_w)))

    @staticmethod
    def tile_mirror_2x2(grid: ARCGrid) -> ARCGrid:
        """4-quadrant reflection symmetry: [[rot180(x), flip_v(x)], [flip_h(x), x]]."""
        top = np.hstack([np.rot90(grid.matrix, 2), np.flipud(grid.matrix)])
        bottom = np.hstack([np.fliplr(grid.matrix), grid.matrix])
        return ARCGrid(np.vstack([top, bottom]))

    @staticmethod
    def tile_alternating_h_flip(grid: ARCGrid, reps_h: int = 3, reps_w: int = 3) -> ARCGrid:
        """Tiled grid with alternating rows flipped horizontally."""
        row_blocks = []
        for r in range(reps_h):
            base = np.fliplr(grid.matrix) if (r % 2 == 1) else grid.matrix
            row_blocks.append(np.hstack([base] * reps_w))
        return ARCGrid(np.vstack(row_blocks))

    @staticmethod
    def scale_grid(grid: ARCGrid, factor: int = 2) -> ARCGrid:
        """Scales each pixel into a factor x factor block."""
        return ARCGrid(np.repeat(np.repeat(grid.matrix, factor, axis=0), factor, axis=1))

    @staticmethod
    def repeat_horizontal(grid: ARCGrid, n: int = 2) -> ARCGrid:
        """Replicates grid horizontally n times."""
        return ARCGrid(np.tile(grid.matrix, (1, n)))

    @staticmethod
    def repeat_vertical(grid: ARCGrid, n: int = 2) -> ARCGrid:
        """Replicates grid vertically n times."""
        return ARCGrid(np.tile(grid.matrix, (n, 1)))

    @staticmethod
    def remove_duplicate_rows(grid: ARCGrid) -> ARCGrid:
        """Removes adjacent duplicate rows."""
        H, W = grid.shape
        if H <= 1:
            return grid.copy()
        keep = [0]
        for r in range(1, H):
            if not np.array_equal(grid.matrix[r], grid.matrix[r - 1]):
                keep.append(r)
        return ARCGrid(grid.matrix[keep, :])

    @staticmethod
    def remove_duplicate_cols(grid: ARCGrid) -> ARCGrid:
        """Removes adjacent duplicate columns."""
        H, W = grid.shape
        if W <= 1:
            return grid.copy()
        keep = [0]
        for c in range(1, W):
            if not np.array_equal(grid.matrix[:, c], grid.matrix[:, c - 1]):
                keep.append(c)
        return ARCGrid(grid.matrix[:, keep])

    @staticmethod
    def extend_pattern_right(grid: ARCGrid) -> ARCGrid:
        """Appends one period of repeating horizontal columns if periodic tiling detected."""
        rep = ARCPerception.find_repeating_pattern(grid)
        if rep and rep[1] < grid.width:
            period_cols = grid.matrix[:, :rep[1]]
            return ARCGrid(np.hstack([grid.matrix, period_cols]))
        return ARCGrid(np.hstack([grid.matrix, grid.matrix[:, -1:]]))

    @staticmethod
    def extend_pattern_down(grid: ARCGrid) -> ARCGrid:
        """Appends one period of repeating vertical rows if periodic tiling detected."""
        rep = ARCPerception.find_repeating_pattern(grid)
        if rep and rep[0] < grid.height:
            period_rows = grid.matrix[:rep[0], :]
            return ARCGrid(np.vstack([grid.matrix, period_rows]))
        return ARCGrid(np.vstack([grid.matrix, grid.matrix[-1:, :]]))

    # ── Frame / Border Operations ──
    @staticmethod
    def add_border(grid: ARCGrid, color: int = 1, width: int = 1) -> ARCGrid:
        """Wraps the grid in a colored perimeter border of specified thickness."""
        H, W = grid.shape
        res = np.full((H + 2 * width, W + 2 * width), color, dtype=int)
        res[width:width + H, width:width + W] = grid.matrix
        return ARCGrid(res)

    @staticmethod
    def remove_border(grid: ARCGrid, width: int = 1) -> ARCGrid:
        """Strips the outer perimeter border rows and columns."""
        H, W = grid.shape
        if H <= 2 * width or W <= 2 * width:
            return grid.copy()
        return ARCGrid(grid.matrix[width:H - width, width:W - width])

    @staticmethod
    def extract_border_ring(grid: ARCGrid, background_color: int = 0) -> ARCGrid:
        """Retains only the 1-pixel outer frame ring, clearing the interior to background."""
        res = np.full_like(grid.matrix, background_color)
        res[0, :] = grid.matrix[0, :]
        res[-1, :] = grid.matrix[-1, :]
        res[:, 0] = grid.matrix[:, 0]
        res[:, -1] = grid.matrix[:, -1]
        return ARCGrid(res)

    @staticmethod
    def fill_border(grid: ARCGrid, color: int = 1) -> ARCGrid:
        """Draws a 1-pixel colored border on the outermost cells without expanding grid shape."""
        res = grid.matrix.copy()
        res[0, :] = color
        res[-1, :] = color
        res[:, 0] = color
        res[:, -1] = color
        return ARCGrid(res)


# ─────────────────────────────────────────────────────────────
# 3. PROGRAM SYNTHESIS ENGINE (PHASE 3 INTELLIGENT SEARCH)
# ─────────────────────────────────────────────────────────────

@dataclass
class ARCTask:
    """An ARC-AGI task containing demonstration pairs and test inputs."""
    task_id: str
    category: str
    train_pairs: List[Tuple[ARCGrid, ARCGrid]]
    test_pairs: List[Tuple[ARCGrid, Optional[ARCGrid]]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], task_id: str = "custom", category: str = "general") -> "ARCTask":
        """
        Loads an ARCTask from the official ARC-AGI JSON dictionary structure:
        {"train": [{"input": [...], "output": [...]}], "test": [{"input": [...], "output": [...]}]}
        """
        train_pairs = []
        for pair in data.get("train", []):
            inp = ARCGrid(pair["input"])
            out = ARCGrid(pair["output"])
            train_pairs.append((inp, out))

        test_pairs = []
        for pair in data.get("test", []):
            inp = ARCGrid(pair["input"])
            out = ARCGrid(pair["output"]) if "output" in pair else None
            test_pairs.append((inp, out))

        return cls(
            task_id=task_id,
            category=category,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
        )

    @classmethod
    def from_json_file(cls, file_path: str, category: str = "general") -> "ARCTask":
        """Loads an ARCTask directly from a standalone official ARC-AGI task JSON file."""
        task_id = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, task_id=task_id, category=category)


@dataclass
class ARCSolution:
    """Result of ARC program synthesis."""
    task_id: str
    is_solved: bool
    synthesized_program: List[str]
    train_accuracy: float
    test_predictions: List[ARCGrid]
    latency_ms: float
    explanation: str

    @property
    def predicted_test_grids(self) -> List[np.ndarray]:
        return [g.matrix for g in self.test_predictions]

    @property
    def candidate_programs(self) -> List[str]:
        return self.synthesized_program

    @property
    def time_taken_ms(self) -> float:
        return self.latency_ms

    @property
    def confidence(self) -> float:
        return self.train_accuracy


@dataclass
class ARCOpDescriptor:
    """Metadata descriptor for a DSL operator facilitating IO-delta guided search pruning."""
    name: str
    fn: Callable[[ARCGrid], ARCGrid]
    shape_effect: str  # "preserve", "reduce", "expand", "variable"
    category: str  # "geometry", "symmetry", "color", "morphology", "object", "topology", "physics", "pattern", "border"
    colors_introduced: Set[int] = field(default_factory=set)
    colors_targeted: Set[int] = field(default_factory=set)


# ─────────────────────────────────────────────────────────────
# 4. OBJECT-CENTRIC MULTI-PROGRAM SYNTHESIS (PHASE 4)
# ─────────────────────────────────────────────────────────────

@dataclass
class ARCObjectRule:
    """A conditional transformation rule applied to individual segmented objects."""
    condition_type: str  # "all", "color_equals", "is_largest", "is_smallest", "is_solid_rectangle"
    condition_val: Any   # e.g., color int or None
    action_type: str     # "delete", "recolor", "rotate", "flip_h", "flip_v", "gravity_to_wall"
    action_val: Any      # e.g., "down", color int, etc.

    @property
    def name(self) -> str:
        cond_str = f"color=={self.condition_val}" if self.condition_type == "color_equals" else self.condition_type
        act_str = f"{self.action_type}({self.action_val})" if self.action_val is not None else self.action_type
        return f"ObjRule[{cond_str} -> {act_str}]"

    def matches(self, obj: ARCObject, all_objects: List[ARCObject]) -> bool:
        if self.condition_type == "all":
            return True
        elif self.condition_type == "color_equals":
            return obj.color == self.condition_val
        elif self.condition_type == "is_largest":
            if not all_objects:
                return False
            max_a = max(o.area for o in all_objects)
            return obj.area == max_a
        elif self.condition_type == "is_smallest":
            if not all_objects:
                return False
            min_a = min(o.area for o in all_objects)
            return obj.area == min_a
        elif self.condition_type == "is_solid_rectangle":
            return obj.is_solid_rectangle()
        return False

    def apply(self, obj: ARCObject, grid_shape: Tuple[int, int]) -> Optional[ARCObject]:
        if self.action_type == "delete":
            return None
        elif self.action_type == "recolor":
            return obj.recolor(int(self.action_val))
        elif self.action_type == "rotate":
            return obj.local_rotate(int(self.action_val))
        elif self.action_type == "flip_h":
            return obj.local_flip_h()
        elif self.action_type == "flip_v":
            return obj.local_flip_v()
        elif self.action_type == "gravity_to_wall":
            H, W = grid_shape
            r1, c1, r2, c2 = obj.bbox
            if self.action_val == "down":
                return obj.translate(H - 1 - r2, 0, target_shape=grid_shape)
            elif self.action_val == "up":
                return obj.translate(-r1, 0, target_shape=grid_shape)
            elif self.action_val == "left":
                return obj.translate(0, -c1, target_shape=grid_shape)
            elif self.action_val == "right":
                return obj.translate(0, W - 1 - c2, target_shape=grid_shape)
        return obj


class ObjectCentricSynthesizer:
    """
    Synthesizes object-level multi-program transformation rules for ARC tasks (Phase 4).
    Solves complex tasks where distinct segmented objects undergo distinct conditional transformations.
    """

    def evaluate_rule_set(
        self,
        rules: List[ARCObjectRule],
        train_pairs: Any,
    ) -> int:
        total_mismatch = 0
        if not train_pairs:
            return 999999
        # Accept precomputed list of (in_g, out_g, bg, objs) or raw train_pairs
        is_precomputed = (len(train_pairs[0]) == 4)
        for item in train_pairs:
            if is_precomputed:
                in_g, out_g, bg, objs = item
            else:
                in_g, out_g = item
                if in_g.shape != out_g.shape:
                    return 999999
                bg = ARCPerception.detect_background_color(in_g)
                objs = ARCPerception.extract_objects(in_g, background_color=bg)

            canvas = np.full(in_g.shape, bg, dtype=int)
            for obj in objs:
                curr_obj: Optional[ARCObject] = obj
                for rule in rules:
                    if curr_obj is not None and rule.matches(curr_obj, objs):
                        curr_obj = rule.apply(curr_obj, in_g.shape)
                if curr_obj is not None and curr_obj.area > 0:
                    curr_obj.render_onto(canvas)

            pred_grid = ARCGrid(canvas)
            total_mismatch += pred_grid.mismatch_count(out_g)
        return total_mismatch

    def synthesize(self, task: ARCTask, max_rules: int = 2) -> ARCSolution:
        t0 = time.perf_counter()
        train_pairs = task.train_pairs
        if not train_pairs:
            return ARCSolution(
                task_id=task.task_id,
                is_solved=False,
                synthesized_program=[],
                train_accuracy=0.0,
                test_predictions=[],
                latency_ms=0.0,
                explanation="No training demonstration pairs provided.",
            )

        # Early check: object rules require identical input and output dimensions
        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=False,
                    synthesized_program=[],
                    train_accuracy=0.0,
                    test_predictions=[],
                    latency_ms=dt_ms,
                    explanation="Object-centric synthesis requires shape-preserving tasks.",
                )

        # Precompute objects and background once per demonstration pair
        precomputed_train: List[Tuple[ARCGrid, ARCGrid, int, List[ARCObject]]] = []
        in_colors = set()
        out_colors = set()
        for in_g, out_g in train_pairs:
            in_colors.update(in_g.colors)
            out_colors.update(out_g.colors)
            bg = ARCPerception.detect_background_color(in_g)
            objs = ARCPerception.extract_objects(in_g, background_color=bg)
            precomputed_train.append((in_g, out_g, bg, objs))

        bg = precomputed_train[0][2]
        fg_in_colors = [c for c in in_colors if c != bg]

        # Candidate conditions
        conditions = [
            ("all", None),
            ("is_largest", None),
            ("is_smallest", None),
            ("is_solid_rectangle", None),
        ]
        for c in fg_in_colors:
            conditions.append(("color_equals", c))

        # Candidate actions
        actions = [
            ("delete", None),
            ("flip_h", None),
            ("flip_v", None),
            ("rotate", 1),
            ("rotate", 2),
        ]
        for c in out_colors:
            actions.append(("recolor", c))
        for d in ["down", "up", "left", "right"]:
            actions.append(("gravity_to_wall", d))

        # Build candidate atomic rules
        candidate_rules: List[ARCObjectRule] = []
        for cond_type, cond_val in conditions:
            for act_type, act_val in actions:
                if cond_type == "color_equals" and act_type == "recolor" and cond_val == act_val:
                    continue
                candidate_rules.append(
                    ARCObjectRule(
                        condition_type=cond_type,
                        condition_val=cond_val,
                        action_type=act_type,
                        action_val=act_val,
                    )
                )

        best_rules: Optional[List[ARCObjectRule]] = None
        best_cost = 999999

        # 1-rule search with precomputed objects
        for rule in candidate_rules:
            cost = self.evaluate_rule_set([rule], precomputed_train)
            if cost == 0:
                best_rules = [rule]
                best_cost = 0
                break

        # 2-rule search if not solved with 1 rule
        if best_cost != 0 and max_rules >= 2:
            scored_1 = []
            for rule in candidate_rules:
                cost = self.evaluate_rule_set([rule], precomputed_train)
                scored_1.append((cost, rule))
            scored_1.sort(key=lambda x: x[0])
            top_candidates = [r for _, r in scored_1[:25]]

            for i in range(len(top_candidates)):
                r1 = top_candidates[i]
                for j in range(i + 1, len(top_candidates)):
                    r2 = top_candidates[j]
                    if r1.condition_type == r2.condition_type and r1.condition_val == r2.condition_val:
                        continue
                    cost = self.evaluate_rule_set([r1, r2], precomputed_train)
                    if cost == 0:
                        best_rules = [r1, r2]
                        best_cost = 0
                        break
                if best_cost == 0:
                    break

        dt_ms = (time.perf_counter() - t0) * 1000.0
        is_solved = (best_cost == 0)
        rule_names = [r.name for r in best_rules] if best_rules else []

        # Predict on test inputs
        test_predictions = []
        for test_in, _ in task.test_pairs:
            if is_solved and best_rules:
                t_bg = ARCPerception.detect_background_color(test_in)
                t_canvas = np.full(test_in.shape, t_bg, dtype=int)
                t_objs = ARCPerception.extract_objects(test_in, background_color=t_bg)
                for obj in t_objs:
                    curr_obj: Optional[ARCObject] = obj
                    for rule in best_rules:
                        if curr_obj is not None and rule.matches(curr_obj, t_objs):
                            curr_obj = rule.apply(curr_obj, test_in.shape)
                    if curr_obj is not None and curr_obj.area > 0:
                        curr_obj.render_onto(t_canvas)
                test_predictions.append(ARCGrid(t_canvas))
            else:
                test_predictions.append(test_in.copy())

        explanation = (
            f"Object-Centric Solution: {' & '.join(rule_names)}"
            if is_solved else
            f"Object-centric synthesis exhausted (residual mismatch {best_cost})."
        )

        return ARCSolution(
            task_id=task.task_id,
            is_solved=is_solved,
            synthesized_program=rule_names,
            train_accuracy=1.0 if is_solved else 0.0,
            test_predictions=test_predictions,
            latency_ms=dt_ms,
            explanation=explanation,
        )


class ARCProgramSynthesizer:
    """
    Synthesizes exact Python transformation programs matching demonstration pairs
    via guided beam search over the ARC DSL with IO-delta intelligent pruning
    and multi-object conditional rule synthesis (Phase 4).
    """

    def __init__(self):
        self.descriptors: List[ARCOpDescriptor] = []
        self.object_synthesizer = ObjectCentricSynthesizer()


        # ── Base Geometric ──
        self._reg("rot90", ARCDSL.rot90, "preserve", "geometry")
        self._reg("rot180", ARCDSL.rot180, "preserve", "geometry")
        self._reg("rot270", ARCDSL.rot270, "preserve", "geometry")
        self._reg("flip_h", ARCDSL.flip_h, "preserve", "geometry")
        self._reg("flip_v", ARCDSL.flip_v, "preserve", "geometry")
        self._reg("transpose", ARCDSL.transpose, "preserve", "geometry")
        self._reg("diagonal_flip_anti", ARCDSL.diagonal_flip_anti, "preserve", "geometry")

        # ── Cropping & Halves ──
        self._reg("crop_to_content", ARCDSL.crop_to_content, "variable", "object")
        self._reg("crop_half_left", ARCDSL.crop_half_left, "reduce", "geometry")
        self._reg("crop_half_right", ARCDSL.crop_half_right, "reduce", "geometry")
        self._reg("crop_half_top", ARCDSL.crop_half_top, "reduce", "geometry")
        self._reg("crop_half_bottom", ARCDSL.crop_half_bottom, "reduce", "geometry")

        # ── Object Extraction & Manipulation ──
        self._reg("extract_largest_object", ARCDSL.extract_largest_object, "variable", "object")
        self._reg("extract_smallest_object", ARCDSL.extract_smallest_object, "variable", "object")
        self._reg("remove_smallest_object", ARCDSL.remove_smallest_object, "preserve", "object")
        self._reg("remove_largest_object", ARCDSL.remove_largest_object, "preserve", "object")
        self._reg("keep_largest_object", ARCDSL.keep_largest_object, "preserve", "object")
        self._reg("keep_smallest_object", ARCDSL.keep_smallest_object, "preserve", "object")
        self._reg("sort_objects_by_size", ARCDSL.sort_objects_by_size, "preserve", "object")
        self._reg("replace_color_by_object_size", ARCDSL.replace_color_by_object_size, "preserve", "object")
        self._reg("filter_objects_by_min_area_2", lambda g: ARCDSL.filter_objects_by_min_area(g, 2), "preserve", "object")

        # ── Morphological ──
        self._reg("dilate", ARCDSL.dilate, "preserve", "morphology")
        self._reg("erode", ARCDSL.erode, "preserve", "morphology")
        self._reg("extract_outline", ARCDSL.extract_outline, "preserve", "morphology")
        self._reg("fill_interior", ARCDSL.fill_interior, "preserve", "morphology")
        self._reg("hollow_objects", ARCDSL.hollow_objects, "preserve", "morphology")

        # ── Symmetry ──
        self._reg("mirror_left_to_right", ARCDSL.mirror_left_to_right, "preserve", "symmetry")
        self._reg("mirror_right_to_left", ARCDSL.mirror_right_to_left, "preserve", "symmetry")
        self._reg("mirror_top_to_bottom", ARCDSL.mirror_top_to_bottom, "preserve", "symmetry")
        self._reg("mirror_bottom_to_top", ARCDSL.mirror_bottom_to_top, "preserve", "symmetry")
        self._reg("complete_rotational_symmetry", ARCDSL.complete_rotational_symmetry, "preserve", "symmetry")
        self._reg("complete_horizontal_symmetry", ARCDSL.complete_horizontal_symmetry, "preserve", "symmetry")
        self._reg("complete_vertical_symmetry", ARCDSL.complete_vertical_symmetry, "preserve", "symmetry")

        # ── Physics & Cellular Mechanics ──
        self._reg("gravity_down", lambda g: ARCDSL.gravity(g, "down"), "preserve", "physics")
        self._reg("gravity_up", lambda g: ARCDSL.gravity(g, "up"), "preserve", "physics")
        self._reg("gravity_left", lambda g: ARCDSL.gravity(g, "left"), "preserve", "physics")
        self._reg("gravity_right", lambda g: ARCDSL.gravity(g, "right"), "preserve", "physics")

        # ── Lines & Connections ──
        self._reg("connect_same_colors_with_lines", ARCDSL.connect_same_colors_with_lines, "preserve", "geometry")
        self._reg("connect_lines_h_then_v", ARCDSL.connect_lines_h_then_v, "preserve", "geometry")
        self._reg("connect_lines_v_then_h", ARCDSL.connect_lines_v_then_h, "preserve", "geometry")
        self._reg("extend_diagonal_arithmetic", ARCDSL.extend_diagonal_arithmetic, "preserve", "geometry")

        # ── Pattern Tiling & Scaling ──
        self._reg("tile_2x2", lambda g: ARCDSL.tile_pattern(g, 2, 2), "expand", "pattern")
        self._reg("tile_3x3", lambda g: ARCDSL.tile_pattern(g, 3, 3), "expand", "pattern")
        self._reg("tile_mirror_2x2", ARCDSL.tile_mirror_2x2, "expand", "symmetry")
        self._reg("tile_alternating_3x3_h", lambda g: ARCDSL.tile_alternating_h_flip(g, 3, 3), "expand", "pattern")
        self._reg("repeat_horizontal_2", lambda g: ARCDSL.repeat_horizontal(g, 2), "expand", "pattern")
        self._reg("repeat_horizontal_3", lambda g: ARCDSL.repeat_horizontal(g, 3), "expand", "pattern")
        self._reg("repeat_vertical_2", lambda g: ARCDSL.repeat_vertical(g, 2), "expand", "pattern")
        self._reg("repeat_vertical_3", lambda g: ARCDSL.repeat_vertical(g, 3), "expand", "pattern")
        self._reg("scale_2x", lambda g: ARCDSL.scale_grid(g, 2), "expand", "pattern")
        self._reg("scale_3x", lambda g: ARCDSL.scale_grid(g, 3), "expand", "pattern")
        self._reg("fractal_expand", ARCDSL.fractal_expand, "expand", "pattern")
        self._reg("fractal_complement_expand", ARCDSL.fractal_complement_expand, "expand", "pattern")
        self._reg("downscale_majority_2", lambda g: ARCDSL.downscale_majority(g, 2), "reduce", "pattern")
        self._reg("downscale_majority_3", lambda g: ARCDSL.downscale_majority(g, 3), "reduce", "pattern")
        self._reg("remove_duplicate_rows", ARCDSL.remove_duplicate_rows, "reduce", "pattern")
        self._reg("remove_duplicate_cols", ARCDSL.remove_duplicate_cols, "reduce", "pattern")
        self._reg("extend_pattern_right", ARCDSL.extend_pattern_right, "expand", "pattern")
        self._reg("extend_pattern_down", ARCDSL.extend_pattern_down, "expand", "pattern")

        # ── Color & Filtering ──
        self._reg("filter_most_frequent_color", ARCDSL.filter_most_frequent_color, "preserve", "color")
        self._reg("filter_least_frequent_color", ARCDSL.filter_least_frequent_color, "preserve", "color")
        self._reg("invert_colors", ARCDSL.invert_colors, "preserve", "color")
        self._reg("map_most_to_least", ARCDSL.map_most_to_least, "preserve", "color")
        self._reg("unique_color_mask", ARCDSL.unique_color_mask, "preserve", "color")
        self._reg("binarize", ARCDSL.binarize, "preserve", "color")
        self._reg("replace_bg_with_dominant_fg", ARCDSL.replace_background_with_dominant_fg, "preserve", "color")

        # ── Frame / Border Operations ──
        self._reg("remove_border_1", lambda g: ARCDSL.remove_border(g, 1), "reduce", "border")
        self._reg("remove_border_2", lambda g: ARCDSL.remove_border(g, 2), "reduce", "border")
        self._reg("extract_border_ring", ARCDSL.extract_border_ring, "preserve", "border")

        # ── Parameterized Colors (1-9) ──
        for color in range(1, 10):
            c = color
            self._reg(f"fill_enclosed_holes_{c}", lambda g, c=c: ARCDSL.fill_enclosed_holes(g, c), "preserve", "topology", colors_introduced={c})
            self._reg(f"fill_background_{c}", lambda g, c=c: ARCDSL.fill_background(g, c), "preserve", "color", colors_introduced={c})
            self._reg(f"add_border_{c}", lambda g, c=c: ARCDSL.add_border(g, c, 1), "expand", "border", colors_introduced={c})
            self._reg(f"fill_border_{c}", lambda g, c=c: ARCDSL.fill_border(g, c), "preserve", "border", colors_introduced={c})
            self._reg(f"flood_fill_corners_{c}", lambda g, c=c: ARCDSL.flood_fill_from_corners(g, c), "preserve", "morphology", colors_introduced={c})
            self._reg(f"filter_by_color_{c}", lambda g, c=c: ARCDSL.filter_by_color(g, c), "preserve", "color", colors_targeted={c})
            self._reg(f"extend_diagonal_arithmetic_{c}", lambda g, c=c: ARCDSL.extend_diagonal_arithmetic(g, ext_color=c), "preserve", "geometry", colors_introduced={c})

        # ── Color-to-Color Remappings & Swaps ──
        for c1 in range(10):
            for c2 in range(10):
                if c1 != c2:
                    self._reg(
                        f"recolor_{c1}_to_{c2}",
                        lambda g, c1=c1, c2=c2: ARCDSL.recolor(g, c1, c2),
                        "preserve",
                        "color",
                        colors_targeted={c1},
                        colors_introduced={c2},
                    )
                if c1 < c2:
                    self._reg(
                        f"swap_colors_{c1}_{c2}",
                        lambda g, c1=c1, c2=c2: ARCDSL.swap_colors(g, c1, c2),
                        "preserve",
                        "color",
                        colors_targeted={c1, c2},
                    )

        # Standard atomic_ops tuple list for backward compatibility
        self.atomic_ops: List[Tuple[str, Callable[[ARCGrid], ARCGrid]]] = [
            (d.name, d.fn) for d in self.descriptors
        ]

    def _reg(
        self,
        name: str,
        fn: Callable[[ARCGrid], ARCGrid],
        shape_effect: str,
        category: str,
        colors_introduced: Optional[Set[int]] = None,
        colors_targeted: Optional[Set[int]] = None,
    ):
        desc = ARCOpDescriptor(
            name=name,
            fn=fn,
            shape_effect=shape_effect,
            category=category,
            colors_introduced=colors_introduced or set(),
            colors_targeted=colors_targeted or set(),
        )
        self.descriptors.append(desc)

    def select_candidate_ops(
        self,
        task: ARCTask,
    ) -> List[Tuple[str, Callable[[ARCGrid], ARCGrid]]]:
        """
        Intelligently filters and ranks candidate DSL operators based on
        comparative IO analysis across training demonstrations.
        Prunes the 200+ operation search space down to the 25-45 most relevant operators.
        """
        analysis = ARCPerception.analyze_io_relationship(task.train_pairs)
        same_shape = analysis.get("same_shape", True)
        output_is_smaller = analysis.get("output_is_smaller", False)
        output_is_larger = analysis.get("output_is_larger", False)
        colors_in = analysis.get("colors_in", set())
        colors_out = analysis.get("colors_out", set())
        colors_added = analysis.get("colors_added", set())
        colors_removed = analysis.get("colors_removed", set())
        colors_preserved = analysis.get("colors_preserved", False)
        has_holes_in = analysis.get("has_holes_in", False)
        symmetry_gained = analysis.get("symmetry_gained", False)
        scale_factor = analysis.get("scale_factor", None)

        scored_ops: List[Tuple[float, ARCOpDescriptor]] = []

        for desc in self.descriptors:
            score = 10.0

            # 1. Dimensionality filtering
            if same_shape:
                if desc.shape_effect in ["reduce", "expand"]:
                    continue
                score += 10.0
            elif output_is_smaller:
                if desc.shape_effect == "expand":
                    continue
                if desc.shape_effect == "reduce":
                    score += 30.0
                elif desc.shape_effect == "variable":
                    score += 20.0
            elif output_is_larger:
                if desc.shape_effect == "reduce":
                    continue
                if desc.shape_effect == "expand":
                    score += 30.0
                    if scale_factor and f"{scale_factor}x" in desc.name:
                        score += 40.0

            # 2. Color heuristics
            if colors_added:
                if desc.colors_introduced & colors_added:
                    score += 40.0
                elif desc.colors_introduced and not (desc.colors_introduced & colors_out):
                    continue

            if colors_preserved:
                if desc.colors_introduced and not desc.colors_introduced.issubset(colors_in):
                    continue
                if desc.category in ["geometry", "physics", "symmetry"]:
                    score += 15.0

            if desc.colors_targeted:
                if not (desc.colors_targeted & colors_in):
                    continue

            # 3. Symmetry heuristics
            if symmetry_gained and desc.category == "symmetry":
                score += 35.0

            # 4. Topological heuristics
            if has_holes_in and desc.category == "topology":
                score += 35.0

            # 5. Object removal heuristics
            if colors_removed and desc.category == "object":
                score += 25.0

            scored_ops.append((score, desc))

        # Sort descending by score
        scored_ops.sort(key=lambda x: x[0], reverse=True)

        max_candidates = 45
        selected = [(desc.name, desc.fn) for _, desc in scored_ops[:max_candidates]]

        # Fallback if pruning was too restrictive
        if len(selected) < 10:
            return self.atomic_ops[:35]

        return selected

    def evaluate_program(
        self,
        program_ops: List[Tuple[str, Callable[[ARCGrid], ARCGrid]]],
        train_pairs: List[Tuple[ARCGrid, ARCGrid]],
    ) -> int:
        """Computes total pixel mismatch cost across all training demonstration pairs."""
        total_mismatch = 0
        for in_grid, expected_out in train_pairs:
            curr = in_grid
            for _, fn in program_ops:
                try:
                    curr = fn(curr)
                except Exception:
                    return 999999
            total_mismatch += curr.mismatch_count(expected_out)
        return total_mismatch

    def _try_solve_kronecker(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes an exact Kronecker fractal expansion if output dimensions
        are an exact multiple of input dimensions across all training pairs.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        # Check if output is larger and shape is a multiple
        for in_grid, out_grid in train_pairs:
            if out_grid.height < in_grid.height or out_grid.width < in_grid.width:
                return None
            if out_grid.height % in_grid.height != 0 or out_grid.width % in_grid.width != 0:
                return None

        for op_name, op_fn in [
            ("fractal_complement_expand", ARCDSL.fractal_complement_expand),
            ("fractal_expand", ARCDSL.fractal_expand),
        ]:
            cost = self.evaluate_program([(op_name, op_fn)], train_pairs)
            if cost == 0:
                test_preds = []
                for test_in, _ in task.test_pairs:
                    try:
                        test_preds.append(op_fn(test_in))
                    except Exception:
                        test_preds.append(test_in.copy())
                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[op_name],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=0.0,
                    explanation=f"Exact Kronecker fractal synthesis: {op_name}",
                )
        return None

    def _try_solve_chamber_filling(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes an inductive topological rule that fills enclosed chambers
        based on chamber dimensions (height, width) or area.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        bg = ARCPerception.detect_background_color(train_pairs[0][0])
        dim_map: Dict[Tuple[int, int], int] = {}
        area_map: Dict[int, int] = {}
        has_chambers = False

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            chambers = ARCPerception.detect_enclosed_chambers(in_grid, background_color=bg)
            if not chambers:
                return None
            has_chambers = True

            for ch in chambers:
                rs = [p[0] for p in ch]
                cs = [p[1] for p in ch]
                dim = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
                area = len(ch)
                changed_colors = [
                    out_grid.matrix[r, c] for r, c in ch
                    if in_grid.matrix[r, c] == bg and out_grid.matrix[r, c] != bg
                ]
                if not changed_colors:
                    continue
                fill_c = int(max(set(changed_colors), key=changed_colors.count))

                if dim in dim_map and dim_map[dim] != fill_c:
                    return None
                dim_map[dim] = fill_c
                if area in area_map and area_map[area] != fill_c:
                    return None
                area_map[area] = fill_c

        if not has_chambers or not dim_map:
            return None

        def apply_chamber_fill(grid: ARCGrid) -> ARCGrid:
            res = grid.matrix.copy()
            chambers = ARCPerception.detect_enclosed_chambers(grid, background_color=bg)
            for ch in chambers:
                rs = [p[0] for p in ch]
                cs = [p[1] for p in ch]
                dim = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
                area = len(ch)
                color = dim_map.get(dim, area_map.get(area, None))
                if color is not None:
                    for r, c in ch:
                        if grid.matrix[r, c] == bg:
                            res[r, c] = color
            return ARCGrid(res)

        # Strict ground-truth validation across all train pairs
        for in_grid, out_grid in train_pairs:
            pred = apply_chamber_fill(in_grid)
            if pred.mismatch_count(out_grid) != 0:
                return None

        test_preds = [apply_chamber_fill(test_in) for test_in, _ in task.test_pairs]
        return ARCSolution(
            task_id=task.task_id,
            is_solved=True,
            synthesized_program=["fill_chambers_by_size"],
            train_accuracy=1.0,
            test_predictions=test_preds,
            latency_ms=0.0,
            explanation=f"Inductive topological rule: filled chambers based on dimension/area mapping {dim_map}.",
        )

    def _try_solve_border_emitters(self, task: ARCTask) -> Optional[ARCSolution]:
        """Solves tasks where border pixels act as laser emitters projecting rays."""
        train_pairs = task.train_pairs
        if not train_pairs:
            return None
        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        for border in ["bottom", "top", "left", "right"]:
            op = (
                f"project_rays_from_{border}_emitters",
                lambda g, b=border: ARCDSL.project_rays_from_border_emitters(g, emitter_border=b)
            )
            cost = self.evaluate_program([op], train_pairs)
            if cost == 0:
                test_preds = [op[1](test_in) for test_in, _ in task.test_pairs]
                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[op[0]],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=0.0,
                    explanation=f"Border laser emitter projection from {border} edge.",
                )
        return None

    def _try_solve_crosshairs_reticle(self, task: ARCTask) -> Optional[ARCSolution]:
        """Solves tasks where seed pixels form crosshair rays with diagonal reticle markers."""
        train_pairs = task.train_pairs
        if not train_pairs:
            return None
        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        in_colors = set()
        for in_g, _ in train_pairs:
            in_colors.update(in_g.colors)

        out_colors = set()
        for _, out_g in train_pairs:
            out_colors.update(out_g.colors)

        for seed_c in in_colors:
            for center_c in out_colors:
                for diag_c in out_colors:
                    op = (
                        f"crosshairs_seed_{seed_c}_center_{center_c}_diag_{diag_c}",
                        lambda g, sc=seed_c, cc=center_c, dc=diag_c: ARCDSL.draw_crosshairs_with_diagonal_reticle(
                            g, seed_color=sc, center_color=cc, diag_color=dc
                        )
                    )
                    cost = self.evaluate_program([op], train_pairs)
                    if cost == 0:
                        test_preds = [op[1](test_in) for test_in, _ in task.test_pairs]
                        return ARCSolution(
                            task_id=task.task_id,
                            is_solved=True,
                            synthesized_program=[op[0]],
                            train_accuracy=1.0,
                            test_predictions=test_preds,
                            latency_ms=0.0,
                            explanation=f"Crosshairs with diagonal reticle: seed={seed_c}, center={center_c}, diag={diag_c}.",
                        )
        return None

    def _try_solve_monochromatic_stripe(self, task: ARCTask) -> Optional[ARCSolution]:
        """Solves tasks where a 3x3 pattern is replicated across a 9x9 grid along the axis of its monochromatic stripe."""
        train_pairs = task.train_pairs
        if not train_pairs:
            return None
        for in_g, out_g in train_pairs:
            if out_g.height != in_g.height * 3 or out_g.width != in_g.width * 3:
                return None

        op = ("project_monochromatic_stripe", ARCDSL.project_monochromatic_stripe)
        cost = self.evaluate_program([op], train_pairs)
        if cost == 0:
            test_preds = [op[1](test_in) for test_in, _ in task.test_pairs]
            return ARCSolution(
                task_id=task.task_id,
                is_solved=True,
                synthesized_program=[op[0]],
                train_accuracy=1.0,
                test_predictions=test_preds,
                latency_ms=0.0,
                explanation="Monochromatic stripe projection and 3x block replication.",
            )
        return None

    def _try_solve_indicator_periodic(self, task: ARCTask) -> Optional[ARCSolution]:
        """Solves tasks where an indicator bar specifies the repeating motif period to tile downward."""
        train_pairs = task.train_pairs
        if not train_pairs:
            return None
        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        for ind_col in [0, -1]:
            for ind_color in range(1, 10):
                op = (
                    f"fill_periodic_ind_{ind_col}_{ind_color}",
                    lambda g, ic=ind_col, clr=ind_color: ARCDSL.fill_periodic_from_indicator_bar(
                        g, indicator_col=ic, indicator_color=clr
                    )
                )
                cost = self.evaluate_program([op], train_pairs)
                if cost == 0:
                    test_preds = [op[1](test_in) for test_in, _ in task.test_pairs]
                    return ARCSolution(
                        task_id=task.task_id,
                        is_solved=True,
                        synthesized_program=[op[0]],
                        train_accuracy=1.0,
                        test_predictions=test_preds,
                        latency_ms=0.0,
                        explanation=f"Periodic motif tiling from indicator bar in col {ind_col} (color {ind_color}).",
                    )
    def _try_solve_topological_feature_mapping(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes an object-level topological or geometric invariant feature recoloring rule.
        Solves tasks where connected objects are recolored strictly by their number of cavities
        (Euler characteristic), area, aspect ratio, or bounding dimension.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        features_to_try = [
            "num_holes", "area", "hole_pixels", "height", "width", "max_dim", "min_dim", "is_square"
        ]

        from scipy.ndimage import label, binary_fill_holes

        for feat_name in features_to_try:
            feat_map: Dict[int, int] = {}
            conflict = False
            all_objs_valid = True

            for in_g, out_g in train_pairs:
                inp = in_g.matrix
                out = out_g.matrix
                lbl, num = label(inp > 0)
                if num == 0:
                    all_objs_valid = False
                    break

                for obj_id in range(1, num + 1):
                    mask = (lbl == obj_id)
                    out_colors = np.unique(out[mask])
                    if len(out_colors) != 1:
                        conflict = True
                        break
                    target_c = int(out_colors[0])

                    if feat_name == "num_holes":
                        filled = binary_fill_holes(mask)
                        holes_mask = filled & (~mask)
                        _, val = label(holes_mask)
                    elif feat_name == "area":
                        val = int(np.sum(mask))
                    elif feat_name == "hole_pixels":
                        filled = binary_fill_holes(mask)
                        val = int(np.sum(filled) - np.sum(mask))
                    elif feat_name == "height":
                        rs = np.where(mask)[0]
                        val = int(rs.max() - rs.min() + 1)
                    elif feat_name == "width":
                        cs = np.where(mask)[1]
                        val = int(cs.max() - cs.min() + 1)
                    elif feat_name == "max_dim":
                        rs = np.where(mask)[0]
                        cs = np.where(mask)[1]
                        val = int(max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                    elif feat_name == "min_dim":
                        rs = np.where(mask)[0]
                        cs = np.where(mask)[1]
                        val = int(min(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                    elif feat_name == "is_square":
                        rs = np.where(mask)[0]
                        cs = np.where(mask)[1]
                        val = int((rs.max() - rs.min() + 1) == (cs.max() - cs.min() + 1))
                    else:
                        val = 0

                    if val in feat_map and feat_map[val] != target_c:
                        conflict = True
                        break
                    feat_map[val] = target_c

                if conflict:
                    break

            if not conflict and all_objs_valid and len(feat_map) > 0:
                train_ok = True
                for in_g, out_g in train_pairs:
                    inp = in_g.matrix
                    out = out_g.matrix
                    lbl, num = label(inp > 0)
                    pred = np.zeros_like(inp)
                    for obj_id in range(1, num + 1):
                        mask = (lbl == obj_id)
                        if feat_name == "num_holes":
                            filled = binary_fill_holes(mask)
                            holes_mask = filled & (~mask)
                            _, val = label(holes_mask)
                        elif feat_name == "area":
                            val = int(np.sum(mask))
                        elif feat_name == "hole_pixels":
                            filled = binary_fill_holes(mask)
                            val = int(np.sum(filled) - np.sum(mask))
                        elif feat_name == "height":
                            rs = np.where(mask)[0]
                            val = int(rs.max() - rs.min() + 1)
                        elif feat_name == "width":
                            cs = np.where(mask)[1]
                            val = int(cs.max() - cs.min() + 1)
                        elif feat_name == "max_dim":
                            rs = np.where(mask)[0]
                            cs = np.where(mask)[1]
                            val = int(max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                        elif feat_name == "min_dim":
                            rs = np.where(mask)[0]
                            cs = np.where(mask)[1]
                            val = int(min(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                        elif feat_name == "is_square":
                            rs = np.where(mask)[0]
                            cs = np.where(mask)[1]
                            val = int((rs.max() - rs.min() + 1) == (cs.max() - cs.min() + 1))
                        else:
                            val = 0

                        if val not in feat_map:
                            train_ok = False
                            break
                        pred[mask] = feat_map[val]

                    if not np.array_equal(pred, out):
                        train_ok = False
                        break

                if train_ok:
                    test_preds = []
                    for test_in, _ in task.test_pairs:
                        inp = test_in.matrix
                        lbl, num = label(inp > 0)
                        pred = np.zeros_like(inp)
                        for obj_id in range(1, num + 1):
                            mask = (lbl == obj_id)
                            if feat_name == "num_holes":
                                filled = binary_fill_holes(mask)
                                holes_mask = filled & (~mask)
                                _, val = label(holes_mask)
                            elif feat_name == "area":
                                val = int(np.sum(mask))
                            elif feat_name == "hole_pixels":
                                filled = binary_fill_holes(mask)
                                val = int(np.sum(filled) - np.sum(mask))
                            elif feat_name == "height":
                                rs = np.where(mask)[0]
                                val = int(rs.max() - rs.min() + 1)
                            elif feat_name == "width":
                                cs = np.where(mask)[1]
                                val = int(cs.max() - cs.min() + 1)
                            elif feat_name == "max_dim":
                                rs = np.where(mask)[0]
                                cs = np.where(mask)[1]
                                val = int(max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                            elif feat_name == "min_dim":
                                rs = np.where(mask)[0]
                                cs = np.where(mask)[1]
                                val = int(min(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))
                            elif feat_name == "is_square":
                                rs = np.where(mask)[0]
                                cs = np.where(mask)[1]
                                val = int((rs.max() - rs.min() + 1) == (cs.max() - cs.min() + 1))
                            else:
                                val = 0

                            pred[mask] = feat_map.get(val, 1)
                        test_preds.append(ARCGrid(pred))

                    return ARCSolution(
                        task_id=task.task_id,
                        is_solved=True,
                        synthesized_program=[f"recolor_by_{feat_name}"],
                        train_accuracy=1.0,
                        test_predictions=test_preds,
                        latency_ms=0.0,
                        explanation=f"Topological/geometric feature mapping: {feat_name} -> {feat_map}",
                    )
        return None

    def _try_solve_unique_anomaly_object(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes an anomaly / odd-one-out object extraction rule.
        Detects tasks where multiple candidate objects exist of a given color, and the output
        is the bounding box crop of the unique object with an outlier feature value.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        # Precondition: output must be strictly smaller than input
        for in_g, out_g in train_pairs:
            if out_g.height >= in_g.height and out_g.width >= in_g.width:
                return None

        from scipy.ndimage import label, binary_fill_holes
        from collections import Counter

        features = ["num_holes", "area", "height", "width"]

        for feat_name in features:
            train_ok = True
            for in_g, out_g in train_pairs:
                inp = in_g.matrix
                expected = out_g.matrix
                colors = [c for c in np.unique(inp) if c != 0]
                matched_pair = False
                for c in colors:
                    c_mask = (inp == c)
                    lbl, num = label(c_mask)
                    if num <= 1:
                        continue
                    objs = []
                    for o in range(1, num + 1):
                        m = (lbl == o)
                        if feat_name == "num_holes":
                            filled = binary_fill_holes(m)
                            _, val = label(filled & ~m)
                        elif feat_name == "area":
                            val = int(np.sum(m))
                        elif feat_name == "height":
                            rs = np.where(m)[0]
                            val = int(rs.max() - rs.min() + 1)
                        elif feat_name == "width":
                            cs = np.where(m)[1]
                            val = int(cs.max() - cs.min() + 1)
                        else:
                            val = 0
                        rs, cs = np.where(m)
                        crop = inp[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
                        objs.append((val, crop))

                    counts = Counter([v for v, _ in objs])
                    uniques = [v for v, cnt in counts.items() if cnt == 1]
                    if len(uniques) == 1:
                        target_val = uniques[0]
                        for val, crop in objs:
                            if val == target_val and np.array_equal(crop, expected):
                                matched_pair = True
                                break
                    if matched_pair:
                        break
                if not matched_pair:
                    train_ok = False
                    break

            if train_ok:
                test_preds = []
                for test_in, _ in task.test_pairs:
                    inp = test_in.matrix
                    colors = [c for c in np.unique(inp) if c != 0]
                    matched_test = False
                    for c in colors:
                        c_mask = (inp == c)
                        lbl, num = label(c_mask)
                        if num <= 1:
                            continue
                        objs = []
                        for o in range(1, num + 1):
                            m = (lbl == o)
                            if feat_name == "num_holes":
                                filled = binary_fill_holes(m)
                                _, val = label(filled & ~m)
                            elif feat_name == "area":
                                val = int(np.sum(m))
                            elif feat_name == "height":
                                rs = np.where(m)[0]
                                val = int(rs.max() - rs.min() + 1)
                            elif feat_name == "width":
                                cs = np.where(m)[1]
                                val = int(cs.max() - cs.min() + 1)
                            else:
                                val = 0
                            rs, cs = np.where(m)
                            crop = inp[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
                            objs.append((val, crop))

                        counts = Counter([v for v, _ in objs])
                        uniques = [v for v, cnt in counts.items() if cnt == 1]
                        if len(uniques) == 1:
                            target_val = uniques[0]
                            for val, crop in objs:
                                if val == target_val:
                                    test_preds.append(ARCGrid(crop))
                                    matched_test = True
                                    break
                        if matched_test:
                            break
                    if not matched_test:
                        test_preds.append(test_in.copy())

                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[f"extract_unique_anomaly_{feat_name}"],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=0.0,
                    explanation=f"Extracted unique odd-one-out object by feature: {feat_name}",
                )
        return None

    def _try_solve_denoised_spatial_blocks(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes a spatial layout of dominant monochromatic rectangular blocks,
        ignoring single-pixel background noise.
        Solves tasks where large colored blocks in an NxM spatial pattern define the output grid.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        # Output must be strictly smaller than input
        for in_g, out_g in train_pairs:
            if out_g.height >= in_g.height or out_g.width >= in_g.width:
                return None

        from scipy.ndimage import label

        def extract_grid(grid: ARCGrid) -> Optional[np.ndarray]:
            inp = grid.matrix
            rects = []
            for c in range(1, 10):
                lbl, num = label(inp == c)
                for o in range(1, num + 1):
                    m = (lbl == o)
                    rs, cs = np.where(m)
                    h = rs.max() - rs.min() + 1
                    w = cs.max() - cs.min() + 1
                    if np.all(inp[rs.min():rs.max() + 1, cs.min():cs.max() + 1] == c):
                        if h >= 2 and w >= 2 and np.sum(m) >= 6:
                            center_r = (rs.min() + rs.max()) / 2.0
                            center_c = (cs.min() + cs.max()) / 2.0
                            rects.append((c, center_r, center_c))
            if not rects:
                return None
            rects.sort(key=lambda x: x[1])
            rows = []
            curr_row = [rects[0]]
            for r in rects[1:]:
                if r[1] - curr_row[-1][1] > 2.5:
                    rows.append(curr_row)
                    curr_row = [r]
                else:
                    curr_row.append(r)
            rows.append(curr_row)
            grid_res = []
            row_lens = set()
            for row in rows:
                row.sort(key=lambda x: x[2])
                grid_res.append([x[0] for x in row])
                row_lens.add(len(row))
            if len(row_lens) != 1:
                return None
            return np.array(grid_res)

        # Check train demonstrations
        for in_g, out_g in train_pairs:
            pred = extract_grid(in_g)
            if pred is None or not np.array_equal(pred, out_g.matrix):
                return None

        # Apply to test
        test_preds = []
        for test_in, _ in task.test_pairs:
            pred = extract_grid(test_in)
            if pred is None:
                test_preds.append(test_in.copy())
            else:
                test_preds.append(ARCGrid(pred))

        return ARCSolution(
            task_id=task.task_id,
            is_solved=True,
            synthesized_program=["denoised_spatial_block_grid"],
            train_accuracy=1.0,
            test_predictions=test_preds,
            latency_ms=0.0,
            explanation="Extracted relative spatial layout of dominant monochromatic rectangular blocks.",
        )

    def _try_solve_subgrid_delimiter_logic(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes subgrid boolean logic across partitions created by solid delimiter lines.
        Detects uniform vertical or horizontal separator lines and applies XOR, OR, or AND
        recombination with learned target coloring.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        for orient in ["vertical", "horizontal"]:
            for op_type in ["xor", "or", "and"]:
                train_ok = True
                learned_out_color = None
                sep_color = None

                for in_g, out_g in train_pairs:
                    inp = in_g.matrix
                    expected = out_g.matrix

                    found_split = False
                    for c in range(1, 10):
                        if orient == "vertical":
                            cols = np.where(np.all(inp == c, axis=0))[0]
                            if len(cols) == 1:
                                sep = cols[0]
                                sub1 = inp[:, :sep]
                                sub2 = inp[:, sep + 1:]
                                if sub1.shape == sub2.shape and sub1.shape == expected.shape:
                                    found_split = True
                                    sep_color = c
                                    break
                        else:
                            rows = np.where(np.all(inp == c, axis=1))[0]
                            if len(rows) == 1:
                                sep = rows[0]
                                sub1 = inp[:sep, :]
                                sub2 = inp[sep + 1:, :]
                                if sub1.shape == sub2.shape and sub1.shape == expected.shape:
                                    found_split = True
                                    sep_color = c
                                    break
                    if not found_split:
                        train_ok = False
                        break

                    if op_type == "xor":
                        mask = (sub1 > 0) ^ (sub2 > 0)
                    elif op_type == "or":
                        mask = (sub1 > 0) | (sub2 > 0)
                    elif op_type == "and":
                        mask = (sub1 > 0) & (sub2 > 0)
                    else:
                        mask = np.zeros_like(expected, dtype=bool)

                    exp_mask = (expected > 0)
                    if not np.array_equal(mask, exp_mask):
                        train_ok = False
                        break

                    out_colors = np.unique(expected[mask])
                    if len(out_colors) == 1:
                        c_val = int(out_colors[0])
                        if learned_out_color is not None and learned_out_color != c_val:
                            train_ok = False
                            break
                        learned_out_color = c_val
                    else:
                        train_ok = False
                        break

                if train_ok and learned_out_color is not None:
                    test_preds = []
                    for test_in, _ in task.test_pairs:
                        inp = test_in.matrix
                        found = False
                        sub1 = sub2 = None
                        for c in ([sep_color] if sep_color is not None else range(1, 10)):
                            if orient == "vertical":
                                cols = np.where(np.all(inp == c, axis=0))[0]
                                if len(cols) == 1:
                                    sep = cols[0]
                                    sub1 = inp[:, :sep]
                                    sub2 = inp[:, sep + 1:]
                                    found = True
                                    break
                            else:
                                rows = np.where(np.all(inp == c, axis=1))[0]
                                if len(rows) == 1:
                                    sep = rows[0]
                                    sub1 = inp[:sep, :]
                                    sub2 = inp[sep + 1:, :]
                                    found = True
                                    break
                        if not found or sub1 is None or sub2 is None:
                            test_preds.append(test_in.copy())
                            continue
                        if op_type == "xor":
                            mask = (sub1 > 0) ^ (sub2 > 0)
                        elif op_type == "or":
                            mask = (sub1 > 0) | (sub2 > 0)
                        elif op_type == "and":
                            mask = (sub1 > 0) & (sub2 > 0)
                        else:
                            mask = np.zeros_like(sub1, dtype=bool)
                        pred = np.where(mask, learned_out_color, 0)
                        test_preds.append(ARCGrid(pred))

                    return ARCSolution(
                        task_id=task.task_id,
                        is_solved=True,
                        synthesized_program=[f"subgrid_{orient}_{op_type}_color_{learned_out_color}"],
                        train_accuracy=1.0,
                        test_predictions=test_preds,
                        latency_ms=0.0,
                        explanation=f"Subgrid delimiter {orient} split recombined via {op_type.upper()} recolored with {learned_out_color}.",
                    )
        return None

    def _try_solve_box_crosshairs_fill(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes a rule where exactly two seed points of the same color define a
        bounding box chamber: crosshair lines are drawn through both seeds, and the interior
        cavity is filled with a learned fill color.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        for fill_c in range(1, 10):
            train_ok = True
            for in_g, out_g in train_pairs:
                inp, expected = in_g.matrix, out_g.matrix
                pts = np.argwhere(inp > 0)
                if len(pts) != 2:
                    train_ok = False
                    break
                (r1, c1), (r2, c2) = pts
                c_val = inp[r1, c1]
                if inp[r2, c2] != c_val:
                    train_ok = False
                    break
                min_r, max_r = min(r1, r2), max(r1, r2)
                min_c, max_c = min(c1, c2), max(c1, c2)
                pred = np.zeros_like(inp)
                pred[min_r, :] = c_val
                pred[max_r, :] = c_val
                pred[:, min_c] = c_val
                pred[:, max_c] = c_val
                if max_r - min_r > 1 and max_c - min_c > 1:
                    pred[min_r + 1:max_r, min_c + 1:max_c] = fill_c
                if not np.array_equal(pred, expected):
                    train_ok = False
                    break

            if train_ok:
                test_preds = []
                for test_in, _ in task.test_pairs:
                    inp = test_in.matrix
                    pts = np.argwhere(inp > 0)
                    if len(pts) != 2:
                        test_preds.append(test_in.copy())
                        continue
                    (r1, c1), (r2, c2) = pts
                    c_val = inp[r1, c1]
                    min_r, max_r = min(r1, r2), max(r1, r2)
                    min_c, max_c = min(c1, c2), max(c1, c2)
                    pred = np.zeros_like(inp)
                    pred[min_r, :] = c_val
                    pred[max_r, :] = c_val
                    pred[:, min_c] = c_val
                    pred[:, max_c] = c_val
                    if max_r - min_r > 1 and max_c - min_c > 1:
                        pred[min_r + 1:max_r, min_c + 1:max_c] = fill_c
                    test_preds.append(ARCGrid(pred))

                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[f"box_crosshairs_fill_{fill_c}"],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=0.0,
                    explanation=f"Box crosshairs bounding chamber filled with color {fill_c}.",
                )
        return None

    def _try_solve_multi_point_crosshair_intersections(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes a crosshair grid where multiple seeds cast rays along rows and columns,
        preserving original seed pixels, while all new intersection vertices are recolored with
        a learned intersection color.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        for in_g, out_g in train_pairs:
            if in_g.shape != out_g.shape:
                return None

        for int_c in range(1, 10):
            train_ok = True
            for in_g, out_g in train_pairs:
                inp, expected = in_g.matrix, out_g.matrix
                pts = np.argwhere(inp > 0)
                if len(pts) < 2:
                    train_ok = False
                    break
                row_colors = {r: inp[r, c] for r, c in pts}
                col_colors = {c: inp[r, c] for r, c in pts}
                seed_set = set((r, c) for r, c in pts)
                pred = np.zeros_like(inp)
                for r, color in row_colors.items():
                    pred[r, :] = color
                for c, color in col_colors.items():
                    pred[:, c] = color
                for r in row_colors:
                    for c in col_colors:
                        if (r, c) not in seed_set:
                            pred[r, c] = int_c
                        else:
                            pred[r, c] = inp[r, c]
                if not np.array_equal(pred, expected):
                    train_ok = False
                    break

            if train_ok:
                test_preds = []
                for test_in, _ in task.test_pairs:
                    inp = test_in.matrix
                    pts = np.argwhere(inp > 0)
                    row_colors = {r: inp[r, c] for r, c in pts}
                    col_colors = {c: inp[r, c] for r, c in pts}
                    seed_set = set((r, c) for r, c in pts)
                    pred = np.zeros_like(inp)
                    for r, color in row_colors.items():
                        pred[r, :] = color
                    for c, color in col_colors.items():
                        pred[:, c] = color
                    for r in row_colors:
                        for c in col_colors:
                            if (r, c) not in seed_set:
                                pred[r, c] = int_c
                            else:
                                pred[r, c] = inp[r, c]
                    test_preds.append(ARCGrid(pred))

                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[f"crosshair_intersections_color_{int_c}"],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=0.0,
                    explanation=f"Crosshairs projected with intersection vertices recolored with {int_c}.",
                )
        return None

    def _try_solve_traversing_line_color(self, task: ARCTask) -> Optional[ARCSolution]:
        """
        Synthesizes a 1x1 color detection rule: identifies the unique color of the complete
        uninterrupted row or column that traverses the canvas.
        """
        train_pairs = task.train_pairs
        if not train_pairs:
            return None

        for _, out_g in train_pairs:
            if out_g.shape != (1, 1):
                return None

        train_ok = True
        for in_g, out_g in train_pairs:
            inp = in_g.matrix
            out_c = out_g.matrix[0, 0]
            found_c = None
            for c in range(1, 10):
                if np.any(np.all(inp == c, axis=1)) or np.any(np.all(inp == c, axis=0)):
                    found_c = c
                    break
            if found_c is None or found_c != out_c:
                train_ok = False
                break

        if train_ok:
            test_preds = []
            for test_in, _ in task.test_pairs:
                inp = test_in.matrix
                found_c = None
                for c in range(1, 10):
                    if np.any(np.all(inp == c, axis=1)) or np.any(np.all(inp == c, axis=0)):
                        found_c = c
                        break
                if found_c is None:
                    test_preds.append(test_in.copy())
                else:
                    test_preds.append(ARCGrid(np.array([[found_c]])))

            return ARCSolution(
                task_id=task.task_id,
                is_solved=True,
                synthesized_program=["traversing_line_color_1x1"],
                train_accuracy=1.0,
                test_predictions=test_preds,
                latency_ms=0.0,
                explanation="Detected color of full traversing grid row/column.",
            )
        return None

    def synthesize(
        self,
        task: ARCTask,
        max_depth: int = 3,
        max_beam_width: int = 15,
        use_intelligent_pruning: bool = True,
    ) -> ARCSolution:
        """
        Executes analysis-guided beam search over the ARC DSL to synthesize an exact program.
        """
        t0 = time.perf_counter()
        train_pairs = task.train_pairs

        # Quick check: Identity
        if self.evaluate_program([], train_pairs) == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return ARCSolution(
                task_id=task.task_id,
                is_solved=True,
                synthesized_program=["identity"],
                train_accuracy=1.0,
                test_predictions=[test_in.copy() for test_in, _ in task.test_pairs],
                latency_ms=dt_ms,
                explanation="Input directly matches output (Identity transformation).",
            )

        # Select candidate operations
        if use_intelligent_pruning:
            candidate_ops = self.select_candidate_ops(task)
        else:
            candidate_ops = self.atomic_ops

        # Fast Tier 1: Evaluate depth 1 candidates
        for op_name, op_fn in candidate_ops:
            cost = self.evaluate_program([(op_name, op_fn)], train_pairs)
            if cost == 0:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                test_preds = []
                for test_in, _ in task.test_pairs:
                    try:
                        test_preds.append(op_fn(test_in))
                    except Exception:
                        test_preds.append(test_in.copy())
                return ARCSolution(
                    task_id=task.task_id,
                    is_solved=True,
                    synthesized_program=[op_name],
                    train_accuracy=1.0,
                    test_predictions=test_preds,
                    latency_ms=dt_ms,
                    explanation=f"Synthesized 1-step program: {op_name}",
                )

        # Tier 1.5: Inductive solvers for specialized mathematical structures
        for solver_fn in [
            self._try_solve_kronecker,
            self._try_solve_chamber_filling,
            self._try_solve_border_emitters,
            self._try_solve_crosshairs_reticle,
            self._try_solve_monochromatic_stripe,
            self._try_solve_indicator_periodic,
            self._try_solve_topological_feature_mapping,
            self._try_solve_unique_anomaly_object,
            self._try_solve_denoised_spatial_blocks,
            self._try_solve_subgrid_delimiter_logic,
            self._try_solve_box_crosshairs_fill,
            self._try_solve_multi_point_crosshair_intersections,
            self._try_solve_traversing_line_color,
        ]:
            sol = solver_fn(task)
            if sol is not None:
                sol.latency_ms = (time.perf_counter() - t0) * 1000.0
                return sol

        # Tier 2: Beam Search up to max_depth
        beam: List[Tuple[int, List[Tuple[str, Callable[[ARCGrid], ARCGrid]]]]] = [
            (self.evaluate_program([], train_pairs), [])
        ]

        best_program = None
        best_cost = 999999
        effective_beam_width = max(max_beam_width, 15)

        for depth in range(1, max_depth + 1):
            if (time.perf_counter() - t0) > 2.5:
                break
            candidates = []
            for cost, prog in beam:
                for op_name, op_fn in candidate_ops:
                    new_prog = prog + [(op_name, op_fn)]
                    eval_cost = self.evaluate_program(new_prog, train_pairs)

                    if eval_cost == 0:
                        best_program = new_prog
                        best_cost = 0
                        break
                    candidates.append((eval_cost, new_prog))

                if best_cost == 0 or (time.perf_counter() - t0) > 2.5:
                    break

            if best_cost == 0:
                break

            candidates.sort(key=lambda x: x[0])
            beam = candidates[:effective_beam_width]

        # Tier 3: Fallback search if not solved and pruning was used
        if best_cost != 0 and use_intelligent_pruning and max_depth >= 2 and (time.perf_counter() - t0) < 2.5:
            candidate_names = {c[0] for c in candidate_ops}
            fallback_ops = [op for op in self.atomic_ops[:40] if op[0] not in candidate_names]
            for cost, prog in beam[:5]:
                if (time.perf_counter() - t0) > 2.5:
                    break
                for op_name, op_fn in fallback_ops:
                    new_prog = prog + [(op_name, op_fn)]
                    eval_cost = self.evaluate_program(new_prog, train_pairs)
                    if eval_cost == 0:
                        best_program = new_prog
                        best_cost = 0
                        break
                if best_cost == 0:
                    break

        dt_ms = (time.perf_counter() - t0) * 1000.0
        is_solved = (best_cost == 0)

        # Tier 4: Object-Centric Multi-Program Synthesis fallback (Phase 4)
        if not is_solved and task.train_pairs:
            obj_sol = self.object_synthesizer.synthesize(task)
            if obj_sol.is_solved:
                return obj_sol

        prog_names = [name for name, _ in best_program] if best_program else []

        # Apply synthesized program to test inputs
        test_predictions = []
        for test_in, _ in task.test_pairs:
            curr = test_in
            if is_solved and best_program:
                for _, fn in best_program:
                    try:
                        curr = fn(curr)
                    except Exception:
                        pass
            test_predictions.append(curr)

        explanation = (
            f"Synthesized {len(prog_names)}-step program: {' -> '.join(prog_names)}"
            if is_solved else
            f"Program synthesis search exhausted at depth {max_depth} (residual mismatch {best_cost})."
        )

        return ARCSolution(
            task_id=task.task_id,
            is_solved=is_solved,
            synthesized_program=prog_names,
            train_accuracy=1.0 if is_solved else 0.0,
            test_predictions=test_predictions,
            latency_ms=dt_ms,
            explanation=explanation,
        )


# Class alias for Phase 3 naming convention
IntelligentSynthesizer = ARCProgramSynthesizer
