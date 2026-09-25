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
Mayon-Cortex ARC-AGI 2D Grid Perception & DSL Program Synthesis Engine
========================================================================
Pure CPU, zero-GPU neuro-symbolic engine for solving François Chollet's
Abstraction and Reasoning Corpus (ARC-AGI) tasks.

Key Components:
1. ARCGrid & ARCObject: 2D matrix representations, connected-component object segmentation,
   bounding boxes, topological features, and ASCII visualization.
2. ARCPerception: Object extraction (4-way/8-way), background detection, symmetry analysis,
   hole/enclosure detection, and grid partitioning.
3. ARCDSL: Comprehensive domain-specific language of 2D geometric, topological,
   cellular physics, color mapping, and pattern operators.
4. ARCProgramSynthesizer: Beam / MCTS program synthesis search engine discovering
   exact executable transformation programs matching demonstration pairs.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import math
import time


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
# 1. 2D SPATIAL PERCEPTION & OBJECT SEGMENTATION
# ─────────────────────────────────────────────────────────────

class ARCPerception:
    """
    Cognitive visual perception module: extracts discrete objects, topological holes,
    background colors, and symmetry features from 2D grids.
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
                            # If monochromatic, must match color; otherwise non-background
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


# ─────────────────────────────────────────────────────────────
# 2. ARC DOMAIN-SPECIFIC LANGUAGE (DSL) OPERATORS
# ─────────────────────────────────────────────────────────────

class ARCDSL:
    """
    Comprehensive primitive operations over 2D grids (Geometric, Color, Object, Gravity, Topology).
    """

    # ── Geometric Transforms ──
    @staticmethod
    def rot90(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(np.rot90(grid.matrix, k=-1))  # 90 deg clockwise

    @staticmethod
    def rot180(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(np.rot90(grid.matrix, k=2))

    @staticmethod
    def rot270(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(np.rot90(grid.matrix, k=1))

    @staticmethod
    def flip_h(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(np.fliplr(grid.matrix))

    @staticmethod
    def flip_v(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(np.flipud(grid.matrix))

    @staticmethod
    def transpose(grid: ARCGrid) -> ARCGrid:
        return ARCGrid(grid.matrix.T)

    # ── Color Transforms ──
    @staticmethod
    def recolor(grid: ARCGrid, old_color: int, new_color: int) -> ARCGrid:
        res = grid.matrix.copy()
        res[res == old_color] = new_color
        return ARCGrid(res)

    @staticmethod
    def fill_background(grid: ARCGrid, new_color: int) -> ARCGrid:
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

    # ── Object & Spatial Transforms ──
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
        """Connects pairs of identical colored points along orthogonal axes."""
        res = grid.matrix.copy()
        colors = [c for c in np.unique(grid.matrix) if c != background_color]

        for color in colors:
            coords = np.argwhere(grid.matrix == color)
            if len(coords) >= 2:
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        r1, c1 = coords[i]
                        r2, c2 = coords[j]
                        # Same row -> draw horizontal line
                        if r1 == r2:
                            min_c, max_c = min(c1, c2), max(c1, c2)
                            res[r1, min_c:max_c + 1] = color
                        # Same col -> draw vertical line
                        elif c1 == c2:
                            min_r, max_r = min(r1, r2), max(r1, r2)
                            res[min_r:max_r + 1, c1] = color
        return ARCGrid(res)

    # ── Pattern Tiling & Scaling ──
    @staticmethod
    def tile_pattern(grid: ARCGrid, reps_h: int = 2, reps_w: int = 2) -> ARCGrid:
        """Replicates the grid as a tiled periodic wallpaper pattern."""
        return ARCGrid(np.tile(grid.matrix, (reps_h, reps_w)))

    @staticmethod
    def scale_grid(grid: ARCGrid, factor: int = 2) -> ARCGrid:
        """Scales each pixel into a factor x factor block."""
        return ARCGrid(np.repeat(np.repeat(grid.matrix, factor, axis=0), factor, axis=1))


# ─────────────────────────────────────────────────────────────
# 3. PROGRAM SYNTHESIS ENGINE (MCTS / BEAM SEARCH)
# ─────────────────────────────────────────────────────────────

@dataclass
class ARCTask:
    """An ARC-AGI task containing demonstration pairs and test inputs."""
    task_id: str
    category: str
    train_pairs: List[Tuple[ARCGrid, ARCGrid]]
    test_pairs: List[Tuple[ARCGrid, Optional[ARCGrid]]]


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


class ARCProgramSynthesizer:
    """
    Synthesizes exact Python transformation programs matching demonstration pairs
    via guided beam search over the ARC DSL.
    """

    def __init__(self):
        # Register atomic DSL primitives: (name, function_generator)
        self.atomic_ops: List[Tuple[str, Callable[[ARCGrid], ARCGrid]]] = [
            ("rot90", ARCDSL.rot90),
            ("rot180", ARCDSL.rot180),
            ("rot270", ARCDSL.rot270),
            ("flip_h", ARCDSL.flip_h),
            ("flip_v", ARCDSL.flip_v),
            ("transpose", ARCDSL.transpose),
            ("crop_to_content", ARCDSL.crop_to_content),
            ("extract_largest_object", ARCDSL.extract_largest_object),
            ("extract_smallest_object", ARCDSL.extract_smallest_object),
            ("filter_most_frequent_color", ARCDSL.filter_most_frequent_color),
            ("filter_least_frequent_color", ARCDSL.filter_least_frequent_color),
            ("gravity_down", lambda g: ARCDSL.gravity(g, "down")),
            ("gravity_up", lambda g: ARCDSL.gravity(g, "up")),
            ("gravity_left", lambda g: ARCDSL.gravity(g, "left")),
            ("gravity_right", lambda g: ARCDSL.gravity(g, "right")),
            ("connect_same_colors_with_lines", ARCDSL.connect_same_colors_with_lines),
            ("tile_2x2", lambda g: ARCDSL.tile_pattern(g, 2, 2)),
            ("tile_3x3", lambda g: ARCDSL.tile_pattern(g, 3, 3)),
            ("scale_2x", lambda g: ARCDSL.scale_grid(g, 2)),
            ("scale_3x", lambda g: ARCDSL.scale_grid(g, 3)),
            ("invert_colors", ARCDSL.invert_colors),
        ]

        # Add parameterized hole fills & recolors
        for color in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            c = color
            self.atomic_ops.append((f"fill_enclosed_holes_{c}", lambda g, c=c: ARCDSL.fill_enclosed_holes(g, c)))
            self.atomic_ops.append((f"fill_background_{c}", lambda g, c=c: ARCDSL.fill_background(g, c)))

        # Add color-to-color remappings for colors 0-9
        for c1 in range(10):
            for c2 in range(10):
                if c1 != c2:
                    self.atomic_ops.append((f"recolor_{c1}_to_{c2}", lambda g, c1=c1, c2=c2: ARCDSL.recolor(g, c1, c2)))

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

    def synthesize(self, task: ARCTask, max_depth: int = 3, max_beam_width: int = 15) -> ARCSolution:
        """
        Executes guided Beam Search over the ARC DSL to synthesize an exact program.
        """
        t0 = time.perf_counter()
        train_pairs = task.train_pairs

        # Beam item: (total_cost, program_ops)
        beam: List[Tuple[int, List[Tuple[str, Callable[[ARCGrid], ARCGrid]]]]] = [
            (self.evaluate_program([], train_pairs), [])
        ]

        best_program = None
        best_cost = 999999

        # Search up to max_depth
        for depth in range(1, max_depth + 1):
            candidates = []
            for cost, prog in beam:
                for op_name, op_fn in self.atomic_ops:
                    new_prog = prog + [(op_name, op_fn)]
                    eval_cost = self.evaluate_program(new_prog, train_pairs)
                    
                    if eval_cost == 0:
                        best_program = new_prog
                        best_cost = 0
                        break
                    candidates.append((eval_cost, new_prog))

                if best_cost == 0:
                    break

            if best_cost == 0:
                break

            # Sort and keep top beam_width
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:max_beam_width]

        dt_ms = (time.perf_counter() - t0) * 1000.0

        is_solved = (best_cost == 0)
        prog_names = [name for name, _ in best_program] if best_program else []

        # Apply synthesized program to test inputs
        test_predictions = []
        for test_in, _ in task.test_pairs:
            curr = test_in
            if is_solved and best_program:
                for _, fn in best_program:
                    curr = fn(curr)
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
