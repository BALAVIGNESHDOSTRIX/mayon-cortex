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
Energy Canvas — Graph-Native Image Generation via Energy Minimization
=======================================================================
Phase Ω of the Brain-Like Intelligence upgrade.

A 2D grid of pixel-nodes where image generation = energy minimization.
Each pixel adjusts its color based on LOCAL neighbors + semantic constraints.
NO backpropagation. NO neural network. Pure local energy minimization.

This is how Markov Random Fields (MRFs) generate images — proven for 30+ years.

How it works:
  1. Knowledge graph provides WHAT to draw (scene description)
  2. Create NxN grid of pixel-nodes
  3. Energy function penalizes:
     - Incoherent adjacent pixels (spatial smoothness)
     - Pixels that don't match semantic region (semantic consistency)
     - Random noise (regularization)
  4. Iterated Conditional Modes (ICM): each pixel updates to minimize local energy
  5. Image EMERGES from energy minimization

Output: PIL Image or numpy array
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class SemanticRegion:
    """A semantic region to render on the canvas."""
    label: str
    x: float        # Normalized [0, 1] center x
    y: float        # Normalized [0, 1] center y
    width: float    # Normalized [0, 1]
    height: float   # Normalized [0, 1]
    color: Tuple[int, int, int]  # RGB target color
    texture: str = "solid"       # "solid", "gradient", "pattern"


@dataclass
class MRFConfig:
    """Configuration for Markov Random Field energy minimization."""
    width: int = 64
    height: int = 64
    iterations: int = 100
    lambda_spatial: float = 1.0
    lambda_semantic: float = 2.0
    noise_penalty: float = 0.5


@dataclass
class SceneDescription:
    """A structured scene to render."""
    regions: List[SemanticRegion]
    background_color: Tuple[int, int, int] = (240, 240, 240)
    title: str = ""


class EnergyCanvas:
    """
    Graph-native image generation through energy minimization.
    """

    def __init__(self, width: int = 64, height: int = 64, config: Optional[MRFConfig] = None):
        self.width = width
        self.height = height
        self.config = config or MRFConfig(width=width, height=height)

    def generate(self, prompt: str = "", iterations: int = 50) -> np.ndarray:
        """Generate image from text prompt by constructing semantic regions."""
        # Simple semantic decomposition of prompt
        regions = [
            SemanticRegion(label=prompt or "entity", x=0.5, y=0.5, width=0.4, height=0.4, color=(60, 120, 220)),
        ]
        scene = SceneDescription(regions=regions, title=prompt)
        return self.render(scene, self.width, self.height, iterations=iterations)


    def generate_from_graph(
        self,
        query: str,
        graph: Any,
        embedder: Any,
        width: Optional[int] = None,
        height: Optional[int] = None,
        iterations: int = 50,
    ) -> np.ndarray:
        """
        Generate an image from knowledge graph traversal.
        
        1. Query the graph to find relevant entities + relations
        2. Build a scene description from the graph structure
        3. Render via energy minimization
        """
        w = width or self.width
        h = height or self.height

        # Query graph for scene components
        scene = self._build_scene_from_graph(query, graph, embedder)

        # Render
        return self.render(scene, w, h, iterations)

    def render(
        self,
        scene: SceneDescription,
        width: Optional[int] = None,
        height: Optional[int] = None,
        iterations: int = 50,
    ) -> np.ndarray:
        """
        Render a scene description into an image via energy minimization.
        
        Algorithm: Iterated Conditional Modes (ICM)
          For each iteration:
            For each pixel:
              Compute energy for current color + neighbors
              Find color that minimizes local energy
              Update pixel
        """
        w = width or self.width
        h = height or self.height

        # Initialize canvas with scene regions
        canvas = self._initialize_canvas(scene, w, h)

        # Energy minimization loop
        for iteration in range(iterations):
            # Sweep order: alternating to improve convergence
            if iteration % 2 == 0:
                x_range = range(w)
                y_range = range(h)
            else:
                x_range = range(w - 1, -1, -1)
                y_range = range(h - 1, -1, -1)

            for y in y_range:
                for x in x_range:
                    # Compute optimal color for this pixel
                    optimal = self._minimize_local_energy(
                        canvas, x, y, w, h, scene, iteration, iterations
                    )
                    canvas[y, x] = optimal

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def render_knowledge_map(
        self,
        graph: Any,
        width: int = 128,
        height: int = 128,
    ) -> np.ndarray:
        """
        Render a visual map of the knowledge graph.
        Each node becomes a colored point, edges become lines.
        """
        canvas = np.full((height, width, 3), 250, dtype=np.float32)
        if graph.num_nodes == 0:
            return canvas.astype(np.uint8)

        # Layout nodes using simple force-directed placement
        positions = self._force_directed_layout(graph, width, height)

        # Draw edges first (underneath)
        for eid, edge in graph.edges.items():
            if edge.source_id in positions and edge.target_id in positions:
                x1, y1 = positions[edge.source_id]
                x2, y2 = positions[edge.target_id]
                self._draw_line(canvas, x1, y1, x2, y2, (200, 200, 200))

        # Draw nodes
        sector_colors = {
            "general": (100, 149, 237),   # Cornflower blue
            "medical": (255, 99, 71),     # Tomato
            "legal": (255, 215, 0),       # Gold
            "science": (50, 205, 50),     # Lime green
            "engineering": (138, 43, 226),# Blue violet
            "code": (0, 191, 255),        # Deep sky blue
            "finance": (255, 165, 0),     # Orange
        }

        for nid, pos in positions.items():
            node = graph.nodes.get(nid)
            if not node:
                continue
            sector = getattr(node, "sector", "general")
            color = sector_colors.get(sector, (100, 149, 237))
            x, y = pos
            self._draw_circle(canvas, x, y, 2, color)

        return canvas.astype(np.uint8)

    def to_pil_image(self, array: np.ndarray) -> Any:
        """Convert numpy array to PIL Image."""
        if Image is None:
            raise ImportError("PIL/Pillow not available. Install with: pip install Pillow")
        return Image.fromarray(array.astype(np.uint8))

    def save(self, array: np.ndarray, path: str) -> None:
        """Save generated image to file."""
        img = self.to_pil_image(array)
        img.save(path)

    # ── Internal Methods ──

    def _initialize_canvas(
        self,
        scene: SceneDescription,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Initialize canvas with background and rough region colors."""
        canvas = np.full(
            (height, width, 3), scene.background_color, dtype=np.float32
        )

        for region in scene.regions:
            x_start = int(max(0, (region.x - region.width / 2) * width))
            x_end = int(min(width, (region.x + region.width / 2) * width))
            y_start = int(max(0, (region.y - region.height / 2) * height))
            y_end = int(min(height, (region.y + region.height / 2) * height))

            # Add some noise for organic feel
            noise = np.random.randn(y_end - y_start, x_end - x_start, 3) * 15
            color = np.array(region.color, dtype=np.float32)
            canvas[y_start:y_end, x_start:x_end] = color + noise

        return canvas

    def _minimize_local_energy(
        self,
        canvas: np.ndarray,
        x: int, y: int,
        width: int, height: int,
        scene: SceneDescription,
        iteration: int,
        max_iterations: int,
    ) -> np.ndarray:
        """
        Find the optimal color for pixel (x, y) that minimizes local energy.
        
        Energy components:
        1. Spatial smoothness: pixel should be similar to neighbors
        2. Semantic consistency: pixel should match its semantic region
        3. Data term: maintain original initialization
        """
        current = canvas[y, x].copy()

        # Get neighbor colors
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append(canvas[ny, nx])

        if not neighbors:
            return current

        # Component 1: Spatial smoothness (average of neighbors)
        neighbor_mean = np.mean(neighbors, axis=0)

        # Component 2: Semantic target color
        semantic_color = self._get_semantic_color(x, y, width, height, scene)

        # Annealing: early iterations favor semantic, later favor smoothness
        t = iteration / max(max_iterations, 1)  # 0→1
        semantic_weight = max(0.2, 1.0 - t * 0.7)  # Decreases over time
        smooth_weight = min(0.7, 0.2 + t * 0.5)    # Increases over time

        if semantic_color is not None:
            optimal = (
                smooth_weight * neighbor_mean +
                semantic_weight * semantic_color +
                (1 - smooth_weight - semantic_weight) * current
            )
        else:
            optimal = 0.6 * neighbor_mean + 0.4 * current

        return optimal

    def _get_semantic_color(
        self,
        x: int, y: int,
        width: int, height: int,
        scene: SceneDescription,
    ) -> Optional[np.ndarray]:
        """Get the target color for a pixel based on which semantic region it falls in."""
        fx = x / max(width, 1)
        fy = y / max(height, 1)

        for region in scene.regions:
            # Check if pixel is within region bounds
            if (abs(fx - region.x) < region.width / 2 and
                abs(fy - region.y) < region.height / 2):
                return np.array(region.color, dtype=np.float32)

        return None

    def _build_scene_from_graph(
        self,
        query: str,
        graph: Any,
        embedder: Any,
    ) -> SceneDescription:
        """Build a scene description from graph query results."""
        if graph.num_nodes == 0:
            return SceneDescription(regions=[], title=query)

        query_vec = embedder.encode_single(query)
        norm = np.linalg.norm(query_vec)
        if norm < 1e-8:
            return SceneDescription(regions=[], title=query)

        q_normed = (query_vec / norm).reshape(1, -1).astype(np.float32)
        k = min(8, graph.num_nodes)
        scores, indices = graph.faiss_index.search(q_normed, k)

        id_list = list(graph.nodes.keys())
        regions = []

        # Assign colors from a palette
        palette = [
            (66, 133, 244),   # Google blue
            (234, 67, 53),    # Google red
            (251, 188, 4),    # Google yellow
            (52, 168, 83),    # Google green
            (138, 43, 226),   # Purple
            (255, 127, 80),   # Coral
            (0, 206, 209),    # Turquoise
            (255, 20, 147),   # Pink
        ]

        # Arrange found concepts as regions in a grid
        cols = math.ceil(math.sqrt(k))
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(id_list):
                continue

            col = i % cols
            row = i // cols
            regions.append(SemanticRegion(
                label=graph.nodes[id_list[idx]].source_text[:30],
                x=(col + 0.5) / cols,
                y=(row + 0.5) / math.ceil(k / cols),
                width=0.8 / cols,
                height=0.8 / math.ceil(k / cols),
                color=palette[i % len(palette)],
            ))

        return SceneDescription(regions=regions, title=query)

    def _force_directed_layout(
        self,
        graph: Any,
        width: int,
        height: int,
        iterations: int = 50,
    ) -> Dict[str, Tuple[int, int]]:
        """Simple force-directed graph layout."""
        positions: Dict[str, np.ndarray] = {}
        margin = 10

        # Random initial positions
        for nid in graph.nodes:
            positions[nid] = np.array([
                np.random.uniform(margin, width - margin),
                np.random.uniform(margin, height - margin),
            ])

        # Iterate
        node_list = list(graph.nodes.keys())
        n = len(node_list)
        if n == 0:
            return {}

        k_rep = (width * height) / max(n, 1)  # Repulsion constant
        k_att = math.sqrt(width * height / max(n, 1))  # Attraction constant

        for _ in range(iterations):
            # Repulsion between all pairs (limit for performance)
            sample = node_list[:min(200, n)]
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    a, b = sample[i], sample[j]
                    delta = positions[a] - positions[b]
                    dist = max(np.linalg.norm(delta), 1.0)
                    force = k_rep / (dist * dist) * (delta / dist)
                    positions[a] += force * 0.5
                    positions[b] -= force * 0.5

            # Attraction along edges
            for eid, edge in list(graph.edges.items())[:500]:
                if edge.source_id in positions and edge.target_id in positions:
                    delta = positions[edge.target_id] - positions[edge.source_id]
                    dist = max(np.linalg.norm(delta), 1.0)
                    force = dist / k_att * (delta / dist)
                    positions[edge.source_id] += force * 0.3
                    positions[edge.target_id] -= force * 0.3

            # Clamp to canvas
            for nid in positions:
                positions[nid] = np.clip(
                    positions[nid], margin, [width - margin, height - margin]
                )

        return {nid: (int(pos[0]), int(pos[1])) for nid, pos in positions.items()}

    def _draw_line(
        self,
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a simple line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        h, w = canvas.shape[:2]
        steps = 0
        max_steps = dx + dy + 1

        while steps < max_steps:
            if 0 <= x1 < w and 0 <= y1 < h:
                canvas[y1, x1] = color
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
            steps += 1

    def _draw_circle(
        self,
        canvas: np.ndarray,
        cx: int, cy: int,
        radius: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a filled circle."""
        h, w = canvas.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        canvas[y, x] = color
