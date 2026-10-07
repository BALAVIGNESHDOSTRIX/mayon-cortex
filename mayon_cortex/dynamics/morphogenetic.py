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
Morphogenetic Generator — Images That GROW From Rules
=======================================================
Phase Ω of the Brain-Like Intelligence upgrade.

Inspired by biological morphogenesis (Turing, 1952):
  DNA doesn't store pixel values → it stores GROWTH RULES
  Rules + local interaction → complex organisms EMERGE

Techniques:
  1. Reaction-Diffusion: Two chemicals interact → spots, stripes, waves
  2. L-Systems: Recursive rewriting → trees, plants, fractals
  3. Cellular Automata: Simple local rules → complex textures

ALL run on CPU. NO neural networks. NO backpropagation.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class ReactionDiffusionParams:
    """Parameters for Turing reaction-diffusion pattern generation."""
    Du: float = 0.16       # Diffusion rate of activator
    Dv: float = 0.08       # Diffusion rate of inhibitor
    feed: float = 0.035    # Feed rate
    kill: float = 0.065    # Kill rate
    dt: float = 1.0        # Time step

    @classmethod
    def spots(cls) -> "ReactionDiffusionParams":
        """Parameters for spot patterns (leopard-like)."""
        return cls(Du=0.16, Dv=0.08, feed=0.035, kill=0.065, dt=1.0)

    @classmethod
    def stripes(cls) -> "ReactionDiffusionParams":
        """Parameters for stripe patterns (zebra-like)."""
        return cls(Du=0.16, Dv=0.08, feed=0.050, kill=0.065, dt=1.0)

    @classmethod
    def waves(cls) -> "ReactionDiffusionParams":
        """Parameters for wave/spiral patterns."""
        return cls(Du=0.16, Dv=0.08, feed=0.014, kill=0.054, dt=1.0)

    @classmethod
    def coral(cls) -> "ReactionDiffusionParams":
        """Parameters for coral-like branching patterns."""
        return cls(Du=0.16, Dv=0.08, feed=0.060, kill=0.062, dt=1.0)

    @classmethod
    def maze(cls) -> "ReactionDiffusionParams":
        """Parameters for maze-like patterns."""
        return cls(Du=0.16, Dv=0.08, feed=0.029, kill=0.057, dt=1.0)


# Aliases
MorphogeneticParams = ReactionDiffusionParams


class MorphogeneticGenerator:

    """
    Generate images through biological growth rules.
    
    No training data. No neural network. No backpropagation.
    Just LOCAL RULES → EMERGENT COMPLEX PATTERNS.
    
    Like nature itself: from simple DNA instructions, complex organisms grow.
    """

    def grow_pattern(
        self,
        pattern_type: str,
        width: int = 128,
        height: int = 128,
        steps: int = 3000,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Grow a visual pattern from rules.
        
        Args:
            pattern_type: "spots", "stripes", "waves", "coral", "maze"
            width: Image width
            height: Image height
            steps: Number of growth steps (more = more defined)
            colormap: Optional color scheme
            
        Returns:
            RGB numpy array (height, width, 3)
        """
        param_map = {
            "spots": ReactionDiffusionParams.spots,
            "stripes": ReactionDiffusionParams.stripes,
            "waves": ReactionDiffusionParams.waves,
            "coral": ReactionDiffusionParams.coral,
            "maze": ReactionDiffusionParams.maze,
        }

        factory = param_map.get(pattern_type.lower(), ReactionDiffusionParams.spots)
        params = factory()

        # Run reaction-diffusion
        pattern = self._reaction_diffusion(width, height, params, steps)

        # Colorize
        return self._colorize(pattern, colormap or pattern_type)

    def grow_tree(
        self,
        width: int = 256,
        height: int = 256,
        depth: int = 7,
        angle: float = 25.0,
        length_ratio: float = 0.7,
    ) -> np.ndarray:
        """
        Grow a fractal tree using L-system rules.
        
        L-system rule:
          F → F[+F][-F]
          F = draw forward
          + = turn right by angle
          - = turn left by angle
          [ = push state
          ] = pop state
        """
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        # Generate L-system string
        axiom = "F"
        rules = {"F": "FF+[+F-F-F]-[-F+F+F]"}
        sentence = axiom
        for _ in range(depth):
            new = []
            for ch in sentence:
                new.append(rules.get(ch, ch))
            sentence = "".join(new)

        # Turtle interpretation
        x = width / 2.0
        y = float(height - 10)
        theta = -90.0  # Start pointing up
        length = height / (3.0 * (depth + 1))
        stack: List[Tuple[float, float, float, float]] = []

        for ch in sentence:
            if ch == 'F':
                # Draw line
                new_x = x + length * math.cos(math.radians(theta))
                new_y = y + length * math.sin(math.radians(theta))
                self._draw_line_aa(canvas, x, y, new_x, new_y, (34, 100, 34))
                x, y = new_x, new_y
            elif ch == '+':
                theta += angle
            elif ch == '-':
                theta -= angle
            elif ch == '[':
                stack.append((x, y, theta, length))
                length *= length_ratio
            elif ch == ']':
                if stack:
                    x, y, theta, length = stack.pop()

        return canvas

    def grow_snowflake(
        self,
        width: int = 256,
        height: int = 256,
        depth: int = 4,
    ) -> np.ndarray:
        """Grow a Koch snowflake fractal."""
        canvas = np.full((height, width, 3), 240, dtype=np.uint8)

        # Koch curve L-system: F → F+F--F+F
        axiom = "F--F--F"
        rules = {"F": "F+F--F+F"}
        sentence = axiom
        for _ in range(depth):
            new = []
            for ch in sentence:
                new.append(rules.get(ch, ch))
            sentence = "".join(new)

        # Calculate total steps to normalize length
        f_count = sentence.count('F')
        length = min(width, height) * 0.7 / max(math.sqrt(f_count), 1)

        # Turtle
        x = width * 0.15
        y = height * 0.7
        theta = 0.0

        for ch in sentence:
            if ch == 'F':
                new_x = x + length * math.cos(math.radians(theta))
                new_y = y + length * math.sin(math.radians(theta))
                self._draw_line_aa(canvas, x, y, new_x, new_y, (0, 100, 200))
                x, y = new_x, new_y
            elif ch == '+':
                theta += 60
            elif ch == '-':
                theta -= 60

        return canvas

    def grow_cellular(
        self,
        rule: int = 110,
        width: int = 256,
        height: int = 256,
    ) -> np.ndarray:
        """
        Grow a 1D cellular automaton pattern.
        
        Rule 110 is Turing-complete — it can compute anything!
        Rule 30 produces cryptographic-quality randomness.
        Rule 90 produces Sierpinski triangles.
        """
        grid = np.zeros((height, width), dtype=np.uint8)

        # Initialize first row with single center cell
        grid[0, width // 2] = 1

        # Apply rule
        for y in range(1, height):
            for x in range(width):
                left = grid[y - 1, (x - 1) % width]
                center = grid[y - 1, x]
                right = grid[y - 1, (x + 1) % width]

                # Three-bit neighborhood → 8 possible states
                neighborhood = (left << 2) | (center << 1) | right
                grid[y, x] = (rule >> neighborhood) & 1

        # Colorize
        canvas = np.full((height, width, 3), 240, dtype=np.uint8)
        canvas[grid == 1] = (30, 30, 60)

        return canvas

    # ── Internal Methods ──

    def _reaction_diffusion(
        self,
        width: int,
        height: int,
        params: ReactionDiffusionParams,
        steps: int,
    ) -> np.ndarray:
        """
        Turing's morphogenesis: two chemicals (activator u + inhibitor v)
        interact on a 2D grid → emerge into spots, stripes, or waves.
        
        The Gray-Scott model:
          ∂u/∂t = Du·∇²u - u·v² + f·(1-u)
          ∂v/∂t = Dv·∇²v + u·v² - (f+k)·v
        
        Pure math. Pure CPU. Zero neural networks.
        """
        # Initialize
        u = np.ones((height, width), dtype=np.float64)
        v = np.zeros((height, width), dtype=np.float64)

        # Define radius for seeding bounds
        r = min(width, height) // 4

        # Seed with multiple organic distributed nucleation points
        rng = np.random.RandomState(42)
        num_seeds = max(6, (width * height) // 1024)
        for _ in range(num_seeds):
            sx = rng.randint(r // 2, width - r // 2)
            sy = rng.randint(r // 2, height - r // 2)
            sr = rng.randint(2, max(3, r // 2))
            y_indices, x_indices = np.ogrid[:height, :width]
            mask = (x_indices - sx)**2 + (y_indices - sy)**2 <= sr**2
            u[mask] = 0.50
            v[mask] = 0.25

        # Also add central perturbation
        cx, cy = width // 2, height // 2
        u[cy - r:cy + r, cx - r:cx + r] = 0.50
        v[cy - r:cy + r, cx - r:cx + r] = 0.25

        # Add random background noise to break symmetry
        u += rng.randn(height, width) * 0.02
        v += rng.randn(height, width) * 0.02

        # Simulation loop
        for _ in range(steps):
            # Laplacian via convolution (5-point stencil)
            Lu = (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
            )
            Lv = (
                np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
            )

            # Gray-Scott equations
            uvv = u * v * v
            du = params.Du * Lu - uvv + params.feed * (1.0 - u)
            dv = params.Dv * Lv + uvv - (params.feed + params.kill) * v

            u += du * params.dt
            v += dv * params.dt

            # Clamp
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)

        return v  # Return inhibitor field (more visually interesting)

    def _colorize(self, pattern: np.ndarray, scheme: str) -> np.ndarray:
        """Apply a color scheme to a grayscale pattern."""
        h, w = pattern.shape

        # Normalize to [0, 1]
        p_min = pattern.min()
        p_max = pattern.max()
        if p_max - p_min > 1e-8:
            norm = (pattern - p_min) / (p_max - p_min)
        else:
            norm = np.zeros_like(pattern)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        color_schemes = {
            "spots": lambda t: (
                int(255 * (1 - t)),
                int(200 * (1 - t * 0.5)),
                int(100 + 155 * t),
            ),
            "stripes": lambda t: (
                int(50 + 200 * t),
                int(50 + 150 * t),
                int(50 + 100 * t),
            ),
            "waves": lambda t: (
                int(20 + 100 * t),
                int(50 + 180 * t),
                int(150 + 105 * t),
            ),
            "coral": lambda t: (
                int(200 + 55 * t),
                int(100 + 100 * (1 - t)),
                int(100 * (1 - t)),
            ),
            "maze": lambda t: (
                int(30 + 60 * t),
                int(30 + 60 * t),
                int(60 + 120 * t),
            ),
        }

        color_fn = color_schemes.get(scheme, color_schemes["spots"])

        for y in range(h):
            for x in range(w):
                r, g, b = color_fn(norm[y, x])
                canvas[y, x] = (
                    min(255, max(0, r)),
                    min(255, max(0, g)),
                    min(255, max(0, b)),
                )

        return canvas

    def _draw_line_aa(
        self,
        canvas: np.ndarray,
        x1: float, y1: float,
        x2: float, y2: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw an anti-aliased line using Bresenham."""
        h, w = canvas.shape[:2]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(int(max(dx, dy)), 1)

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            if 0 <= x < w and 0 <= y < h:
                canvas[y, x] = color

    def to_pil_image(self, array: np.ndarray) -> Any:
        """Convert numpy array to PIL Image."""
        if Image is None:
            raise ImportError("Pillow not available")
        return Image.fromarray(array.astype(np.uint8))

    def save(self, array: np.ndarray, path: str) -> None:
        """Save to file."""
        img = self.to_pil_image(array)
        img.save(path)
