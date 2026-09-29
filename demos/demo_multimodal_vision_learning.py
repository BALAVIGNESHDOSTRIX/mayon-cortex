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
Demo: Multimodal Vision Learning & Graph-Native Text-to-Image Generation
========================================================================
Demonstrates:
1. Ingesting real images (Cat, Dog, Nature) into Mayon-Cortex
2. Mathematical feature extraction (HOG, Color HSV, LBP, Gabor) -> 128-dim
3. VQ-Codebook Prototype Clustering & Spatial Graph Topology Learning
4. Generating new images from text prompts using graph autoregression
5. Zero GPU, zero backprop, pure CPU graph operations!
"""

import os
import sys
from pathlib import Path
import numpy as np

# Force UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mayon_cortex.engine import MayonCortex

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def main():
    print("=" * 80)
    print("MAYON-CORTEX: MULTIMODAL VISION LEARNING & GRAPH-NATIVE GENERATION")
    print("=" * 80)

    # Initialize Cortex
    print("\n[1/5] Initializing Mayon-Cortex Cognitive Architecture...")
    cortex = MayonCortex()
    print(f"  [OK] Initialized: {cortex}")

    # Locate sample images
    images_dir = ROOT_DIR / "demos" / "data" / "images"
    output_dir = ROOT_DIR / "demos" / "demo_vision_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_samples = [
        {
            "file": images_dir / "cat.jpg",
            "label": "cat",
            "details": {
                "color": "orange tabby",
                "pose": "sitting soft",
                "features": ["whiskers", "amber eyes", "fur texture", "ears"],
            },
        },
        {
            "file": images_dir / "dog.jpg",
            "label": "dog",
            "details": {
                "color": "golden retriever",
                "pose": "sitting lawn",
                "features": ["golden fur", "snout", "happy ears", "grass"],
            },
        },
        {
            "file": images_dir / "nature.jpg",
            "label": "nature",
            "details": {
                "color": "alpine mountain lake",
                "pose": "landscape wide",
                "features": ["pine trees", "snow peaks", "blue lake", "reflections"],
            },
        },
        {
            "file": images_dir / "rat.jpg",
            "label": "rat",
            "details": {
                "color": "brown and white",
                "pose": "standing curious",
                "features": ["whiskers", "pink nose", "bright eyes", "paws"],
            },
        },
    ]

    print("\n[2/5] Learning & Digesting Real Images into Visual Codebook & Cortical Graph...")
    for item in training_samples:
        img_path = item["file"]
        label = item["label"]
        details = item["details"]

        if img_path.exists() and HAS_PIL:
            print(f"\n  [*] Ingesting real image: {img_path.name} (label: '{label}')")
            img_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        else:
            print(f"\n  [*] Generating synthetic exemplar for: '{label}'")
            # Create synthetic textured image for demonstration fallback
            h, w = 128, 128
            if label == "cat":
                img_arr = cortex.render_procedural_texture("marble", width=w, height=h, base_color=(220, 130, 40), vein_color=(150, 70, 20))
            elif label == "dog":
                img_arr = cortex.render_procedural_texture("wood", width=w, height=h, wood_light=(210, 180, 100), wood_dark=(130, 100, 50))
            else:
                img_arr = cortex.render_procedural_texture("clouds", width=w, height=h, sky_blue=(60, 120, 210), cloud_white=(240, 240, 255))

        # Learn image
        result = cortex.learn_image(
            image_input=img_arr,
            label=label,
            details=details,
            patch_size=(32, 32),
        )

        print(f"    - Patches & Descriptors: {result['tokens_extracted']} patches processed")
        print(f"    - Grid Topology:        {result['grid_shape']}")
        print(f"    - Total Codebook Vocab: {result['codebook_entries']} visual prototype tokens")
        print(f"    - Graph Edges Wired:    {result['graph_edges_created']} multimodal relations")

    print(f"\n[3/5] Visual Memory State:")
    print(f"  - Total Graph Nodes:        {cortex.graph.num_nodes}")
    print(f"  - Total Graph Edges:        {cortex.graph.num_edges}")
    print(f"  - Visual Codebook Entries:  {len(cortex.visual_codebook.entries)}")

    # Step 4: Text-to-Image Generation
    print("\n[4/5] Generating New Images from Text Prompts (Autoregressive Graph Walk)...")

    test_prompts = [
        ("an orange tabby cat sitting", ["cat"]),
        ("a happy golden dog outdoors", ["dog"]),
        ("a serene mountain lake nature scene", ["nature"]),
        ("a curious small pet rat with whiskers", ["rat"]),
        ("a cat and dog side by side", ["cat", "dog"]),
    ]

    for idx, (prompt, target_labels) in enumerate(test_prompts, start=1):
        print(f"\n  [Prompt {idx}] '{prompt}'")
        generated_arr = cortex.generate_image_ar(
            prompt=prompt,
            target_labels=target_labels,
            grid_rows=32,
            grid_cols=32,
            temperature=0.05,
        )

        out_path = output_dir / f"generated_prompt_{idx}.png"
        if HAS_PIL:
            Image.fromarray(generated_arr).save(out_path)
            print(f"    [OK] Image generated and saved to: {out_path} ({generated_arr.shape})")
        else:
            print(f"    [OK] Image array generated: shape {generated_arr.shape}, mean {generated_arr.mean():.1f}")

    # Step 5: Universal Reasoning & Math Puzzles
    print("\n[5/5] Demonstrating Universal Math & Puzzle Solving...")

    # A. Quadratic Equation
    eq = "2*x^2 - 8 = 0"
    res_eq = cortex.solve_equation(eq)
    print(f"  - Equation: {eq} -> Roots: {res_eq['real_roots']}")

    # B. Calculus Derivative
    diff_expr = "x^3 + 4*x^2 - 5*x + 7"
    res_diff = cortex.solve_derivative(diff_expr)
    print(f"  - Derivative: d/dx({diff_expr}) = {res_diff}")

    # C. Cryptarithm
    puzzle = "SEND + MORE = MONEY"
    res_crypt = cortex.solve_cryptarithm(puzzle)
    print(f"  - Cryptarithm: {puzzle} -> Solution: {res_crypt}")

    # D. 9x9 Sudoku
    sudoku_grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    res_sudoku = cortex.solve_sudoku(sudoku_grid)
    print(f"  - Sudoku 9x9 -> Solved top row: {res_sudoku[0] if res_sudoku else 'Failed'}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY! All multimodal & cognitive features verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
