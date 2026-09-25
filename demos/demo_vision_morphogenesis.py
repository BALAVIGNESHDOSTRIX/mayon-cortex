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
================================================================================
 [DEMO] MAYON-CORTEX MULTIMODAL VISION INGESTION & BIOLOGICAL PATTERN GENESIS
 100% CPU Graph-Native Visual Perception & Morphogenetic Image Generation
================================================================================
"""

import math
import os
import sys
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from mayon_cortex import MayonCortex
from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator, ReactionDiffusionParams
from mayon_cortex.dynamics.energy_canvas import EnergyCanvas, SemanticRegion, SceneDescription


def create_sample_images(input_dir: Path) -> list:
    """Create 10 distinct synthetic input images to ingest into the cortex."""
    input_dir.mkdir(parents=True, exist_ok=True)
    images_meta = [
        ("img_01_cell_cluster.png", "Biological cell cluster in mitosis", ["biology", "cell", "mitosis"], (220, 80, 100)),
        ("img_02_network_topology.png", "Enterprise L3 network core switch topology", ["network", "switch", "topology"], (40, 120, 240)),
        ("img_03_crystal_lattice.png", "Hexagonal silicon crystal lattice structure", ["physics", "crystal", "semiconductor"], (100, 220, 200)),
        ("img_04_coral_ecosystem.png", "Marine coral reef branching colony", ["ecology", "marine", "coral"], (240, 140, 60)),
        ("img_05_fluid_waves.png", "Fluid vortex dynamics and laminar flow waves", ["fluid", "hydrodynamics", "waves"], (30, 160, 220)),
        ("img_06_neural_synapse.png", "Cortical neural synaptic interconnection mesh", ["neuroscience", "neuron", "synapse"], (180, 80, 220)),
        ("img_07_maze_circuit.png", "Integrated circuit micro-routing maze", ["hardware", "circuit", "microchip"], (80, 200, 100)),
        ("img_08_fractal_leaf.png", "Botanical fern leaf fractal geometry", ["botany", "plant", "fractal"], (40, 180, 60)),
        ("img_09_snowflake.png", "Hexagonal ice snowflake crystallographic structure", ["crystallography", "ice", "geometry"], (160, 200, 255)),
        ("img_10_cellular_matrix.png", "Turing-complete cellular automata computational matrix", ["computing", "automata", "logic"], (240, 200, 40)),
    ]

    image_paths = []
    for filename, caption, tags, base_color in images_meta:
        file_path = input_dir / filename
        img = Image.new("RGB", (128, 128), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Draw structured geometric patterns based on tags
        if "cell" in tags or "mitosis" in tags:
            for cx, cy, r in [(40, 40, 25), (85, 75, 30), (50, 90, 20)]:
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=base_color, outline=(150, 40, 60))
        elif "network" in tags or "circuit" in tags:
            for i in range(20, 120, 25):
                draw.line([(20, i), (108, i)], fill=base_color, width=3)
                draw.line([(i, 20), (i, 108)], fill=base_color, width=3)
            for pt in [(20, 20), (70, 70), (108, 108), (20, 108), (108, 20)]:
                draw.ellipse([pt[0]-6, pt[1]-6, pt[0]+6, pt[1]+6], fill=(20, 40, 80))
        elif "crystal" in tags or "ice" in tags or "geometry" in tags:
            points = [(64, 20), (104, 44), (104, 84), (64, 108), (24, 84), (24, 44)]
            draw.polygon(points, fill=base_color, outline=(40, 80, 160))
            draw.line([(64, 20), (64, 108)], fill=(255, 255, 255), width=2)
            draw.line([(24, 44), (104, 84)], fill=(255, 255, 255), width=2)
            draw.line([(24, 84), (104, 44)], fill=(255, 255, 255), width=2)
        elif "waves" in tags or "fluid" in tags:
            for y_off in range(20, 110, 20):
                sine_pts = [(x, int(y_off + 8 * math.sin(x / 10.0))) for x in range(10, 118)]
                draw.line(sine_pts, fill=base_color, width=3)
        elif "coral" in tags or "plant" in tags or "fractal" in tags:
            draw.line([(64, 120), (64, 70)], fill=(80, 50, 20), width=5)
            draw.line([(64, 70), (35, 35)], fill=base_color, width=4)
            draw.line([(64, 70), (95, 35)], fill=base_color, width=4)
            draw.line([(35, 35), (20, 15)], fill=(120, 220, 80), width=3)
            draw.line([(35, 35), (50, 15)], fill=(120, 220, 80), width=3)
            draw.line([(95, 35), (80, 15)], fill=(120, 220, 80), width=3)
            draw.line([(95, 35), (110, 15)], fill=(120, 220, 80), width=3)
        else:
            # Default rich gradient / block pattern
            for y in range(0, 128, 16):
                for x in range(0, 128, 16):
                    if (x + y) % 32 == 0:
                        draw.rectangle([x, y, x+15, y+15], fill=base_color)

        img.save(file_path)
        image_paths.append((file_path, caption, tags))

    return image_paths


def run_vision_and_morphogenesis_demo():
    print("=" * 80)
    print(" [DEMO] MAYON-CORTEX MULTIMODAL VISION INGESTION & BIOLOGICAL MORPHOGENESIS")
    print(" Pure CPU Neuro-Symbolic Intelligence -- Zero GPUs, Zero Backprop")
    print("=" * 80)

    # 1. Initialize Cortex Brain & Morphogenetic Generators
    print("\n1. Initializing Cortex Brain & Visual Perceptual Engine...")
    cortex = MayonCortex()
    morpho = MorphogeneticGenerator()
    canvas = EnergyCanvas(width=128, height=128)

    base_dir = Path("demos/demo_vision_output")
    input_dir = base_dir / "input_images"
    output_dir = base_dir / "generated_morphogenesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate 10 Synthetic Multimodal Images
    print("\n2. Creating 10 Synthetic Domain Images on Disk...")
    image_items = create_sample_images(input_dir)
    for i, (path, caption, tags) in enumerate(image_items, 1):
        print(f"   [{i:02d}/10] Created: {path.name:<30} | Caption: '{caption}'")

    # 3. Ingest All 10 Images into Cortex Knowledge Graph
    print("\n3. Ingesting & Grounding Images into Mayon-Cortex Knowledge Graph...")
    ingested_nodes = []
    for path, caption, tags in image_items:
        node = cortex.see(
            image_input=str(path),
            caption=caption,
            tags=tags,
        )
        ingested_nodes.append((node, caption, tags))
        print(f"   * Ingested ImageNode '{node.image_id}' -> Vector Dim: {node.vector.shape}, Tags: {node.detected_tags}")

    print(f"\n   [Graph State]: {cortex.graph.num_nodes} Concept/Image Nodes, {cortex.graph.num_edges} Multimodal Edges")

    # 4. Perform Visual Cross-Modal Querying
    print("\n4. Performing Cross-Modal Visual Knowledge Retrieval:")
    test_queries = [
        "Find biological cell mitosis structures",
        "Find network switch routing topology",
        "Find crystal lattice geometry"
    ]
    for q in test_queries:
        res = cortex.query(q, sector="general")
        print(f"\n   Q: {q}")
        print(f"   A: {res.text}")

    # 5. Generate Images via Biological Pattern Genesis (Morphogenesis)
    print("\n5. Generating Emergent Images via Biological Pattern Genesis (Turing & L-Systems)...")
    generated_files = []

    # Growth 1: Cellular Spots Morphogenesis (Gray-Scott Reaction-Diffusion)
    print("   [Gen 01] Growing Turing Cellular Spots Reaction-Diffusion (Du=0.16, Dv=0.08, feed=0.035, kill=0.065)...")
    img_spots = morpho.grow_pattern("spots", width=128, height=128, steps=1800, colormap="spots")
    out_1 = output_dir / "morpho_01_cellular_spots.png"
    morpho.save(img_spots, str(out_1))
    generated_files.append((out_1, "Turing Spots Reaction-Diffusion (Leopard/Cellular Mitosis Pattern)"))

    # Growth 2: Network / Neural Wave Stripes Morphogenesis
    print("   [Gen 02] Growing Turing Wave / Stripe Resonances (feed=0.050, kill=0.065)...")
    img_stripes = morpho.grow_pattern("stripes", width=128, height=128, steps=1500, colormap="stripes")
    out_2 = output_dir / "morpho_02_neural_stripes.png"
    morpho.save(img_stripes, str(out_2))
    generated_files.append((out_2, "Turing Stripes Reaction-Diffusion (Neural / Zebra Wave Pattern)"))

    # Growth 3: Marine Coral Branching Morphogenesis
    print("   [Gen 03] Growing Marine Coral Colony Reaction-Diffusion (feed=0.060, kill=0.062)...")
    img_coral = morpho.grow_pattern("coral", width=128, height=128, steps=1500, colormap="coral")
    out_3 = output_dir / "morpho_03_coral_colony.png"
    morpho.save(img_coral, str(out_3))
    generated_files.append((out_3, "Turing Coral Reaction-Diffusion (Branching Marine Colony)"))

    # Growth 4: Micro-Routing Maze Circuit Morphogenesis
    print("   [Gen 04] Growing Integrated Circuit Maze Pattern (feed=0.029, kill=0.057)...")
    img_maze = morpho.grow_pattern("maze", width=128, height=128, steps=1500, colormap="maze")
    out_4 = output_dir / "morpho_04_circuit_maze.png"
    morpho.save(img_maze, str(out_4))
    generated_files.append((out_4, "Turing Maze Reaction-Diffusion (Circuit Routing Labyrinth)"))

    # Growth 5: Spiral Wave Dynamics Morphogenesis
    print("   [Gen 05] Growing Hydrodynamic Spiral Waves (feed=0.014, kill=0.054)...")
    img_waves = morpho.grow_pattern("waves", width=128, height=128, steps=1500, colormap="waves")
    out_5 = output_dir / "morpho_05_spiral_waves.png"
    morpho.save(img_waves, str(out_5))
    generated_files.append((out_5, "Turing Spiral Waves (Fluid Vortex Hydrodynamics)"))

    # Growth 6: Fractal Botanical L-System Tree
    print("   [Gen 06] Growing Fractal L-System Botanical Tree (Axiom F, Rules: FF+[+F-F-F]-[-F+F+F])...")
    img_tree = morpho.grow_tree(width=256, height=256, depth=5, angle=22.0)
    out_6 = output_dir / "morpho_06_fractal_botanical_tree.png"
    morpho.save(img_tree, str(out_6))
    generated_files.append((out_6, "L-System Recursive Fractal Botanical Tree"))

    # Growth 7: Koch Snowflake Crystallographic Geometry
    print("   [Gen 07] Growing Koch Snowflake Crystal (Axiom F--F--F, Rule F -> F+F--F+F)...")
    img_snow = morpho.grow_snowflake(width=256, height=256, depth=4)
    out_7 = output_dir / "morpho_07_koch_snowflake.png"
    morpho.save(img_snow, str(out_7))
    generated_files.append((out_7, "L-System Koch Snowflake Crystallographic Geometry"))

    # Growth 8: Cellular Automaton Rule 110 (Turing-Complete Universal Computation)
    print("   [Gen 08] Growing 1D Cellular Automaton Rule 110 (Universal Computation)...")
    img_ca110 = morpho.grow_cellular(rule=110, width=256, height=128)
    out_8 = output_dir / "morpho_08_cellular_automata_rule110.png"
    morpho.save(img_ca110, str(out_8))
    generated_files.append((out_8, "Rule 110 Cellular Automaton (Turing-Complete Texture)"))

    # Growth 9: Cellular Automaton Rule 90 (Sierpinski Fractal Triangles)
    print("   [Gen 09] Growing 1D Cellular Automaton Rule 90 (Sierpinski Triangles)...")
    img_ca90 = morpho.grow_cellular(rule=90, width=256, height=128)
    out_9 = output_dir / "morpho_09_cellular_automata_rule90.png"
    morpho.save(img_ca90, str(out_9))
    generated_files.append((out_9, "Rule 90 Cellular Automaton (Sierpinski Fractal Geometry)"))

    # Growth 10: Graph-Conditioned EnergyCanvas MRF Scene Synthesis
    print("   [Gen 10] Synthesizing Graph-Conditioned MRF Semantic Scene on CPU...")
    img_scene = canvas.generate_from_graph(
        query="Enterprise network core switch and cellular mitosis biology",
        graph=cortex.graph,
        embedder=cortex.embedder,
        iterations=40
    )
    out_10 = output_dir / "morpho_10_energy_canvas_scene.png"
    canvas.save(img_scene, str(out_10))
    generated_files.append((out_10, "Markov Random Field Energy Minimization Scene Synthesis"))

    # 6. Summary of Generated Output Images
    print("\n" + "=" * 80)
    print(" [GENERATION REPORT] 10 BIOLOGICAL & GRAPH-NATIVE PNG IMAGES GENERATED:")
    print("=" * 80)
    for i, (path, desc) in enumerate(generated_files, 1):
        size_kb = path.stat().st_size / 1024.0
        print(f" [{i:02d}/10] {desc}")
        print(f"        -> File: {path.resolve()} ({size_kb:.1f} KB)")

    print("\n" + "=" * 80)
    print(" [DEMO COMPLETE] Multimodal Vision Ingestion and Morphogenesis Succeeded!")
    print("=" * 80)


if __name__ == "__main__":
    run_vision_and_morphogenesis_demo()
