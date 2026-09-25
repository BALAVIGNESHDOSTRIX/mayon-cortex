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
Unified Pipeline Demo — Cortex-Graph v5
========================================
Demonstrates the high-level CortexPipeline:
  1. Instant Graph Ingestion (from_text / from_file)
  2. ask() — Fast factual query with verified proof chains
  3. think() — System 2 multi-step executive reasoning with subgoals & metacognition
  4. generate() — Fluent prose / poetry generation via graph walk & templates
  5. judge() — Precedent-based case judgment
  6. solve() — Problem solving and constraint reasoning
"""

import sys
from pathlib import Path

# Ensure cortex_graph is on path
sys.path.insert(0, str(Path(__file__).parent))

from mayon_cortex.pipeline import CortexPipeline, PipelineMode
from mayon_cortex.perception.scene_graph import SceneGraph, VisualObject


def main():
    print("=" * 70)
    print("  CORTEX-GRAPH v5 — UNIFIED PIPELINE DEMO")
    print("  Third Paradigm: Knowledge Graph + System 2 Thinking + CPU Native")
    print("=" * 70 + "\n")

    # 1. Initialize pipeline with pre-bootstrapped multi-domain knowledge
    print("[1/6] Initializing Unified CortexPipeline...")
    pipeline = CortexPipeline.from_bootstrap()
    print("      [OK] Brain substrate initialized and pre-bootstrapped.")
    print(f"      [OK] Total knowledge nodes: {pipeline.brain.graph.num_nodes}")

    # 2. Ingest dynamic domain corpus
    print("\n[2/6] Ingesting Dynamic Knowledge Corpus...")
    corpus = (
        "Quantum computing relies on quantum bits or qubits. "
        "Superconducting circuits are used to build stable qubits. "
        "Cryogenic refrigeration maintains millikelvin temperatures for qubits. "
        "Quantum error correction prevents decoherence in quantum circuits."
    )
    ingest_stats = pipeline.ingest_text(corpus, sector="physics")
    print(f"      [OK] Ingested sentences: {ingest_stats.get('sentences_ingested', 4)}")
    print(f"      [OK] Nodes added: {ingest_stats.get('knowledge_nodes_added', 0)}")

    # 3. Fast Factual Query (ask mode)
    print("\n[3/6] Running Factual Query: 'What prevents decoherence in quantum circuits?'")
    ask_res = pipeline.ask("What prevents decoherence in quantum circuits?")
    print(f"      Response:   {ask_res.text}")
    print(f"      Confidence: {ask_res.confidence:.2f}")
    if ask_res.paths:
        print(f"      Proof Path: {' -> '.join(ask_res.paths[0].node_ids)}")

    # 4. Deep System 2 Thinking (think mode)
    print("\n[4/6] Running System 2 Thinking: 'How can we scale quantum computing systems?'")
    think_res = pipeline.think("How can we scale quantum computing systems?", mode="analytical")
    print(f"      Conclusion: {think_res.conclusion}")
    print(f"      Subgoals Executed: {len(think_res.subgoals)}")
    for sg in think_res.subgoals[:3]:
        print(f"        - [{sg.status}] {sg.goal}")
    print(f"      Epistemic Humility / Metacognition: {think_res.metacognition.get('epistemic_state', 'confident')}")

    # 5. Precedent-Based Judgment & Math Problem Solving
    print("\n[5/6] Running Case Judgment & Problem Solving...")
    # Teach precedent case
    pipeline.brain.case_memory.store_case(
        situation="Patient with severe hypertension prescribed beta blockers without recent kidney function check.",
        judgment="CONTRAINDICATED: Renal panel and serum creatinine required prior to starting therapy.",
        reasoning="Clinical protocol states beta blocker metabolism is sensitive to renal clearance. Unchecked renal impairment elevates toxicity risk.",
        graph=pipeline.brain.graph,
        embedder=pipeline.brain.embedder,
        sector="medical",
        factors=["hypertension", "beta_blockers", "kidney_function", "renal_panel"],
    )

    # Precedent judgment
    judge_res = pipeline.judge(
        situation="Patient with severe hypertension prescribed beta blockers without recent kidney function check."
    )
    print(f"      Judgment Verdict:    {judge_res.verdict}")
    print(f"      Executive Rationale: {judge_res.rationale}")

    # Problem solving
    solve_res = pipeline.solve("Given distance = 120 km and time = 2 hours, calculate speed.")
    print(f"      Problem Solved:      {solve_res.answer}")


    # 6. Visual Scene Ingestion & Graph Grounding
    print("\n[6/6] Grounding Visual Scene Graph into Knowledge Tissue...")
    scene = SceneGraph(scene_id="lab_setup")
    scene.add_object(VisualObject("qubit_chip", "superconducting_qubit", bbox=(0.4, 0.4, 0.6, 0.6)))
    scene.add_object(VisualObject("cryo_chamber", "cryogenic_fridge", bbox=(0.1, 0.1, 0.9, 0.9)))
    scene.infer_spatial_relations()
    node_ids = pipeline.brain.see_scene(scene)
    print(f"      Scene Description:   {scene.describe_scene()}")
    print(f"      Created Graph Nodes: {node_ids}")

    print("\n" + "=" * 70)
    print("  ALL PIPELINE MODES EXECUTED SUCCESSFULLY [OK]")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    main()
