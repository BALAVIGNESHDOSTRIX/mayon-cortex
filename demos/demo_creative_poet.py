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
Demo: 100% Graph-Native Creative Poetry & Zero-Hallucination Factual Queries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mayon_cortex import CortexGraph
from mayon_cortex.learning.bootstrap import bootstrap_minimal

def main():
    print("=" * 70)
    print(" CORTEX-GRAPH: 100% GRAPH-NATIVE INTELLIGENCE (ZERO TRANSFORMERS)")
    print("=" * 70)

    brain = CortexGraph()
    print("\n[1] Bootstrapping Brain with Language & Rhyme Graph...")
    bootstrap_minimal(brain, verbose=False)

    # Ingest verified facts for factual zero-hallucination demonstration
    brain.ingest("Lisinopril is an ACE inhibitor that treats hypertension.", sector="medical")
    brain.ingest("Metformin is an oral medication that treats type 2 diabetes.", sector="medical")
    brain.ingest("Aspirin is an antiplatelet medication that prevents blood clots.", sector="medical")

    print(f"    Graph ready: {brain.graph.num_nodes} nodes, {brain.graph.num_edges} edges, {len(brain.word_graph.nodes)} vocabulary nodes.")

    print("\n" + "=" * 70)
    print(" DEMO 1: METRIC & RHYMED POETRY — HEROIC COUPLETS (AABB)")
    print(" Theme: 'Cosmic Stars and Eternity'")
    print("=" * 70)
    poem1 = brain.compose_poem(theme="stars and eternity", form="couplet", mood="epic", lines=4, meter="tetrameter")
    print(poem1)

    print("\n" + "=" * 70)
    print(" DEMO 2: CONCEPTUAL BLENDING POETRY — QUATRAIN (ABAB)")
    print(" Blending Distant Concepts: ['Time', 'Ocean Waves']")
    print("=" * 70)
    poem2 = brain.compose_poem(theme=["time", "ocean waves"], form="quatrain", mood="serene", lines=4, meter="tetrameter")
    print(poem2)

    print("\n" + "=" * 70)
    print(" DEMO 3: CREATIVE PROSE GENERATION")
    print(" Style: Epic fantasy narrative")
    print("=" * 70)
    prose = brain.compose_prose("The ancient hero faced the storm with unyielding courage", style="epic", max_words=30)
    print(prose)

    print("\n" + "=" * 70)
    print(" DEMO 4: ZERO-HALLUCINATION FACTUAL REASONING (J.A.R.V.I.S. Core)")
    print(" Query: 'What treats hypertension?'")
    print("=" * 70)
    response = brain.query("What treats hypertension?")
    print(f"Answer: {response.text}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Uncertain: {response.is_uncertain}")
    print(f"Proof paths: {len(response.paths)} verified graph paths")

    print("\n" + "=" * 70)
    print(" DEMO 5: EPISTEMIC HUMILITY ON UNKNOWN FACT (Refuses to Guess)")
    print(" Query: 'What is the secret recipe of kryptonite pizza?'")
    print("=" * 70)
    resp_unknown = brain.query("What is the secret recipe of kryptonite pizza?")
    print(f"Answer: {resp_unknown.text}")
    print(f"Confidence: {resp_unknown.confidence:.2f}")
    print(f"Uncertain: {resp_unknown.is_uncertain}")

    print("\n" + "=" * 70)
    print(" SUCCESS: Complete Graph-Native Creative + Zero-Hallucination Pipeline")
    print("=" * 70)

if __name__ == "__main__":
    main()
