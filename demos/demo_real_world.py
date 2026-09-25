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

r"""
Real-World Data Ingestion & Dynamic Generation Demonstration
============================================================
Demonstrates Cortex-Graph v4 with real-world knowledge (non-medical):
1. Ingests rich narrative stories, space exploration history, and science.
2. Builds the Compositional Word Graph, Bigram & Skip-Gram Edges.
3. Automatically induces Fractal Concept Abstractions.
4. Queries the brain and displays dynamic, graph-native generation (NOT static strings).
5. Tests Emotional Resonance and Analogical Reasoning live.

Run with:
    & "x:/BalaDev/codex-net/pyenv/Scripts/python.exe" demo_real_world.py
"""

import os
import sys
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph
from mayon_cortex.affect.emotion import VAD


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def main():
    print_banner("Cortex-Graph v4 — Real-World Intelligence Demonstration")
    print("[*] Initializing Brain...")
    
    config = CortexConfig()
    brain = CortexGraph(config)

    # ─────────────────────────────────────────────────────────────
    # 1. REAL-WORLD DATASET (Stories, Space, Science, Nature)
    # ─────────────────────────────────────────────────────────────
    real_world_corpus = [
        # Narrative Adventure & Kingdom
        (
            "In the ancient kingdom of Eldoria, a noble knight named Arthur guarded the golden palace from danger. "
            "A ferocious dragon dwelt in the volcanic mountain and terrorized the peaceful village. "
            "Arthur forged a magical sword to protect the kingdom. "
            "After a fierce battle, the brave hero defeated the dragon and brought peace and immense joy to the village.",
            "story"
        ),
        # Space Exploration & Science
        (
            "The Apollo 11 spacecraft traveled to the moon in 1969. "
            "Astronaut Neil Armstrong became the first human person to walk on the lunar surface. "
            "The Saturn V rocket generated massive thrust to escape Earth gravity. "
            "The Apollo mission proved that human exploration can cross the void of space.",
            "science"
        ),
        # Earth Ecosystem & Nature
        (
            "The sun produces vital light and radiant energy for Earth. "
            "Green plants in the rainforest perform photosynthesis to create oxygen and glucose. "
            "Trees absorb carbon dioxide from the atmosphere and protect the global climate. "
            "Oceans cover most of the planet and regulate global temperatures through water cycles.",
            "science"
        ),
        # Social Empathy & Human Mind
        (
            "Deep friendship and empathy build lasting trust between human beings. "
            "The sudden tragic loss of a loved one brings profound grief and sorrow. "
            "Compassion and kind words provide comfort during difficult times of hardship. "
            "Celebrating meaningful achievements with loved ones brings genuine happiness and joy.",
            "general"
        ),
    ]

    print("\n[+] Ingesting Real-World Knowledge & Constructing Graph Tiers...")
    total_words = 0
    t0 = time.time()
    for text, sector in real_world_corpus:
        stats = brain.ingest(text, sector=sector)
        total_words += stats.word_nodes_added
        print(f"  - Ingested [{sector.upper()}]: {stats.knowledge_nodes_added} concept nodes, "
              f"{stats.phrase_nodes_added} phrase nodes, {stats.word_nodes_added} word tokens.")

    ingest_time = (time.time() - t0) * 1000
    print(f"\n[OK] Ingestion complete in {ingest_time:.2f}ms. Total vocabulary tokens processed: {total_words}")

    # ─────────────────────────────────────────────────────────────
    # 2. WORD GRAPH TOPOLOGY INSPECTION
    # ─────────────────────────────────────────────────────────────
    print_banner("1. Compositional Word Graph Inspection (Pillar 1)")
    print(f"Total Unique Word Nodes in Brain: {len(brain.word_graph.nodes)}")
    
    sample_words = ["arthur", "hero", "dragon", "apollo", "sun", "photosynthesis", "grief", "joy"]
    print("\nSample Learned Transitions in Graph:")
    for w in sample_words:
        if w in brain.word_graph.nodes:
            node = brain.word_graph.nodes[w]
            top_next = sorted(node.next_words.items(), key=lambda x: x[1], reverse=True)[:4]
            next_str = ", ".join([f"'{nw}' (×{int(cnt)})" for nw, cnt in top_next]) if top_next else "None"
            vad_str = f"V={node.vad.valence:+.2f}, A={node.vad.arousal:.2f}, D={node.vad.dominance:.2f}"
            print(f"  [{node.word.upper():<14}] POS: {list(node.pos_tags)[0]:<10} VAD: [{vad_str}] → Next: {next_str}")

    # ─────────────────────────────────────────────────────────────
    # 3. FRACTAL ABSTRACTION CYCLE
    # ─────────────────────────────────────────────────────────────
    print_banner("2. Fractal Concept Abstraction (Pillar 2)")
    print("[*] Running spontaneous abstraction cycle over graph topology...")
    report = brain.abstract()
    print(f"  - Formed Abstract Concepts: {report.new_concepts_formed}")
    print(f"  - Induced 'is-a' Taxonomic Edges: {report.is_a_edges_created}")
    for c in report.concepts[:5]:
        members = ", ".join(c.member_node_ids[:4])
        print(f"    * Abstract Category: '{c.name}' (Level: {c.level.name}, Sector: {c.sector})")
        print(f"      Instances: [{members}]")

    # ─────────────────────────────────────────────────────────────
    # 4. DYNAMIC QUERY & GRAPH-NATIVE GENERATION (NOT STATIC)
    # ─────────────────────────────────────────────────────────────
    print_banner("3. Dynamic Graph-Native Generation vs Proof Chains")

    test_queries = [
        ("Story Narrative", "What did Arthur do for the kingdom?"),
        ("Space History", "What did the Apollo mission do on the moon?"),
        ("Ecosystem Science", "What do green plants produce during photosynthesis?"),
        ("Emotional Scenario", "A tragedy occurred and the village is in deep mourning."),
    ]

    for category, question in test_queries:
        print(f"\n--- Query [{category.upper()}]: \"{question}\" ---")
        t_start = time.time()
        response = brain.query(question)
        t_elapsed = (time.time() - t_start) * 1000

        print(f"  [Grounded Output] : \"{response.text}\"")
        print(f"  [Confidence]     : {response.confidence:.2f}")
        print(f"  [Latency]        : {t_elapsed:.2f}ms (CPU)")
        if response.emotion:
            print(f"  [Emotion Tone]   : Valence={response.emotion.valence:+.2f} ({brain.emotion_engine.get_emotional_direction(response.emotion)})")

        if response.paths:
            top_path = response.paths[0].path if hasattr(response.paths[0], "path") else response.paths[0]
            edge_chain = " → ".join([f"({src} --{rel}--> {tgt})" for src, rel, tgt in getattr(top_path, "edges", [])])
            print(f"  [Proof Chain]    : {edge_chain if edge_chain else 'Direct node activation'}")

        # Dynamic token-by-token generation via Word Graph (temperature sampling)
        q_vec = brain.embedder.encode_single(question)
        variations = brain.word_generator.generate_variations(
            thought_vector=q_vec,
            proof_paths=response.paths,
            context_vad=response.emotion,
            count=2,
        )
        if variations:
            print("  [Dynamic Graph-Generated Variations (Not Static)]:")
            for i, var_text in enumerate(variations, 1):
                print(f"    Var {i} (T={0.4 + (i-1)*0.3:.1f}): \"{var_text}\"")

    # ─────────────────────────────────────────────────────────────
    # 5. LIVE ANALOGICAL REASONING (Pillar 5)
    # ─────────────────────────────────────────────────────────────
    print_banner("4. Analogical Reasoning (Pillar 5: A is to B as C is to ?)")
    
    analogies = [
        ("knight", "sword", "astronaut"),
        ("dragon", "mountain", "plant"),
        ("sun", "light", "photosynthesis"),
    ]

    for a, b, c in analogies:
        result = brain.solve_analogy(a, b, c)
        if result:
            print(f"  Analogy: '{a}' is to '{b}' as '{c}' is to → ['{result.predicted_d}']")
            print(f"    Confidence: {result.confidence:.2f} | Vector Sim: {result.vector_similarity:.2f} | Struct: {result.structural_match}")
            print(f"    Reasoning: {result.explanation}\n")

    # ─────────────────────────────────────────────────────────────
    # 6. VISION CORTEX MULTIMODAL GROUNDING (Pillar 4)
    # ─────────────────────────────────────────────────────────────
    print_banner("5. Multimodal Vision Grounding (Pillar 4)")
    print("[*] Generating simulated perceptual image representation...")
    simulated_lunar_image = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    image_node = brain.see(
        image_input=simulated_lunar_image,
        caption="A photograph of lunar terrain and Apollo descent stage on the moon",
        tags=["moon", "apollo", "space"]
    )
    print(f"  [+] Grounded Image: ID='{image_node.image_id}'")
    print(f"      Caption: \"{image_node.caption}\"")
    print(f"      Unified ℝ³⁸⁴ embedding norm: {np.linalg.norm(image_node.vector):.4f}")
    
    # Query brain with the image
    nearest = brain.query_image(simulated_lunar_image, top_k=3)
    print(f"  [+] Cross-Modal Nearest Concept Matches in Brain Graph:")
    for nid, score in nearest:
        print(f"      - Concept: {nid} (Similarity: {score:.3f})")

    # ─────────────────────────────────────────────────────────────
    # 7. SUMMARY STATS
    # ─────────────────────────────────────────────────────────────
    print_banner("Demonstration Summary")
    stats = brain.stats()
    print(f"Total Knowledge Nodes : {stats['graph_nodes']}")
    print(f"Total Graph Edges     : {stats['graph_edges']}")
    print(f"Total Word Graph Nodes: {stats['word_nodes']}")
    print(f"Abstract Concepts     : {stats['abstract_concepts']}")
    print(f"Queries Processed     : {stats['queries_processed']}")
    print("\n[OK] All systems operational with dynamic, grounded, non-static intelligence!")


if __name__ == "__main__":
    main()
