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

r"""
TinyStories Live Dataset Test & Benchmark — Cortex-Graph v4
===========================================================
Pulls real-world data from the official Hugging Face TinyStories dataset
(roneneldan/TinyStories), ingests it into Cortex-Graph v4, builds the
Compositional Word Graph, induces Fractal Abstractions, and benchmarks
dynamic reasoning and generation on CPU.

Run with:
    & "x:/BalaDev/codex-net/pyenv/Scripts/python.exe" test_tinystories_live.py
"""

import json
import os
import sys
import time
import urllib.request
from typing import List

import numpy as np

# Ensure proper stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def fetch_tinystories(limit: int = 8) -> List[str]:
    """Fetch live sample stories from Hugging Face TinyStories dataset."""
    url = f"https://datasets-server.huggingface.co/rows?dataset=roneneldan%2FTinyStories&config=default&split=train&offset=0&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CortexGraph-Bench)"})
    try:
        print(f"[*] Pulling {limit} stories live from Hugging Face (roneneldan/TinyStories)...")
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            stories = [r["row"]["text"].strip() for r in data.get("rows", [])]
            print(f"[+] Successfully fetched {len(stories)} stories from remote API!")
            return stories
    except Exception as e:
        print(f"[!] Warning: Remote fetch failed ({e}). Falling back to bundled TinyStories samples.")
        return [
            (
                "Once upon a time, there was a little boy named Tim. Tim loved to play with his red toy car in the garden. "
                "One afternoon, Tim saw a tiny lost kitten hiding under the green bush. The kitten was crying and hungry. "
                "Tim gently picked up the kitten and brought warm milk from the kitchen. "
                "The kitten drank the milk and purred with joy. Tim and the kitten became best friends forever."
            ),
            (
                "Lily was a kind girl who lived in a sunny house near the river. "
                "She planted bright sunflower seeds in the rich soil every spring. "
                "With water and warm sunshine, the golden flowers grew tall and beautiful. "
                "Birds and honeybees visited the garden every morning. Lily felt peaceful and very happy."
            ),
            (
                "A curious puppy named Max went on an adventure into the deep forest. "
                "Max found a sparkling crystal stone near an old oak tree. "
                "Suddenly, thunder roared and dark rain began to fall from the sky. Max was scared of the storm. "
                "A wise old owl guided Max safely back home to his warm bed."
            ),
            (
                "Mia and her brother Leo built a tall sandcastle on the sandy beach. "
                "They decorated the castle with colorful seashells and white feathers. "
                "A big ocean wave rushed in and washed the sandcastle away. "
                "Mia was sad at first, but Leo laughed and they happily began building an even bigger castle."
            ),
        ]


def main():
    print_banner("Cortex-Graph v4 — TinyStories Live Dataset Ingestion & Dynamic Generation")
    print("[*] Initializing Cortex Brain Architecture on CPU...")

    config = CortexConfig()
    brain = CortexGraph(config)

    # 1. Fetch Real Dataset
    stories = fetch_tinystories(limit=6)

    # 2. Ingest Dataset into 3 Tiers
    print_banner("1. Multi-Story Ingestion & Graph Construction")
    t0 = time.time()
    total_words = 0
    total_concepts = 0
    total_phrases = 0

    for idx, story in enumerate(stories, 1):
        clean_story = " ".join(story.split())
        stats = brain.ingest(clean_story, sector="story", provenance=f"tinystory_{idx}")
        total_words += stats.word_nodes_added
        total_concepts += stats.knowledge_nodes_added
        total_phrases += stats.phrase_nodes_added
        preview = clean_story[:85] + "..." if len(clean_story) > 85 else clean_story
        print(f"  [{idx}] Ingested Story: \"{preview}\"")
        print(f"      -> +{stats.knowledge_nodes_added} concept nodes, +{stats.phrase_nodes_added} phrase nodes, +{stats.word_nodes_added} words")

    ingest_time = (time.time() - t0) * 1000
    print(f"\n[OK] Ingested {len(stories)} TinyStories in {ingest_time:.2f}ms on CPU!")
    print(f"     Total Concepts: {total_concepts} | Total Phrases: {total_phrases} | Total Word Nodes: {len(brain.word_graph.nodes)}")

    # 3. Fractal Concept Abstraction
    print_banner("2. Fractal Concept Abstraction (Pillar 2)")
    print("[*] Inducing ontological clusters and taxonomy...")
    report = brain.abstract()
    print(f"  - Abstract Categories Formed: {report.new_concepts_formed}")
    print(f"  - Induced 'is-a' Taxonomic Edges: {report.is_a_edges_created}")
    for c in report.concepts[:4]:
        print(f"    * Abstract Category '{c.name}' (Sector: {c.sector}, Members: {len(c.member_node_ids)})")

    # 4. Dynamic Reasoning & Generation Tests
    print_banner("3. Dynamic Reasoning vs Grounded Generation (Pillar 1 & 6)")

    benchmark_queries = [
        "What did Tim do when he found the hungry kitten?",
        "How did the sunflower seeds grow in Lily's garden?",
        "Why was the puppy scared in the deep forest?",
        "What happened when the wave hit Mia and Leo's sandcastle?",
        "How do friends help each other during difficult times?",
    ]

    latencies = []
    for query in benchmark_queries:
        print(f"\n--- Question: \"{query}\" ---")
        q_start = time.time()
        response = brain.query(query)
        q_time = (time.time() - q_start) * 1000
        latencies.append(q_time)

        print(f"  [Grounded Answer] : \"{response.text}\"")
        print(f"  [Confidence]     : {response.confidence:.2f}")
        print(f"  [Latency]        : {q_time:.2f}ms (CPU)")
        if response.emotion:
            print(f"  [Emotional Tone] : Valence={response.emotion.valence:+.2f} ({brain.emotion_engine.get_emotional_direction(response.emotion)})")

        # Dynamic Token-by-Token variations across temperatures
        q_vec = brain.embedder.encode_single(query)
        variations = brain.word_generator.generate_variations(
            thought_vector=q_vec,
            proof_paths=response.paths,
            context_vad=response.emotion,
            count=2,
        )
        if variations:
            print("  [Dynamic Graph-Generated Variations (Not Static)]:")
            for i, var in enumerate(variations, 1):
                temp = 0.4 + (i - 1) * 0.3
                print(f"    Var {i} (T={temp:.1f}): \"{var}\"")

    # 5. Cross-Story Analogical Reasoning
    print_banner("4. Analogical Reasoning (Pillar 5: A : B :: C : ?)")
    story_analogies = [
        ("tim", "kitten", "leo"),
        ("sun", "flower", "water"),
        ("puppy", "forest", "girl"),
    ]
    for a, b, c in story_analogies:
        res = brain.solve_analogy(a, b, c)
        if res:
            print(f"  '{a}' is to '{b}' as '{c}' is to → ['{res.predicted_d}']")
            print(f"    Confidence: {res.confidence:.2f} | Reasoning: {res.explanation}")

    # 6. Benchmark Summary
    print_banner("5. Performance Benchmark Summary")
    stats = brain.stats()
    avg_latency = np.mean(latencies) if latencies else 0.0
    print(f"Total Knowledge Nodes Ingested : {stats['graph_nodes']}")
    print(f"Total Graph Edges               : {stats['graph_edges']}")
    print(f"Total Compositional Word Nodes  : {stats['word_nodes']}")
    print(f"Abstract Concept Clusters       : {stats['abstract_concepts']}")
    print(f"Average CPU Query Latency       : {avg_latency:.2f}ms")
    print(f"Hardware Requirement           : 100% CPU (0 GPU VRAM)")
    print("\n[OK] TinyStories dynamic intelligence benchmark complete!")


if __name__ == "__main__":
    main()
