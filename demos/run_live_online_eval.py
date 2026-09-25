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
Live Online Data Ingestion & Evaluation — Cortex-Graph v4
=========================================================
1. Directly downloads live stories from Hugging Face (roneneldan/TinyStories).
2. Ingests the real downloaded text into Cortex-Graph (Knowledge + Word Graph + Emotion).
3. Executes Fractal Abstraction over the multi-story graph.
4. Evaluates dynamic reasoning, grounded proof retrieval, and temperature variations.
5. Provides interactive testing support.

Run with:
    & "x:/BalaDev/codex-net/pyenv/Scripts/python.exe" run_live_online_eval.py
"""

import json
import os
import sys
import time
import urllib.request
from typing import List, Dict, Any

import numpy as np

# Ensure clean UTF-8 console output on Windows
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


def download_live_tinystories(limit: int = 5) -> List[Dict[str, Any]]:
    """Download live stories directly from Hugging Face dataset API."""
    url = f"https://datasets-server.huggingface.co/rows?dataset=roneneldan%2FTinyStories&config=default&split=train&offset=0&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    print(f"[*] Downloading {limit} live stories from Hugging Face (roneneldan/TinyStories)...", flush=True)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("rows", [])[:limit]
        stories = []
        for r in rows:
            text = r["row"]["text"].strip()
            stories.append({"id": r["row_idx"], "text": text})
        print(f"[+] Successfully downloaded {len(stories)} full real stories from Hugging Face!\n", flush=True)
        return stories


def main():
    print_banner("Cortex-Graph v4 — Live Internet Data Ingestion & Evaluation")
    
    print("[*] Initializing Cortex-Graph Cognitive Architecture on CPU...")
    config = CortexConfig()
    brain = CortexGraph(config)

    # 1. Download live real-world dataset
    stories = download_live_tinystories(limit=5)

    # 2. Ingest downloaded stories into the graph
    print_banner("1. Ingesting Live Downloaded Stories into Graph Tiers")
    t0 = time.time()
    total_words = 0
    
    for idx, item in enumerate(stories, 1):
        text = item["text"]
        stats = brain.ingest(text, sector="story", provenance=f"hf_tinystory_{item['id']}")
        total_words += stats.word_nodes_added
        
        # Show first 120 chars of downloaded story
        preview = " ".join(text.split())[:110] + "..."
        print(f"  [Story {idx}] (ID: {item['id']}): \"{preview}\"")
        print(f"    └── +{stats.knowledge_nodes_added} concept nodes, +{stats.phrase_nodes_added} phrase bridges, +{stats.word_nodes_added} word tokens")

    ingest_time = (time.time() - t0) * 1000
    print(f"\n[OK] Ingested all {len(stories)} live stories in {ingest_time:.2f}ms on CPU.")
    print(f"     Knowledge Nodes: {len(brain.graph.nodes)} | Word Nodes: {len(brain.word_graph.nodes)}")

    # 3. Fractal Concept Abstraction
    print_banner("2. Fractal Concept Abstraction Over Live Data")
    report = brain.abstract()
    print(f"  - Abstract Categories Formed: {report.new_concepts_formed}")
    print(f"  - Induced 'is-a' Taxonomic Edges: {report.is_a_edges_created}")
    for c in report.concepts[:4]:
        print(f"    * Category '{c.name}' (Sector: {c.sector}, Members: {len(c.member_node_ids)})")

    # 4. Evaluation Questions based on the downloaded stories
    print_banner("3. Live Dynamic Evaluation & Reasoning Benchmark")

    eval_questions = [
        # Based on Story 1 (Lily and the needle / balloon / room)
        "What did the little girl named Lily find in her room?",
        "Why was it difficult for Lily to play with the needle?",
        # Based on Story 2 (Jane and the big tree with green powder)
        "What did Jane see around the big green tree in the park?",
        # Based on Story 3 (Tim / dog / garden adventure)
        "What happens when animals or children play together in the garden?",
        # Emotional / Moral reasoning
        "How do characters feel when they solve a difficult situation?",
    ]

    print("[*] Running automated evaluation across downloaded content:")
    
    latencies = []
    for q_idx, question in enumerate(eval_questions, 1):
        print(f"\n{'─'*70}")
        print(f"Query {q_idx}: \"{question}\"")
        print(f"{'─'*70}")
        
        t_start = time.time()
        response = brain.query(question)
        latency = (time.time() - t_start) * 1000
        latencies.append(latency)

        print(f"  [Grounded Answer] : \"{response.text}\"")
        print(f"  [Confidence]     : {response.confidence:.2f}")
        print(f"  [CPU Latency]    : {latency:.2f}ms")
        if response.emotion:
            emo_dir = brain.emotion_engine.get_emotional_direction(response.emotion)
            print(f"  [Emotion State]  : Valence={response.emotion.valence:+.2f} ({emo_dir})")

        # Dynamic token-by-token generation across temperatures
        q_vec = brain.embedder.encode_single(question)
        variations = brain.word_generator.generate_variations(
            thought_vector=q_vec,
            proof_paths=response.paths,
            context_vad=response.emotion,
            count=2,
        )
        if variations:
            print("  [Dynamic Graph Predictions (Not Static)]:")
            for i, var in enumerate(variations, 1):
                temp = 0.4 + (i - 1) * 0.3
                print(f"    • Variation {i} (T={temp:.1f}): \"{var}\"")

    # 5. Performance Summary
    print_banner("4. System Evaluation Summary")
    stats = brain.stats()
    avg_latency = np.mean(latencies) if latencies else 0.0
    print(f"Total Graph Entities Ingested : {stats['graph_nodes']}")
    print(f"Total Structural Edges        : {stats['graph_edges']}")
    print(f"Total Word Graph Nodes        : {stats['word_nodes']}")
    print(f"Abstract Concept Taxonomies   : {stats['abstract_concepts']}")
    print(f"Average Reasoning Latency     : {avg_latency:.2f}ms (CPU)")
    print(f"GPU / VRAM Consumption        : 0 MB (100% CPU)")

    print("\n" + "=" * 75)
    print("  EVALUATION QUESTIONS FOR YOU TO TEST LIVE:")
    print("=" * 75)
    for idx, q in enumerate(eval_questions, 1):
        print(f"  {idx}. \"{q}\"")
    print(f"  6. \"What did Lily do after finding the object?\"")
    print(f"  7. \"Who helped in the garden?\"")
    print("\nRun any custom query directly in Python:")
    print("  brain.query(\"<your question here>\")")
    print("=" * 75)


if __name__ == "__main__":
    main()
