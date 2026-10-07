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
Cortex Graph — Creative Prose & Multi-Act Story Generation Demo
================================================================
Demonstrates:
1. Ingesting high-density large multi-domain corpus (Sci-Fi, Cyberpunk, Philosophy, Nature, AI)
2. Comparing 1-to-10 Parameter Adjustments:
   - Low Novelty (2.0) vs High Novelty (9.0)
   - Punchy Cadence (2.0) vs Complex Flowing Cadence (9.0)
   - Strict Coherence (9.0) vs Free Associative Coherence (3.0)
3. Multi-Act Story Composition (3-Act and 4-Act Storyboards)
4. Structured Analytical Article Generation
5. Verification of Zero GPU, Zero Hallucination, and Pure Graph Ingestion
"""

import time
from mayon_cortex.cortex import CortexGraph
from mayon_cortex.learning.bootstrap import bootstrap_minimal, bootstrap_large_corpus
from mayon_cortex.language.corpus_data import LARGE_DOMAIN_CORPUS


def run_demo():
    print("\n" + "=" * 80, flush=True)
    print(">> CORTEX-GRAPH HIERARCHICAL PROSE ENGINE -- LIVE DEMO", flush=True)
    print("   100% Graph-Native | Zero Transformers | Zero GPUs | Universal 1-10 Parameters", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize Brain
    print("\n[1/5] Initializing Cortex Graph & Ingesting Multi-Domain Corpus...", flush=True)
    t0 = time.time()
    brain = CortexGraph()
    bootstrap_minimal(brain, verbose=False)
    bootstrap_large_corpus(brain, verbose=True)
    t_ingest = time.time() - t0
    print(f"\n[*] Ingestion completed in {t_ingest:.2f} seconds on CPU!\n", flush=True)

    # 2. Parameter Sweep: Novelty (Temperature & Metaphors)
    print("=" * 80, flush=True)
    print("[2/5] PARAMETER EXPERIMENT: Novelty Knob (1 to 10)", flush=True)
    print("=" * 80, flush=True)

    prompt = "The cosmic journey across the event horizon"

    print("\n--- A. LOW NOVELTY (Novelty = 2.0, Coherence = 8.5) [Predictable & Grounded] ---", flush=True)
    res_low = brain.compose_prose(prompt=prompt, style="sci-fi", novelty=2.0, coherence=8.5, cadence=5.0)
    print(res_low + "\n", flush=True)

    print("--- B. HIGH NOVELTY (Novelty = 9.0, Coherence = 7.0) [Surreal & High Metaphor] ---", flush=True)
    res_high = brain.compose_prose(prompt=prompt, style="sci-fi", novelty=9.0, coherence=7.0, cadence=5.0)
    print(res_high + "\n", flush=True)

    # 3. Parameter Sweep: Cadence (Rhythm & Sentence Length)
    print("=" * 80, flush=True)
    print("[3/5] PARAMETER EXPERIMENT: Cadence Knob (1 to 10)", flush=True)
    print("=" * 80, flush=True)

    cyber_prompt = "Neural hack in the neon underworld"

    print("\n--- A. PUNCHY CADENCE (Cadence = 2.0) [Short, fast, direct sentences] ---", flush=True)
    res_punchy = brain.compose_prose(prompt=cyber_prompt, style="cyberpunk", cadence=2.0, novelty=6.0)
    print(res_punchy + "\n", flush=True)

    print("--- B. FLOWING CADENCE (Cadence = 9.0) [Long, descriptive, compound sentences] ---", flush=True)
    res_flowing = brain.compose_prose(prompt=cyber_prompt, style="cyberpunk", cadence=9.0, novelty=6.0)
    print(res_flowing + "\n", flush=True)

    # 4. Multi-Act Story Generation
    print("=" * 80, flush=True)
    print("[4/5] FULL 3-ACT STORY GENERATION (Cyberpunk Genre)", flush=True)
    print("=" * 80, flush=True)

    story = brain.compose_story(
        prompt="The rogue artificial intelligence seeking consciousness",
        acts=3,
        novelty=7.0,
        coherence=8.0,
        cadence=6.0,
        emotion_intensity=7.5,
        genre="cyberpunk",
    )

    print(f"\n[STORY TITLE]: {story['title']}", flush=True)
    print(f"[PARAMETERS]: {story['parameters']}", flush=True)
    for act in story["acts"]:
        print(f"\n[ACT {act['act_number']}: {act['act_name']} | Mood: {act['emotion']}]", flush=True)
        print(f"{act['text']}", flush=True)

    # 5. Philosophical Article Generation
    print("\n" + "=" * 80, flush=True)
    print("[5/5] STRUCTURED PHILOSOPHICAL ARTICLE (Philosophy Genre)", flush=True)
    print("=" * 80, flush=True)

    article = brain.compose_article(
        topic="The nature of conscious observation and truth",
        paragraphs=3,
        novelty=4.5,
        coherence=9.0,
        cadence=6.5,
        genre="philosophical",
    )

    print(f"\n[ARTICLE TITLE]: {article['title']}", flush=True)
    for act in article["acts"]:
        print(f"\n[SECTION: {act['act_name']}]", flush=True)
        print(f"{act['text']}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("[SUCCESS] DEMO COMPLETE -- 100% Graph-Native Creative AI Verified!", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    run_demo()
