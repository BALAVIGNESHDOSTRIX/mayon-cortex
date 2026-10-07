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
Cortex-Graph v5 — Third Paradigm AI Demo
==========================================
Demonstrates ALL capabilities of the upgraded system:
  Phase 0:  Unified Pipeline (from_text → ask/think/generate)
  Phase 1:  System 2 Thinking (multi-step reasoning)
  Phase 1.5: Cognitive Modes (factual vs creative vs solving)
  Phase 7:  Case-Based Judgment (precedent-based decisions)
  Phase Ω:  Hyperdimensional + Morphogenetic + Forward-Forward

Run: python demo_full.py
"""

import sys
import time
sys.path.insert(0, ".")


def main():
    print("=" * 60)
    print("  CORTEX-GRAPH v5 -- Third Paradigm AI Demo")
    print("  No GPU. No Backpropagation. Pure Graph Intelligence.")
    print("=" * 60)
    print()

    # ── Phase 0: Unified Pipeline ──
    print("[*] Phase 0: Creating brain from text...")
    from mayon_cortex.pipeline import CortexPipeline

    brain = CortexPipeline.from_text([
        "Aspirin is a nonsteroidal anti-inflammatory drug.",
        "Aspirin treats headaches, fever, and inflammation.",
        "Aspirin can cause stomach ulcers as a side effect.",
        "Ibuprofen is also a nonsteroidal anti-inflammatory drug.",
        "Ibuprofen treats pain and reduces fever.",
        "Acetaminophen treats pain but is not anti-inflammatory.",
        "Acetaminophen overdose can cause liver damage.",
        "NSAIDs should not be used with blood thinners.",
        "Warfarin is a blood thinner used to prevent blood clots.",
        "Python is a programming language used for data science.",
        "Machine learning uses algorithms to learn from data.",
        "Neural networks are inspired by the human brain.",
        "Knowledge graphs store structured relationships between concepts.",
        "The heart pumps blood through the circulatory system.",
        "The brain controls thinking, memory, and emotions.",
        "Photosynthesis converts sunlight into glucose in plants.",
    ], sector="general")

    stats = brain._stats
    print(f"  [OK] Brain created: {stats.total_nodes} nodes, {stats.total_edges} edges")
    print()

    # ── Phase 1: Factual Q&A (System 1) ──
    print("[*] Phase 1: Factual Q&A (System 1 -- fast lookup)...")
    response = brain.ask("What treats headaches?")
    print(f"  Q: What treats headaches?")
    print(f"  A: {response.text[:200]}")
    print(f"  Confidence: {response.confidence:.2f}")
    print()

    # ── Phase 1: System 2 Thinking ──
    print("[*] Phase 1: System 2 Thinking (multi-step reasoning)...")
    result = brain.think("Should a patient taking Warfarin also take Aspirin?")
    print(f"  Q: Should a patient on Warfarin also take Aspirin?")
    print(f"  A: {result.answer[:300]}")
    print(f"  Complexity: {result.complexity}")
    print(f"  Strategy: {result.strategy_used}")
    print(f"  Steps taken: {result.steps_taken}")
    print(f"  Confidence: {result.confidence:.2f}")
    print()

    # ── Phase 1.5: Cognitive Modes ──
    print("[*] Phase 1.5: Cognitive Mode Detection...")
    from mayon_cortex.executive.cognitive_modes import CognitiveModeController
    controller = CognitiveModeController()

    test_queries = [
        ("What is Aspirin?", "FACTUAL"),
        ("Write a poem about the brain", "CREATIVE"),
        ("How to solve 3x + 5 = 20", "PROBLEM_SOLVING"),
        ("Should we approve this treatment?", "JUDGING"),
        ("What connects the heart and the brain?", "EXPLORATORY"),
    ]
    for q, expected in test_queries:
        mode = controller.detect_mode(q)
        print(f"  '{q[:45]}...' -> {mode.value}")
    print()

    # ── Phase 1.5: Feedforward Cascade ──
    print("[*] Phase 1.5: Feedforward Cascade (bio-plausible)...")
    from mayon_cortex.dynamics.feedforward_cascade import FeedforwardCascade
    from mayon_cortex.executive.cognitive_modes import ModeConfig

    cascade = FeedforwardCascade(concept_dim=brain.config.concept_dim)
    query_vec = brain.brain.embedder.encode_single("How do drugs interact?")

    # Factual mode
    factual_result = cascade.cascade(query_vec, brain.brain.graph, ModeConfig.factual())
    # Creative mode
    creative_result = cascade.cascade(query_vec, brain.brain.graph, ModeConfig.creative())

    print(f"  Factual mode:  {len(factual_result.fired_nodes)} nodes fired")
    print(f"  Creative mode: {len(creative_result.fired_nodes)} nodes fired")
    print()

    # ── Phase 7: Case-Based Judgment ──
    print("[*] Phase 7: Case-Based Judgment (learn from precedent)...")
    from mayon_cortex.memory.case_memory import CaseMemory
    from mayon_cortex.reasoning.judgment_engine import JudgmentEngine

    case_mem = CaseMemory()
    judge = JudgmentEngine(case_mem)

    # Teach cases
    case_mem.store_case(
        situation="Employee shared confidential client data on social media",
        judgment="Immediate termination with cause",
        reasoning="Breach of NDA, violation of company policy Section 8.1, potential legal liability",
        graph=brain.brain.graph,
        embedder=brain.brain.embedder,
        sector="general",
        factors=["confidential", "social media", "employee", "data breach"],
    )
    case_mem.store_case(
        situation="Employee accidentally sent internal email to wrong recipient",
        judgment="Written warning, mandatory training",
        reasoning="Unintentional, low severity, no malice, first offense",
        graph=brain.brain.graph,
        embedder=brain.brain.embedder,
        sector="general",
        factors=["accidental", "email", "employee", "internal"],
    )

    # Judge new case
    judgment = judge.judge(
        "Employee posted client project details on LinkedIn",
        graph=brain.brain.graph,
        embedder=brain.brain.embedder,
    )
    print(f"  Situation: Employee posted client project details on LinkedIn")
    print(f"  Verdict: {judgment.verdict[:100]}")
    print(f"  Confidence: {judgment.confidence:.2f}")
    for step in judgment.reasoning_chain[:3]:
        print(f"    {step[:80]}")
    if judgment.dissent:
        print(f"  Dissent: {judgment.dissent[:100]}")
    print()

    # ── Phase Ω: Hyperdimensional Computing ──
    print("[*] Phase Omega: Hyperdimensional Computing (10,000-dim binary)...")
    from mayon_cortex.memory.hyperdimensional import HyperVector, HyperdimensionalMemory

    aspirin_hv = HyperVector(10000)
    drug_hv = HyperVector(10000)
    headache_hv = HyperVector(10000)
    random_hv = HyperVector(10000)

    # Bind: aspirin TREATS headache
    treats_rel = HyperVector.bind(aspirin_hv, headache_hv)
    # Bundle: drug category = {aspirin, ibuprofen}
    ibuprofen_hv = HyperVector(10000)
    drug_category = HyperVector.bundle(aspirin_hv, ibuprofen_hv)

    print(f"  aspirin <-> drug similarity:     {aspirin_hv.similarity(drug_hv):.3f}")
    print(f"  aspirin <-> random similarity:   {aspirin_hv.similarity(random_hv):.3f}")
    print(f"  drug_category <-> aspirin:       {drug_category.similarity(aspirin_hv):.3f}")
    print(f"  drug_category <-> ibuprofen:     {drug_category.similarity(ibuprofen_hv):.3f}")
    print()

    # ── Phase Ω: Morphogenetic Patterns ──
    print("[*] Phase Omega: Morphogenetic Pattern Generation...")
    from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator
    morph = MorphogeneticGenerator()

    # Generate a cellular automaton (fast)
    rule110 = morph.grow_cellular(rule=110, width=64, height=64)
    print(f"  Rule 110 cellular automaton: {rule110.shape} (Turing-complete!)")

    # Generate a quick tree
    tree = morph.grow_tree(width=128, height=128, depth=4)
    print(f"  L-system fractal tree: {tree.shape}")
    print()

    # ── Phase Ω: Forward-Forward Learning ──
    print("[*] Phase Omega: Forward-Forward Learning (no backprop)...")
    from mayon_cortex.learning.forward_forward import ForwardForwardLearner
    ff = ForwardForwardLearner()

    separations = ff.train_from_graph(brain.brain.graph, num_epochs=3)
    if separations:
        avg_sep = sum(separations.values()) / len(separations)
        print(f"  Trained {len(separations)} nodes")
        print(f"  Average positive/negative separation: {avg_sep:.3f}")
    print()

    # ── Phase Ω: Predictive Coding ──
    print("[*] Phase Omega: Predictive Coding (brain algorithm)...")
    from mayon_cortex.learning.predictive_coder import PredictiveCoder
    pc = PredictiveCoder()

    errors = pc.learn_all(brain.brain.graph, num_epochs=2)
    if errors:
        avg_err = sum(errors.values()) / len(errors)
        print(f"  Trained {len(errors)} nodes with predictive coding")
        print(f"  Average prediction error: {avg_err:.4f}")

        # Find surprises
        surprises = pc.find_surprises(brain.brain.graph, top_k=3)
        if surprises:
            print(f"  Top surprises (knowledge frontiers):")
            for src, tgt, err in surprises[:3]:
                src_text = brain.brain.graph.nodes.get(src, None)
                tgt_text = brain.brain.graph.nodes.get(tgt, None)
                s = src_text.source_text[:30] if src_text else src[:15]
                t = tgt_text.source_text[:30] if tgt_text else tgt[:15]
                print(f"    {s} -> {t}: error={err:.4f}")

        # Strengthen well-predicted edges
        strengthened = pc.strengthen_edges(brain.brain.graph)
        print(f"  Edges strengthened by predictive coding: {strengthened}")
    print()

    # ── Summary ──
    print("=" * 60)
    print("  DEMO COMPLETE -- Third Paradigm AI Running on CPU")
    print("=" * 60)
    print()
    print("  Components demonstrated:")
    print("  [OK] Phase 0:  Unified Pipeline (from_text -> intelligence)")
    print("  [OK] Phase 1:  System 2 Thinking (multi-step reasoning)")
    print("  [OK] Phase 1.5: Cognitive Modes (5 brain modes)")
    print("  [OK] Phase 1.5: Feedforward Cascade (bio-plausible)")
    print("  [OK] Phase 7:  Case-Based Judgment (precedent + explanation)")
    print("  [OK] Phase Omega: Hyperdimensional Computing (10K-dim binary)")
    print("  [OK] Phase Omega: Morphogenetic Patterns (growth-based images)")
    print("  [OK] Phase Omega: Forward-Forward Learning (no backprop)")
    print("  [OK] Phase Omega: Predictive Coding (brain algorithm)")
    print()
    print(f"  Total graph: {brain.brain.graph.num_nodes} nodes, {len(brain.brain.graph.edges)} edges")
    print()
    print("  No GPU. No backpropagation. No trillion parameters.")
    print("  Pure graph-native intelligence. The Third Paradigm. [DONE]")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
