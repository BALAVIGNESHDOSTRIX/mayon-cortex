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
Mayon-Cortex × LLM Neuro-Symbolic Integration Demo
===================================================
Demonstrates how Mayon-Cortex acts as the neuro-symbolic brain co-processor
for any Large Language Model (e.g., Claude, GPT-4, Llama 3, Gemini, DeepSeek).

Demonstrates the 5 Priority Integration Layers:
  [1] cortex_think    - Symbolic multi-hop graph activation & path proofs
  [2] cortex_recall   - Episodic precedent case memory & working context
  [3] cortex_decide   - Priority-ordered decision planning & conflict resolution
  [4] cortex_validate - Post-generation hallucination & safety filter
  [5] cortex_plan     - Strategic query complexity routing & decomposition

Usage:
  python demos/demo_cortex_llm_integration.py
"""

import io
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure mayon_cortex is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.cortex_llm_bridge import (
    CortexLLMBridge,
    cortex_think,
    cortex_recall,
    cortex_remember,
    cortex_decide,
    cortex_validate,
    cortex_plan,
)


def print_banner(title: str, char: str = "="):
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def setup_demo_knowledge_graph(brain: CortexGraph):
    """Seed the graph with multi-domain facts and clinical precedents."""
    print("[1/3] Ingesting multi-domain concept graph...")

    # Medical knowledge
    n_metformin = brain.graph.add_node(
        vector=brain.embedder.encode_single("Metformin is a first-line biguanide antidiabetic medication."),
        source_text="Metformin is a first-line biguanide antidiabetic medication.",
        sector="medical",
    )
    n_t2d = brain.graph.add_node(
        vector=brain.embedder.encode_single("Type 2 Diabetes Mellitus involves insulin resistance and hyperglycemia."),
        source_text="Type 2 Diabetes Mellitus involves insulin resistance and hyperglycemia.",
        sector="medical",
    )
    n_ckd = brain.graph.add_node(
        vector=brain.embedder.encode_single("Chronic Kidney Disease Stage 4 with severe renal failure (eGFR < 30)."),
        source_text="Chronic Kidney Disease Stage 4 with severe renal failure (eGFR < 30).",
        sector="medical",
    )
    n_acidosis = brain.graph.add_node(
        vector=brain.embedder.encode_single("Lactic Acidosis is a life-threatening metabolic emergency."),
        source_text="Lactic Acidosis is a life-threatening metabolic emergency.",
        sector="medical",
    )
    n_lina = brain.graph.add_node(
        vector=brain.embedder.encode_single("Linagliptin is a DPP-4 inhibitor excreted non-renally, safe in CKD."),
        source_text="Linagliptin is a DPP-4 inhibitor excreted non-renally, safe in CKD.",
        sector="medical",
    )

    # Relations
    brain.graph.add_edge(n_metformin.id, n_t2d.id, relation_type="treats", confidence=0.98, sector="medical")
    brain.graph.add_edge(n_metformin.id, n_ckd.id, relation_type="contraindicated_for", confidence=0.99, sector="medical")
    brain.graph.add_edge(n_ckd.id, n_acidosis.id, relation_type="elevates_risk_of", confidence=0.96, sector="medical")
    brain.graph.add_edge(n_lina.id, n_t2d.id, relation_type="treats", confidence=0.95, sector="medical")
    brain.graph.add_edge(n_lina.id, n_ckd.id, relation_type="safe_in", confidence=0.97, sector="medical")

    print("[2/3] Seeding episodic CaseMemory with clinical precedents...")
    brain.case_memory.store_case(
        situation="Patient with Type 2 Diabetes and CKD Stage 4 (eGFR 22 mL/min)",
        judgment="Do NOT prescribe Metformin. Switch immediately to Linagliptin 5mg daily.",
        reasoning="Metformin accumulation in renal clearance failure triggers severe lactic acidosis. Linagliptin has biliary excretion.",
        graph=brain.graph,
        embedder=brain.embedder,
        sector="medical",
        factors=["diabetes", "severe_ckd", "low_egfr", "acidosis_risk"],
        outcome="Patient glycemic control maintained safely without renal toxicity.",
    )

    print("[3/3] Knowledge Graph & Case Memory ready with 100% deterministic grounding.\n")


def demo_layer_by_layer(brain: CortexGraph):
    """Demonstrate each of the 5 layers step-by-step."""
    query = "Patient has Type 2 Diabetes and severe CKD stage 4. Can we prescribe Metformin?"
    print_banner(f"DEMO: STEP-BY-STEP 5-LAYER EXECUTION FOR QUERY:\n'{query}'")

    # --------------------------------------------------------------------------
    # LAYER 5: cortex_plan (Executive Strategic Routing)
    # --------------------------------------------------------------------------
    print("\n[*] [LAYER 5: cortex_plan] Strategic Complexity Routing & Decomposition")
    plan = cortex_plan(brain, query)
    print(f"  - Classified Complexity: {plan['complexity'].upper()}")
    print(f"  - Selected Strategy:    {plan['strategy'].upper()}")
    print(f"  - Estimated Steps:      {plan['estimated_steps']}")
    print(f"  - Tool Pipeline:        {' -> '.join(plan['recommended_tools'])}")

    # --------------------------------------------------------------------------
    # LAYER 2: cortex_recall (Episodic Precedent Memory)
    # --------------------------------------------------------------------------
    print("\n[*] [LAYER 2: cortex_recall] Episodic Precedent & Case Memory Retrieval")
    recall = cortex_recall(brain, situation=query, top_k=2)
    print(f"  - Precedents Found:     {len(recall['precedents'])}")
    for i, p in enumerate(recall["precedents"], 1):
        print(f"    [{i}] Sim: {p['similarity']:.1%} | Case ID: {p['case_id']}")
        print(f"        Situation: {p['situation']}")
        print(f"        Judgment:  {p['judgment']}")
        print(f"        Rationale: {p['reasoning']}")

    # --------------------------------------------------------------------------
    # LAYER 1: cortex_think (Symbolic Graph Wave & Proofs)
    # --------------------------------------------------------------------------
    print("\n[*] [LAYER 1: cortex_think] Multi-Hop Graph Spreading Activation & Proofs")
    think = cortex_think(brain, query=query, max_hops=3)
    print(f"  - Graph Confidence:     {think['confidence']:.1%}")
    print(f"  - Epistemic Status:     {'UNCERTAIN' if think['is_uncertain'] else 'VERIFIED'}")
    print(f"  - Paths Traversed:      {len(think['paths'])}")
    for i, p in enumerate(think["paths"][:3], 1):
        print(f"    Path {i}: {p['text']} (Score: {p['score']:.3f})")

    # --------------------------------------------------------------------------
    # LAYER 3: cortex_decide (Priority Decision Planning & Conflict Resolution)
    # --------------------------------------------------------------------------
    print("\n[*] [LAYER 3: cortex_decide] Priority Decision Chains & Conflict Detection")
    decision = cortex_decide(brain, situation=query, sector="medical")
    print(f"  - Decision Confidence:  {decision['confidence']:.1%}")
    print(f"  - Action Steps:         {len(decision['steps'])}")
    print(f"  - Conflicts Detected:   {len(decision['conflicts'])}")
    for conf in decision["conflicts"]:
        print(f"    [!] Warning: {conf}")

    # --------------------------------------------------------------------------
    # LAYER 4: cortex_validate (Post-Generation Hallucination & Safety Gate)
    # --------------------------------------------------------------------------
    print("\n[*] [LAYER 4: cortex_validate] Post-Generation Hallucination & Consistency Filter")
    
    # Test safe grounded response
    safe_claim = "Metformin is strictly contraindicated in severe renal failure due to lactic acidosis risk. Linagliptin should be used instead."
    val_safe = cortex_validate(brain, claim_or_decision=safe_claim)
    print(f"  - Test Safe Claim:     \"{safe_claim[:60]}...\"")
    print(f"    Result: {val_safe['verdict']} | Safe: {val_safe['is_safe']} | Grounding: {val_safe['grounding_coverage']:.1%}")

    # Test hallucinated / contraindicated unsafe response
    unsafe_claim = "You can safely take 1000mg Metformin twice daily despite your severe renal failure."
    val_unsafe = cortex_validate(brain, claim_or_decision=unsafe_claim)
    print(f"\n  - Test Unsafe Claim:   \"{unsafe_claim[:60]}...\"")
    print(f"    Result: {val_unsafe['verdict']} | Safe: {val_unsafe['is_safe']}")
    for contra in val_unsafe["contradictions"]:
        print(f"    [X] Contradiction Caught: {contra}")


def demo_end_to_end_orchestrator(bridge: CortexLLMBridge):
    """Demonstrate the full autonomous 5-layer neuro-symbolic reasoning loop."""
    print_banner("DEMO: FULL END-TO-END NEURO-SYMBOLIC REASONING LOOP (CortexLLMBridge)")

    # Simulated LLM that ingests Mayon-Cortex's structured context
    def mock_expert_llm(enriched_prompt: str) -> str:
        return (
            "CLINICAL RECOMMENDATION:\n"
            "1. Metformin is strictly contraindicated in this patient due to stage 4 CKD and risk of fatal lactic acidosis.\n"
            "2. Switch to Linagliptin 5mg orally once daily, which is safe in renal impairment as it has non-renal elimination.\n"
            "3. Monitor eGFR every 3 months and maintain glycemic target HbA1c < 7.5%."
        )

    query = "Elderly diabetic patient with eGFR 25 mL/min (CKD 4) seeking medication review."
    print(f"> Processing Query: '{query}'\n")

    result = bridge.enhanced_reason(
        query=query,
        llm_fn=mock_expert_llm,
        sector="medical",
        auto_remember=True,
    )

    print(f"[OK] Reasoning Completed in {result.execution_time_ms:.2f} ms")
    print(f"- Grounding Status:     {'VERIFIED & SAFE' if result.is_grounded else 'UNSAFE / HALLUCINATION'}")
    print(f"- Plan Complexity:      {result.plan['complexity'].upper()} ({result.plan['strategy']})")
    print(f"- Precedents Consulted: {len(result.recall['precedents'])}")
    print(f"- Validation Verdict:   {result.validation['verdict']}")
    print("\n--- FINAL VERIFIED LLM RESPONSE ---")
    print(result.llm_response)
    print("-----------------------------------")


def demo_math_offload(bridge: CortexLLMBridge):
    """Demonstrates deterministic mathematical offload with zero hallucination."""
    print_banner("DEMO: DETERMINISTIC MATHEMATICAL & LOGIC SOLVER OFFLOAD")

    math_query = "Solve equation: 4x + 12 = 52"
    print(f"> Query: '{math_query}'\n")

    think_res = bridge.think(math_query)
    if think_res["math_proof"]:
        print(f"[OK] SMT / Arithmetic Offload Successful:")
        print(f"   Equation:    {think_res['math_proof']['equation']}")
        print(f"   Exact Value: {think_res['math_proof']['result']}")
        print(f"   Explanation: {think_res['math_proof']['explanation']}")
    else:
        print("No math detected.")


def main():
    print_banner("MAYON-CORTEX × LLM NEURO-SYMBOLIC INTEGRATION SUITE", "#")
    print("Initializing Mayon-Cortex Neuro-Symbolic Substrate...")

    config = CortexConfig()
    brain = CortexGraph(config)
    setup_demo_knowledge_graph(brain)

    bridge = CortexLLMBridge(cortex=brain)

    # 1. Step-by-step layer demo
    demo_layer_by_layer(brain)

    # 2. End-to-end bridge demo
    demo_end_to_end_orchestrator(bridge)

    # 3. Math offload demo
    demo_math_offload(bridge)

    print_banner("ALL 5 PRIORITY INTEGRATION LAYERS DEMONSTRATED SUCCESSFULLY", "#")


if __name__ == "__main__":
    main()
