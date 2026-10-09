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
Mayon-Cortex Phase 2: Real-Data Inference & Pluggable LLM Demo
==============================================================
Runs real neuro-symbolic inference across all 5 Phase 2 Pillars:
  [Pillar 1] DualProcessRouter: Dynamic System 1 (fast) & System 2 (deterministic proofs)
  [Pillar 2] AutonomousLearner: Curiosity-driven epistemic gap detection & self-expansion
  [Pillar 3] CortexEnsemble: Federated multi-agent debate, contraindication checks & voting
  [Pillar 4] EmbodiedCortexAgent: Perceive → Reason → Act (tools) → Learn loop
  [Pillar 5] SelfImprovementEngine: Proof quality scoring & Hebbian self-adaptation

Ready for LLM connection:
  Simply set `my_llm_connector` to your local Ollama / vLLM / OpenAI / Claude API.
  If none is connected, Mayon-Cortex runs in pure deterministic CPU verbalization mode.

Usage:
  python demos/demo_phase2_real_data_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure UTF-8 output and line buffering on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

# Ensure mayon_cortex is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.dual_process_router import DualProcessRouter
from mayon_cortex.learning.autonomous_learner import AutonomousLearner
from mayon_cortex.integration.cortex_ensemble import CortexEnsemble
from mayon_cortex.integration.embodied_agent import ActionRegistry, EmbodiedCortexAgent
from mayon_cortex.learning.self_improvement import SelfImprovementEngine
from mayon_cortex.memory.error_memory import ErrorMemory


# ==============================================================================
# OPTIONAL: Plug In Your LLM Here When Connected
# ==============================================================================
def create_llm_connector(model_name: str = "auto") -> Optional[Callable[[str], str]]:
    """
    Hook to connect your local or cloud LLM (e.g. Ollama, vLLM, OpenAI, Claude).
    To connect Ollama (e.g., Llama 3 8B, Qwen, DeepSeek):
        import urllib.request
        def ollama_fn(prompt: str) -> str:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": "llama3:8b", "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode())["response"]
        return ollama_fn
    """
    # By default returns None (Mayon-Cortex uses pure deterministic symbolic reasoning)
    return None


def print_banner(title: str, char: str = "="):
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def load_real_world_corpus(brain: CortexGraph, corpus_path: Path):
    """Load real multi-domain facts from knowledge corpus."""
    if not corpus_path.exists():
        print(f"  ⚠️ Corpus file {corpus_path} not found, using baseline facts.")
        brain.ingest("Metformin treats Type 2 Diabetes Mellitus.", sector="medical_pharmacology")
        brain.ingest("Severe Renal Impairment contraindicates Metformin.", sector="medical_pharmacology")
        brain.ingest("Linagliptin is safe in Severe Renal Impairment.", sector="medical_pharmacology")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for sector, facts in data.items():
            for fact in facts:
                brain.ingest(fact, sector=sector)
    print(f"  ✓ Ingested real domain corpus into MayonGraph tissue ({brain.graph.num_nodes} concepts, {brain.graph.num_edges} edges)")


def main():
    print_banner("MAYON-CORTEX PHASE 2: REAL-DATA INFERENCE PIPELINE")
    storage_root = Path("cortex_storage/phase2")
    storage_root.mkdir(parents=True, exist_ok=True)

    config = CortexConfig()
    cortex = CortexGraph(config)

    # 1. Load real domain knowledge
    corpus_file = Path("data/real_world_domains/knowledge_corpus.json")
    print("\n[Step 1/6] Ingesting Real Multi-Domain Knowledge...")
    load_real_world_corpus(cortex, corpus_file)

    # Save initial brain state to disk
    brain_storage = storage_root / "cortex_brain"
    cortex.save(str(brain_storage))
    print(f"  ✓ Brain state persisted to: {brain_storage}")

    # Check for optional connected LLM
    llm_fn = create_llm_connector()
    llm_mode = "External LLM Connected" if llm_fn else "Pure Deterministic CPU Verbalizer (No GPU needed)"
    print(f"  • Verbalization Mode: {llm_mode}")

    # ==========================================================================
    # PILLAR 1: DualProcessRouter (System 1 vs System 2)
    # ==========================================================================
    print_banner("[Pillar 1] DualProcessRouter: Dynamic System 1 & System 2 Routing", "-")
    router = DualProcessRouter(cortex=cortex, storage_dir=storage_root / "router")

    # Query A: Trivial query (System 1)
    q1 = "What is Metformin?"
    print(f"Query 1 (Trivial): '{q1}'")
    res1 = router.route(q1, llm_fn=llm_fn)
    print(f"  • Mode: {res1.routing_mode} | Complexity: {res1.complexity}")
    print(f"  • Confidence: {res1.confidence:.2f} | Latency: {res1.execution_time_ms:.1f}ms")
    print(f"  • Output:\n    {res1.llm_output[:140]}...")

    # Query B: Complex clinical decision with contraindication (System 2)
    q2 = "Can we safely prescribe Metformin to a patient with severe renal impairment (eGFR < 30)?"
    print(f"\nQuery 2 (Complex Clinical): '{q2}'")
    res2 = router.route(q2, llm_fn=llm_fn, sector="medical_pharmacology", force_system_2=True)
    print(f"  • Mode: {res2.routing_mode} | Complexity: {res2.complexity}")
    print(f"  • Confidence: {res2.confidence:.2f} | Latency: {res2.execution_time_ms:.1f}ms")
    print(f"  • Contradiction Flag: {res2.has_contradiction}")
    print(f"  • Verified Output:\n    {res2.llm_output}")

    # ==========================================================================
    # PILLAR 2: AutonomousLearner (Curiosity-Driven Self-Expansion)
    # ==========================================================================
    print_banner("[Pillar 2] AutonomousLearner: Curiosity-Driven Epistemic Self-Expansion", "-")
    learner = AutonomousLearner(cortex=cortex, storage_dir=storage_root / "curiosity")

    print("Detecting knowledge gaps across graph topology...")
    gaps = learner.detect_knowledge_gaps(failed_query="Superconducting graphene avionics", max_gaps=2)
    for g in gaps:
        print(f"  • Discovered Epistemic Void: '{g.concept}' (Gap Score: {g.gap_score:.2f})")
        print(f"    Inquiry Question: {g.inquiry_question}")

    print("\nExecuting curiosity resolution cycle with grounded facts...")
    expansion = learner.run_curiosity_cycle(
        trigger_query="Superconducting graphene avionics",
        external_facts={
            "default": [
                "Superconducting graphene reduces electrical resistance in high-frequency avionics.",
                "Reduced electrical resistance minimizes thermal dissipation in flight computers.",
            ]
        },
        sector="aerospace_engineering",
    )
    print(f"  ✓ Resolved {expansion.gaps_resolved} gaps; added {expansion.new_nodes_added} nodes, {expansion.new_edges_added} edges")
    print(f"  ✓ Epistemic Confidence Gain: +{expansion.confidence_gain:.2f}")

    # ==========================================================================
    # PILLAR 3: CortexEnsemble (Multi-Agent Consensus & Voting)
    # ==========================================================================
    print_banner("[Pillar 3] CortexEnsemble: Federated Multi-Agent Voting", "-")
    ensemble = CortexEnsemble(storage_dir=storage_root / "ensembles")

    # Create specialized Medical & Legal brains
    med_agent = CortexGraph(config)
    med_agent.ingest("Metformin treats Type 2 Diabetes.", sector="medical")
    med_agent.ingest("Severe Renal Failure contraindicates Metformin.", sector="medical")

    legal_agent = CortexGraph(config)
    legal_agent.ingest("GDPR Article 44 mandates encryption for international transfers.", sector="legal")
    legal_agent.ingest("Transferring unencrypted patient records violates data privacy.", sector="legal")

    ensemble.register_agent("medical_specialist", med_agent, sector="medical")
    ensemble.register_agent("compliance_officer", legal_agent, sector="legal")

    ensemble_q = "What is the contraindication for Metformin?"
    print(f"Querying Ensemble: '{ensemble_q}'")
    ens_res = ensemble.query_ensemble(ensemble_q, llm_fn=llm_fn)
    print(f"  • Winner: [{ens_res.winner_agent_id}] (Sector: {ens_res.winner_sector})")
    print(f"  • Winning Confidence: {ens_res.winning_confidence:.2f}")
    print(f"  • Consensus Reached: {ens_res.consensus_reached}")
    print(f"  • Result Output: {ens_res.llm_output[:120]}...")

    # Save ensemble state
    ensemble.save_ensemble("clinical_compliance_team")
    print(f"  ✓ Ensemble state persisted to: {storage_root / 'ensembles' / 'clinical_compliance_team'}")

    # ==========================================================================
    # PILLAR 4: EmbodiedCortexAgent (Perception → Reason → Act → Learn)
    # ==========================================================================
    print_banner("[Pillar 4] EmbodiedCortexAgent: Perceive → Reason → Act → Learn", "-")
    actions = ActionRegistry()
    embodied_agent = EmbodiedCortexAgent(cortex=cortex, action_registry=actions, storage_dir=storage_root / "embodied")

    telemetry = {
        "turbine_inlet_temp": "1520C",
        "blade_vibration": "94Hz",
    }
    print(f"Perceiving Telemetry Observation: {telemetry}")
    episode = embodied_agent.execute_cycle(
        goal="Assess thermal stress and execute engine safety computation",
        observation=telemetry,
        required_actions=[
            ("compute_math", {"expression": "1520 - 1200"}),
            ("dispatch_alert", {"level": "WARNING", "message": "Turbine inlet temperature exceeds 1500C threshold"}),
        ],
        sector="aerospace_engineering",
    )

    print(f"  ✓ Episode Executed: {episode.episode_id} (Success: {episode.success})")
    for act in episode.actions_executed:
        print(f"    - Action [{act.action_name}]: {act.output}")
    print(f"  ✓ Hebbian Precedent Archived in CaseMemory: {episode.learned_precedent_id}")

    # ==========================================================================
    # PILLAR 5: SelfImprovementEngine (Proof Evaluation & Error Memory)
    # ==========================================================================
    print_banner("[Pillar 5] SelfImprovementEngine: Recursive Proof Evaluation", "-")
    error_store = ErrorMemory(storage_root / "errors")
    improver = SelfImprovementEngine(cortex=cortex, error_memory=error_store, storage_dir=storage_root / "self_improvement")

    proof_eval_q = "Metformin and Type 2 Diabetes Mellitus"
    c_resp = cortex.query(proof_eval_q)
    score = improver.evaluate_proof(c_resp, query=proof_eval_q)
    print(f"Evaluating Proof Quality for: '{proof_eval_q}'")
    print(f"  • Overall Score: {score.overall_score:.2f} ({score.tier})")
    print(f"  • Deductive Soundness: {score.logic_validity_score:.2f} | Contradiction Freedom: {score.contradiction_freedom:.2f}")

    action = improver.improve_from_proof(proof_eval_q, c_resp, sector="medical_pharmacology")
    print(f"  ✓ Adaptation Action: {action.description}")

    # Demonstrate ErrorMemory prevention
    print("\nSimulating Invalid Claim Refutation:")
    err_id = error_store.record_error(
        query="Prescribe high-dose Metformin in stage 5 kidney failure",
        faulty_reasoning="Metformin does not depend on renal excretion",
        contradiction_type="contraindication",
        correction="Fatal contraindication: Metformin accumulation causes metabolic lactic acidosis.",
        sector="medical_pharmacology",
    )
    print(f"  ✓ Contradiction archived in ErrorMemory as: {err_id}")
    refuted = error_store.check_for_known_error("Can we Prescribe high-dose Metformin in stage 5 kidney failure?")
    if refuted:
        print(f"  🛡️ Guardrail Intercepted Repeated Error: '{refuted.correction}'")

    print_banner("INFERENCE COMPLETE — ALL 5 PILLARS ACTIVE WITH STORAGE")
    print(f"All data, models, episodes, telemetry, and error memories saved in:")
    print(f"  👉 {storage_root.resolve()}")
    print("\nTo connect an LLM, implement `create_llm_connector()` in this file with your endpoint.")


if __name__ == "__main__":
    main()
