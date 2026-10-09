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
Comprehensive Phase 2 Pillars Test Suite with Real Domain Data & Storage
========================================================================
Tests:
  1. Pillar 1: DualProcessRouter (System 1/2 routing, telemetry, pluggable LLM, math)
  2. Pillar 2: AutonomousLearner (Epistemic gap detection, curiosity cycle, growth)
  3. Pillar 3: CortexEnsemble (Multi-agent routing, ConfidenceVoter, save/load)
  4. Pillar 4: EmbodiedCortexAgent (Perceive-Reason-Act-Learn, ActionRegistry, safety)
  5. Pillar 5: SelfImprovementEngine (Proof quality scoring, 3-tier adaptation, ErrorMemory)
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph
from mayon_cortex.integration.dual_process_router import (
    DualProcessRouter,
    HybridResponse,
)
from mayon_cortex.learning.autonomous_learner import (
    AutonomousLearner,
    KnowledgeGap,
    ExpansionResult,
)
from mayon_cortex.integration.cortex_ensemble import (
    CortexEnsemble,
    ConfidenceVoter,
    EnsembleResponse,
)
from mayon_cortex.integration.embodied_agent import (
    EmbodiedCortexAgent,
    ActionRegistry,
    ActionResult,
)
from mayon_cortex.learning.self_improvement import (
    SelfImprovementEngine,
    ProofQualityScore,
    ImprovementAction,
)
from mayon_cortex.memory.error_memory import ErrorMemory


@pytest.fixture(scope="module")
def real_data_cortex():
    """Build and seed a CortexGraph with real multi-domain data."""
    config = CortexConfig()
    brain = CortexGraph(config)

    corpus_file = Path("data/real_world_domains/knowledge_corpus.json")
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sector, facts in data.items():
                for fact in facts:
                    brain.ingest(fact, sector=sector)
    else:
        # Fallback inline seed
        brain.ingest("Metformin treats Type 2 Diabetes Mellitus.", sector="medical")
        brain.ingest("Severe Renal Impairment contraindicates Metformin.", sector="medical")
        brain.ingest("Linagliptin is safe in Severe Renal Impairment.", sector="medical")

    return brain


# ==============================================================================
# 1. PILLAR 1: DualProcessRouter
# ==============================================================================

def test_pillar_1_dual_process_router(real_data_cortex, tmp_path):
    storage_dir = tmp_path / "router_storage"
    router = DualProcessRouter(cortex=real_data_cortex, storage_dir=storage_dir)

    # 1. System 1: Trivial query
    res_s1 = router.route("What is Metformin?")
    assert res_s1.complexity in ["trivial", "moderate"]
    assert res_s1.llm_output is not None

    # 2. System 2: Complex clinical contraindication query with pluggable LLM mock
    llm_invoked = False
    def mock_llm_callback(prompt: str) -> str:
        nonlocal llm_invoked
        llm_invoked = True
        assert "Mayon-Cortex" in prompt
        return "Grounded Answer: In severe renal failure, Metformin is contraindicated. Use Linagliptin instead."

    res_s2 = router.route(
        query="Can we prescribe Metformin if patient has severe renal impairment?",
        llm_fn=mock_llm_callback,
        sector="medical_pharmacology",
        force_system_2=True,
    )

    assert llm_invoked is True
    assert res_s2.cortex_used is True
    assert res_s2.routing_mode == "SYSTEM_2_DECIDE"
    assert "Linagliptin" in res_s2.llm_output or "contraindicated" in res_s2.llm_output.lower()

    # 3. Verify telemetry saved to persistent storage
    summary = router.get_telemetry_summary()
    assert summary["total_queries"] >= 2
    assert router.history_file.exists()


# ==============================================================================
# 2. PILLAR 2: AutonomousLearner
# ==============================================================================

def test_pillar_2_autonomous_learner(real_data_cortex, tmp_path):
    storage_dir = tmp_path / "curiosity_storage"
    learner = AutonomousLearner(cortex=real_data_cortex, storage_dir=storage_dir)

    # 1. Detect knowledge gaps
    gaps = learner.detect_knowledge_gaps(failed_query="Hypothetical unknown quantum bio-catalyst", max_gaps=3)
    assert len(gaps) > 0
    assert isinstance(gaps[0], KnowledgeGap)

    # 2. Execute a full curiosity cycle with resolving facts
    ext_facts = {
        "default": [
            "Quantum bio-catalyst stabilizes ATP synthase enzymes during metabolic stress.",
            "ATP synthase stabilization prevents cellular apoptosis in ischemic tissue.",
        ]
    }
    cycle_res = learner.run_curiosity_cycle(
        trigger_query="Quantum bio-catalyst stabilizes ATP synthase",
        external_facts=ext_facts,
        sector="biochemistry",
    )

    assert isinstance(cycle_res, ExpansionResult)
    assert cycle_res.gaps_resolved >= 1
    assert cycle_res.new_nodes_added >= 0

    # 3. Check persistent storage files
    assert learner.gaps_file.exists()
    assert learner.cycles_file.exists()
    summary = learner.get_summary()
    assert summary["total_cycles_run"] >= 1


# ==============================================================================
# 3. PILLAR 3: CortexEnsemble & ConfidenceVoter
# ==============================================================================

def test_pillar_3_cortex_ensemble_and_storage(tmp_path):
    # Create two specialized brains
    config = CortexConfig()
    med_brain = CortexGraph(config)
    med_brain.ingest("Metformin treats Type 2 Diabetes.", sector="medical")
    med_brain.ingest("Severe Renal Failure contraindicates Metformin.", sector="medical")

    legal_brain = CortexGraph(config)
    legal_brain.ingest("GDPR Article 44 mandates encryption for international transfers.", sector="legal")
    legal_brain.ingest("Transferring unencrypted patient data violates data privacy.", sector="legal")

    ensemble_storage = tmp_path / "ensembles"
    ensemble = CortexEnsemble(storage_dir=ensemble_storage)
    ensemble.register_agent("medical_agent", med_brain, sector="medical")
    ensemble.register_agent("legal_agent", legal_brain, sector="legal")

    # 1. Query ensemble on medical question
    med_resp = ensemble.query_ensemble("What is the contraindication for Metformin?", target_sectors=["medical"])
    assert med_resp.winner_agent_id == "medical_agent"
    assert med_resp.winning_confidence > 0.0

    # 2. Save ensemble to disk
    ensemble.save_ensemble("clinical_compliance_v1")
    manifest_path = ensemble_storage / "clinical_compliance_v1" / "manifest.json"
    assert manifest_path.exists()

    # 3. Load ensemble back from disk
    loaded_ensemble = CortexEnsemble.load_ensemble(ensemble_storage / "clinical_compliance_v1")
    assert "medical_agent" in loaded_ensemble.agents
    assert "legal_agent" in loaded_ensemble.agents

    # Verify inference on loaded ensemble
    loaded_resp = loaded_ensemble.query_ensemble("What is the contraindication for Metformin?", target_sectors=["medical"])
    assert loaded_resp.winner_agent_id == "medical_agent"


# ==============================================================================
# 4. PILLAR 4: EmbodiedCortexAgent & ActionRegistry
# ==============================================================================

def test_pillar_4_embodied_agent(real_data_cortex, tmp_path):
    storage_dir = tmp_path / "embodied_storage"
    registry = ActionRegistry()

    # Add custom diagnostic action
    custom_executed = False
    def custom_logger(params):
        nonlocal custom_executed
        custom_executed = True
        return f"Logged telemetry event {params.get('event')}"

    registry.register_action("telemetry_logger", custom_logger)

    agent = EmbodiedCortexAgent(cortex=real_data_cortex, action_registry=registry, storage_dir=storage_dir)

    # 1. Perceive environment telemetry
    observation = {
        "engine_inlet_temperature": "1450C",
        "vibration_frequency": "120Hz",
    }
    percep = agent.perceive(observation, sector="aerospace_engineering")
    assert "1450C" in percep

    # 2. Execute Perceive-Reason-Act-Learn cycle
    episode = agent.execute_cycle(
        goal="Log engine telemetry and compute safety coefficient",
        observation=observation,
        required_actions=[
            ("telemetry_logger", {"event": "high_thermal_load"}),
            ("compute_math", {"expression": "1450 / 100"}),
        ],
        sector="aerospace_engineering",
    )

    assert episode.success is True
    assert custom_executed is True
    assert len(episode.actions_executed) == 2
    assert episode.actions_executed[1].output == "14.5"
    assert episode.learned_precedent_id is not None

    # 3. Check persistent storage
    assert agent.episodes_file.exists()
    history = agent.get_episode_history()
    assert len(history) >= 1
    assert history[0]["episode_id"] == episode.episode_id


# ==============================================================================
# 5. PILLAR 5: SelfImprovementEngine & ErrorMemory
# ==============================================================================

def test_pillar_5_self_improvement_and_error_memory(real_data_cortex, tmp_path):
    storage_dir = tmp_path / "self_improvement"
    err_memory = ErrorMemory(storage_dir / "errors")
    improver = SelfImprovementEngine(cortex=real_data_cortex, error_memory=err_memory, storage_dir=storage_dir)

    # 1. Evaluate high-quality response
    query = "Metformin and Type 2 Diabetes Mellitus"
    c_resp = real_data_cortex.query(query)

    score = improver.evaluate_proof(c_resp, query)
    assert isinstance(score, ProofQualityScore)
    assert score.overall_score > 0.0

    # 2. Trigger improvement adaptation
    action = improver.improve_from_proof(query, c_resp, sector="medical")
    assert isinstance(action, ImprovementAction)
    assert action.tier in ["TIER_1_EXCELLENT", "TIER_2_MODERATE", "TIER_3_DEFICIENT"]

    # 3. Test ErrorMemory recording and recall
    err_id = err_memory.record_error(
        query="Give Metformin to ESRD patient with creatinine 6.0",
        faulty_reasoning="Administer standard 1000mg Metformin",
        contradiction_type="contraindication",
        correction="Do not administer. Metformin accumulates causing fatal lactic acidosis.",
        sector="medical",
    )
    assert err_id is not None
    assert err_memory.stats()["total_errors_recorded"] >= 1

    # Check known error lookup
    detected_err = err_memory.check_for_known_error("Can I Give Metformin to ESRD patient with creatinine 6.0?")
    assert detected_err is not None
    assert "lactic acidosis" in detected_err.correction.lower()

    # 4. Check persistent storage
    assert improver.eval_file.exists()
    assert improver.actions_file.exists()
    assert err_memory.records_file.exists()
    progression = improver.get_quality_progression()
    assert progression["total_proofs_evaluated"] >= 1
