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
Unit & Integration Tests for Mayon-Cortex 5 Priority LLM Integration Layers
===========================================================================
Tests:
  1. Priority 1: cortex_think (ActivationWave, PathRanker, Math Proofs)
  2. Priority 2: cortex_recall & cortex_remember (CaseMemory, Precedent Matching)
  3. Priority 3: cortex_decide (DecisionPlanner, Cross-Domain Conflict Resolution)
  4. Priority 4: cortex_validate (Post-Generation Hallucination & Consistency Filter)
  5. Priority 5: cortex_plan (Executive Controller, Complexity Routing)
  6. Orchestrator: CortexLLMBridge (Full 5-Layer Cognitive Loop)
  7. MCP Server: MayonCortexMCPServer Tool Dispatch
"""

import pytest
import numpy as np

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
from mayon_cortex.integration.cortex_mcp_server import MayonCortexMCPServer


@pytest.fixture(scope="module")
def cortex_fixture():
    """Create and populate a test CortexGraph instance."""
    config = CortexConfig()
    brain = CortexGraph(config)

    # Populate sample medical & domain knowledge
    # 1. Metformin & Diabetes
    n_metformin = brain.graph.add_node(
        vector=brain.embedder.encode_single("Metformin is a first-line medication for type 2 diabetes."),
        source_text="Metformin is a first-line medication for type 2 diabetes.",
        level=0,
        sector="medical",
    )
    n_diabetes = brain.graph.add_node(
        vector=brain.embedder.encode_single("Type 2 Diabetes mellitus is a metabolic disorder."),
        source_text="Type 2 Diabetes mellitus is a metabolic disorder.",
        level=0,
        sector="medical",
    )
    n_renal = brain.graph.add_node(
        vector=brain.embedder.encode_single("Severe Renal Failure or high creatinine contraindicates Metformin."),
        source_text="Severe Renal Failure or high creatinine contraindicates Metformin.",
        level=0,
        sector="medical",
    )
    n_lactic = brain.graph.add_node(
        vector=brain.embedder.encode_single("Lactic Acidosis is a rare, severe metabolic complication."),
        source_text="Lactic Acidosis is a rare, severe metabolic complication.",
        level=0,
        sector="medical",
    )

    # Edges
    brain.graph.add_edge(n_metformin.id, n_diabetes.id, relation_type="treats", confidence=0.98, sector="medical")
    brain.graph.add_edge(n_metformin.id, n_renal.id, relation_type="contraindicated_for", confidence=0.99, sector="medical")
    brain.graph.add_edge(n_renal.id, n_lactic.id, relation_type="causes_risk_of", confidence=0.95, sector="medical")

    # Populate case memory precedent
    brain.case_memory.store_case(
        situation="Patient with Diabetes Type 2 and moderate CKD stage 3",
        judgment="Switch from Metformin to Linagliptin (DPP-4 inhibitor)",
        reasoning="Metformin poses high risk of lactic acidosis in kidney dysfunction",
        graph=brain.graph,
        embedder=brain.embedder,
        sector="medical",
        factors=["diabetes", "renal_impairment", "high_creatinine"],
        factor_weights={"diabetes": 0.8, "renal_impairment": 0.95},
        outcome="Blood glucose stabilized without acidosis",
    )

    return brain


# ==============================================================================
# TEST PRIORITY 1: cortex_think
# ==============================================================================

def test_priority_1_cortex_think(cortex_fixture):
    brain = cortex_fixture
    
    # 1. Multi-hop medical query
    res = cortex_think(brain, query="Metformin diabetes renal failure", max_hops=3)
    assert res["success"] is True
    assert "formatted_context" in res
    assert "Mayon-Cortex Symbolic Reasoning Context" in res["formatted_context"]
    assert res["confidence"] > 0.0

    # 2. Math offloading
    math_res = cortex_think(brain, query="Solve equation: 2x + 4 = 14", allow_math=True)
    assert math_res["success"] is True
    assert math_res["math_proof"] is not None
    assert "5" in str(math_res["math_proof"]["result"]) or "x" in str(math_res["math_proof"]["result"])


# ==============================================================================
# TEST PRIORITY 2: cortex_recall & cortex_remember
# ==============================================================================

def test_priority_2_cortex_recall_and_remember(cortex_fixture):
    brain = cortex_fixture

    # 1. Recall existing precedent
    recall_res = cortex_recall(brain, situation="Elderly patient with diabetes and chronic kidney disease", top_k=2)
    assert recall_res["success"] is True
    assert len(recall_res["precedents"]) >= 1
    top_p = recall_res["precedents"][0]
    assert "Linagliptin" in top_p["judgment"] or "Metformin" in top_p["situation"]
    assert "formatted_context" in recall_res

    # 2. Store new precedent
    case_id = cortex_remember(
        brain=brain,
        situation="Patient with acute anaphylaxis to penicillin",
        judgment="Administer Epinephrine 0.3mg IM immediately and switch to Azithromycin",
        reasoning="Penicillin cross-reactivity causes airway compromise",
        sector="medical",
        factors=["anaphylaxis", "allergy", "infection"],
    )
    assert case_id is not None
    assert case_id in brain.case_memory.cases

    # 3. Recall the newly stored precedent
    new_recall = cortex_recall(brain, situation="Severe penicillin allergy with acute respiratory distress")
    assert any(p["case_id"] == case_id for p in new_recall["precedents"])


# ==============================================================================
# TEST PRIORITY 3: cortex_decide
# ==============================================================================

def test_priority_3_cortex_decide(cortex_fixture):
    brain = cortex_fixture

    dec_res = cortex_decide(brain, situation="Patient presents with type 2 diabetes and renal failure", sector="medical")
    assert dec_res["success"] is True
    assert "formatted_context" in dec_res
    assert isinstance(dec_res["steps"], list)
    assert isinstance(dec_res["conflicts"], list)


# ==============================================================================
# TEST PRIORITY 4: cortex_validate
# ==============================================================================

def test_priority_4_cortex_validate(cortex_fixture):
    brain = cortex_fixture

    # 1. Grounded assertion
    val_grounded = cortex_validate(
        brain,
        claim_or_decision="Metformin is a medication for type 2 diabetes.",
    )
    assert val_grounded["success"] is True
    assert val_grounded["is_safe"] is True
    assert val_grounded["verdict"] in ["VERIFIED", "UNCERTAIN"]

    # 2. Contradictory assertion (Metformin with Renal Failure)
    val_contra = cortex_validate(
        brain,
        claim_or_decision="Administer high dose Metformin to patient with severe renal failure",
    )
    assert val_contra["success"] is True
    assert len(val_contra["contradictions"]) >= 1 or val_contra["verdict"] in ["CONTRADICTION", "UNCERTAIN"]


# ==============================================================================
# TEST PRIORITY 5: cortex_plan
# ==============================================================================

def test_priority_5_cortex_plan(cortex_fixture):
    brain = cortex_fixture

    # Trivial query
    plan_triv = cortex_plan(brain, query="What is Metformin?")
    assert plan_triv["success"] is True
    assert plan_triv["complexity"] in ["trivial", "moderate"]

    # Complex query
    plan_comp = cortex_plan(
        brain,
        query="Should we prescribe Metformin or SGLT2 inhibitors given patient has severe renal failure and high risk lactic acidosis?",
    )
    assert plan_comp["success"] is True
    assert plan_comp["complexity"] in ["complex", "moderate", "multi_part"]
    assert len(plan_comp["recommended_tools"]) >= 1


# ==============================================================================
# TEST ORCHESTRATOR & MCP SERVER
# ==============================================================================

def test_cortex_llm_bridge_orchestrator(cortex_fixture):
    brain = cortex_fixture
    bridge = CortexLLMBridge(cortex=brain)

    # 1. Default deterministic execution
    res = bridge.enhanced_reason("How to treat type 2 diabetes in renal failure?")
    assert res.is_grounded is True
    assert res.plan is not None
    assert res.think is not None
    assert res.recall is not None
    assert res.execution_time_ms > 0

    # 2. Mock LLM function execution
    mock_llm_called = False
    def mock_llm(prompt: str) -> str:
        nonlocal mock_llm_called
        mock_llm_called = True
        assert "Mayon-Cortex" in prompt
        return "Based on graph evidence, avoid Metformin in renal impairment and use Linagliptin."

    res_mock = bridge.enhanced_reason("How to treat type 2 diabetes in renal failure?", llm_fn=mock_llm)
    assert mock_llm_called is True
    assert "Linagliptin" in res_mock.llm_response


def test_mcp_server_dispatch(cortex_fixture):
    brain = cortex_fixture
    server = MayonCortexMCPServer(brain=brain)

    # Test all 6 tools through the server dispatcher
    t_think = server.handle_tool_call("cortex_think", {"query": "Metformin"})
    assert t_think["success"] is True

    t_plan = server.handle_tool_call("cortex_plan", {"query": "What is Metformin?"})
    assert t_plan["success"] is True

    t_recall = server.handle_tool_call("cortex_recall", {"situation": "diabetes kidney disease"})
    assert t_recall["success"] is True

    t_decide = server.handle_tool_call("cortex_decide", {"situation": "diabetes"})
    assert t_decide["success"] is True

    t_validate = server.handle_tool_call("cortex_validate", {"claim_or_decision": "Metformin treats diabetes"})
    assert t_validate["success"] is True

    t_remember = server.handle_tool_call("cortex_remember", {
        "situation": "Test situation",
        "judgment": "Test judgment",
        "reasoning": "Test reasoning",
    })
    assert t_remember["success"] is True
