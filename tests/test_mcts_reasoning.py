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
Unit Tests for Graph-MCTS Multi-Hop Proof Engine
"""

import pytest
from mayon_cortex.engine import MayonCortex
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.reasoning.mcts_reasoner import GraphMCTSEngine, MCTSProofResult


class TestGraphMCTS:
    def setup_method(self):
        self.cortex = MayonCortex(CortexConfig())
        # Construct multi-hop causal chain:
        # A -> B -> C -> D
        # Lisinopril -> ACE_Inhibition -> Vasodilation -> Lowers_BP -> Reduces_Stroke_Risk
        self.cortex.ingest("Lisinopril causes ACE inhibition.", sector="medical")
        self.cortex.ingest("ACE inhibition induces systemic vasodilation.", sector="medical")
        self.cortex.ingest("Systemic vasodilation lowers blood pressure.", sector="medical")
        self.cortex.ingest("Lowers blood pressure prevents severe strokes.", sector="medical")

    def test_multi_hop_proof_discovery(self):
        # Prove from Lisinopril to Stroke
        proof = self.cortex.prove(start_concept="lisinopril", goal_concept="strokes")
        assert isinstance(proof, MCTSProofResult)
        assert proof.is_proven
        assert len(proof.path_nodes) >= 3
        assert len(proof.steps) >= 2
        assert proof.total_proof_score > 0.0
        assert len(proof.human_explanation) > 0

    def test_unknown_concept_epistemic_gap(self):
        proof = self.cortex.prove(start_concept="UnobtainiumXYZ", goal_concept="strokes")
        assert not proof.is_proven
        assert len(proof.epistemic_gaps) >= 1
