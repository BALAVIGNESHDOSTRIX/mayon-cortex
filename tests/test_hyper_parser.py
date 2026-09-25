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
Unit Tests for Hyper-Relational Semantic Role & Math Parser
"""

import pytest
from mayon_cortex.language.hyper_parser import (
    SemanticRoleExtractor,
    HyperEdge,
    Modality,
    MathConstraint,
)


class TestHyperParser:
    def setup_method(self):
        self.extractor = SemanticRoleExtractor()

    def test_parse_simple_relation(self):
        text = "Lisinopril treats hypertension."
        edges = self.extractor.parse_sentence(text, sector="medical")
        assert len(edges) >= 1
        edge = edges[0]
        assert "lisinopril" in edge.source.lower()
        assert edge.relation == "treats"
        assert "hypertension" in edge.target.lower()
        assert edge.modality == Modality.FACTUAL

    def test_parse_condition_and_math_inequality(self):
        text = "If clearance < 30 ml/min then Metformin contraindicates severe renal failure."
        edges = self.extractor.parse_sentence(text, sector="medical")
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.condition is not None
        assert "clearance < 30" in edge.condition
        assert len(edge.math_constraints) >= 1
        mc = edge.math_constraints[0]
        assert mc.variable == "clearance"
        assert mc.operator == "<"
        assert mc.value == 30.0
        assert edge.modality == Modality.CONDITIONAL

        # Test evaluation
        valid, msg = edge.is_valid_under({"clearance": 20.0})
        assert valid
        valid_false, msg_f = edge.is_valid_under({"clearance": 50.0})
        assert not valid_false

    def test_parse_modality_negation(self):
        text = "Aspirin is strictly prohibited with active bleeding ulcers."
        edges = self.extractor.parse_sentence(text, sector="medical")
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.modality == Modality.NEGATED
        valid, _ = edge.is_valid_under({})
        assert not valid

    def test_parse_physics_and_equality(self):
        text = "Force equals mass multiplied by acceleration."
        edges = self.extractor.parse_sentence(text, sector="physics")
        assert len(edges) >= 1
        assert edges[0].relation == "equals"
