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
Cortex-Graph Intelligence Tests
=================================
Tests for the higher-order intelligence systems:
- Cross-domain reasoning
- Decision planning (A→Z ordered chains)
- Decision priority ordering
- Teacher curriculum learning
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex.core.config import CortexConfig, DecisionPriority, RELATION_PRIORITY_MAP
from mayon_cortex.executive.decision_planner import DecisionPlanner, DecisionFactor, DecisionChain
from mayon_cortex.reasoning.cross_domain import CrossDomainEvidence, DomainEvidence, CrossDomainResolver
from mayon_cortex.dynamics.activation_wave import ActivatedPath
from mayon_cortex.dynamics.salience import SalienceWeighter


# ── Decision Priority Tests ──

class TestDecisionPriority:
    def test_safety_is_highest(self):
        """Safety must always be the highest priority."""
        assert DecisionPriority.SAFETY > DecisionPriority.LEGAL
        assert DecisionPriority.SAFETY > DecisionPriority.CLINICAL
        assert DecisionPriority.SAFETY > DecisionPriority.FINANCIAL
        assert DecisionPriority.SAFETY == 100

    def test_priority_ordering(self):
        """Full priority hierarchy must be correct."""
        assert DecisionPriority.SAFETY > DecisionPriority.LEGAL > DecisionPriority.CLINICAL
        assert DecisionPriority.CLINICAL > DecisionPriority.FINANCIAL > DecisionPriority.PREFERENCE
        assert DecisionPriority.MONITORING > DecisionPriority.DOCUMENTATION

    def test_relation_priority_mapping(self):
        """Safety-critical relations must map to safety priority."""
        assert RELATION_PRIORITY_MAP["contraindicates"] == DecisionPriority.CONTRAINDICATION
        assert RELATION_PRIORITY_MAP["causes"] == DecisionPriority.SAFETY
        assert RELATION_PRIORITY_MAP["treats"] == DecisionPriority.CLINICAL
        assert RELATION_PRIORITY_MAP["costs"] == DecisionPriority.FINANCIAL


# ── Decision Planner Tests ──

class TestDecisionPlanner:
    def setup_method(self):
        self.planner = DecisionPlanner()

    def test_topological_sort_respects_priority(self):
        """Higher-priority factors must come before lower-priority ones."""
        factors = [
            DecisionFactor(
                text="drug costs $10", action="CHECK COST",
                sector="finance", priority=DecisionPriority.FINANCIAL,
                confidence=0.9, evidence_edges=[],
            ),
            DecisionFactor(
                text="drug interaction risk", action="CHECK INTERACTION",
                sector="medical", priority=DecisionPriority.SAFETY,
                confidence=0.95, evidence_edges=[],
            ),
            DecisionFactor(
                text="drug treats condition", action="PRESCRIBE",
                sector="medical", priority=DecisionPriority.CLINICAL,
                confidence=0.85, evidence_edges=[],
            ),
        ]

        deps = {i: [] for i in range(len(factors))}
        ordered = self.planner._topological_priority_sort(factors, deps)

        # Safety should be first
        assert ordered[0].priority == DecisionPriority.SAFETY
        # Financial should be last
        assert ordered[-1].priority == DecisionPriority.FINANCIAL

    def test_dependencies_respected(self):
        """Dependencies must be satisfied before dependent steps."""
        safety = DecisionFactor(
            text="check safety", action="CHECK",
            sector="medical", priority=DecisionPriority.SAFETY,
            confidence=0.9, evidence_edges=[],
        )
        clinical = DecisionFactor(
            text="prescribe", action="PRESCRIBE",
            sector="medical", priority=DecisionPriority.CLINICAL,
            confidence=0.8, evidence_edges=[],
        )

        factors = [clinical, safety]  # clinical first in list
        deps = {0: [1], 1: []}  # clinical depends on safety
        ordered = self.planner._topological_priority_sort(factors, deps)

        # Safety must come before clinical even though clinical was listed first
        safety_idx = ordered.index(safety)
        clinical_idx = ordered.index(clinical)
        assert safety_idx < clinical_idx

    def test_empty_evidence_produces_empty_chain(self):
        """Empty evidence should produce empty decision chain."""
        evidence = CrossDomainEvidence(
            sector_evidence={}, all_paths=[], conflicts=[],
            resolved_paths=[], relevant_sectors=["general"],
            primary_sector="general",
        )
        chain = self.planner.plan(evidence)
        assert chain.num_steps == 0

    def test_plan_from_paths(self):
        """A path with safety relation should produce a SAFETY step first."""
        path = ActivatedPath(
            node_ids=["drug", "patient"],
            edges=[("drug", "contraindicates", "patient")],
            confidence=0.9, relevance=0.8, hop_count=1,
        )
        evidence = CrossDomainEvidence(
            sector_evidence={"medical": DomainEvidence(
                "medical", [path], 0.9, 2, ["contraindicates"],
            )},
            all_paths=[path], conflicts=[],
            resolved_paths=[path],
            relevant_sectors=["medical"],
            primary_sector="medical",
        )
        chain = self.planner.plan(evidence)
        assert chain.num_steps > 0
        # First step should be safety-related
        assert chain.steps[0].priority >= DecisionPriority.CONTRAINDICATION


# ── Cross-Domain Tests ──

class TestCrossDomain:
    def test_conflict_detection(self):
        """Opposing relations from different sectors = conflict."""
        resolver = CrossDomainResolver()

        # Medical says: treats
        path_med = ActivatedPath(
            node_ids=["drug", "disease"],
            edges=[("drug", "treats", "disease")],
            confidence=0.9, relevance=0.8, hop_count=1,
        )
        # Legal says: contraindicates (hypothetical)
        path_legal = ActivatedPath(
            node_ids=["drug", "disease"],
            edges=[("drug", "contraindicates", "disease")],
            confidence=0.7, relevance=0.6, hop_count=1,
        )

        sector_evidence = {
            "medical": DomainEvidence("medical", [path_med], 0.9, 2, ["treats"]),
            "legal": DomainEvidence("legal", [path_legal], 0.7, 2, ["contraindicates"]),
        }

        conflicts = resolver._detect_cross_domain_conflicts(sector_evidence, None)
        assert len(conflicts) >= 1
        # Legal should win (higher domain priority for safety)

    def test_single_sector_no_cross_domain(self):
        """Single sector queries shouldn't trigger cross-domain resolution."""
        config = CortexConfig(cross_domain_min_sectors=2)
        resolver = CrossDomainResolver(config)
        # The resolve method needs a full graph/ndu/wave — tested via integration


# ── Salience Integration Tests ──

class TestSalienceIntegration:
    def test_path_salience_dominated_by_safety(self):
        """A path with ONE safety edge should have high salience."""
        weighter = SalienceWeighter()

        # Path with all routine edges
        routine_salience = weighter.compute_path_salience(
            ["is-a", "part-of", "has-property"],
            [0.9, 0.9, 0.9],
        )

        # Path with one safety edge
        safety_salience = weighter.compute_path_salience(
            ["is-a", "contraindicates", "has-property"],
            [0.9, 0.9, 0.9],
        )

        # Safety path must score much higher
        assert safety_salience > routine_salience * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
