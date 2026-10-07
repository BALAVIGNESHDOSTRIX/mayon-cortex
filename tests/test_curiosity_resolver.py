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
Unit Tests for Autonomous Curiosity & Dynamic Gap Resolver
"""

import pytest
from mayon_cortex.engine import MayonCortex
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.executive.curiosity_resolver import CuriosityGapResolver


class TestCuriosityResolver:
    def setup_method(self):
        self.cortex = MayonCortex(CortexConfig())
        self.cortex.ingest("Sunlight provides solar photons.", sector="science")
        self.cortex.ingest("Photosynthesis produces oxygen.", sector="science")

    def test_dynamic_gap_resolution(self):
        # Missing link: solar photons -> chlorophyll activation -> photosynthesis
        available_corpus = [
            "Solar photons trigger chlorophyll activation in green plant leaves.",
            "Chlorophyll activation initiates photosynthesis in plant cells.",
        ]

        # Before resolving, no path between sunlight and oxygen
        proof_before = self.cortex.prove("sunlight", "oxygen")
        
        # Trigger dynamic gap resolver
        attempt = self.cortex.curiosity_resolver.resolve_gap(
            start_concept="solar photons",
            goal_concept="photosynthesis",
            graph=self.cortex.graph,
            available_corpus=available_corpus,
            sector="science",
        )

        assert attempt.is_resolved
        assert attempt.edges_added >= 1

        # Now re-evaluate proof
        proof_after = self.cortex.prove("sunlight", "oxygen")
        assert len(proof_after.path_nodes) >= 2 or proof_after.is_proven
