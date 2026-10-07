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
Tests for Pillar 7: ImagineEngine (ImagineNet)
==============================================
Verifies mental sandboxing, counterfactual simulation, conceptual blending,
dream-state consolidation with synaptic pruning, and curiosity exploration.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph
from mayon_cortex.imagination.imagination import (
    MentalSandbox,
    ConceptBlender,
    DreamConsolidator,
    CuriosityEngine,
    ImagineEngine,
)


@pytest.fixture
def brain():
    b = CortexGraph()
    # Ingest a few facts for the tests
    b.ingest("Lisinopril treats hypertension and lowers high blood pressure.", sector="medical")
    b.ingest("Aspirin reduces blood clotting and prevents strokes.", sector="medical")
    b.ingest("Mycelium network routes nutrients across the forest ecosystem.", sector="science")
    b.ingest("Internet routers route data packets across the network.", sector="code")
    return b


def test_mental_sandbox_isolation(brain):
    """Test that actions in MentalSandbox do not mutate the main graph."""
    initial_node_count = brain.graph.num_nodes
    initial_edge_count = brain.graph.num_edges

    sandbox = MentalSandbox(brain.graph, config=brain.config)
    assert sandbox.sandbox_graph.num_nodes == initial_node_count

    # Inject fact into sandbox only
    vec = np.random.randn(brain.config.concept_dim).astype(np.float32)
    hypo_id = sandbox.inject_fact(
        source_text="Low gravity increases plant height",
        vector=vec,
        sector="science",
        confidence=0.9,
    )

    assert hypo_id in sandbox.sandbox_graph.nodes
    assert hypo_id not in brain.graph.nodes
    assert brain.graph.num_nodes == initial_node_count
    assert brain.graph.num_edges == initial_edge_count

    # Test committing verified nodes
    added = sandbox.commit_to_graph(brain.graph, [hypo_id])
    assert added == 1
    assert hypo_id in brain.graph.nodes
    assert brain.graph.num_nodes == initial_node_count + 1


def test_imagine_what_if(brain):
    """Test counterfactual what-if simulation through CortexGraph API."""
    res = brain.imagine_what_if(
        hypothetical_fact="Lisinopril is combined with drug X to enhance vasodilation",
        query="What lowers blood pressure?",
        sector="medical",
    )

    assert res is not None
    assert "Lisinopril" in res.scenario_name
    assert len(res.hypothetical_facts) == 1
    assert res.target_query == "What lowers blood pressure?"
    assert isinstance(res.summary, str)
    assert len(res.summary) > 0


def test_concept_blending(brain):
    """Test blending two distinct concepts across domains."""
    blend = brain.blend_concepts("hypertension", "aspirin", alpha=0.5)

    assert blend is not None
    assert blend.concept_id.startswith("blend_")
    assert "hypertension" in blend.name or "aspirin" in blend.name
    assert blend.vector.shape == (brain.config.concept_dim,)
    assert isinstance(blend.blended_properties, list)
    assert blend.confidence > 0.0
    assert len(blend.explanation) > 0


def test_dream_consolidation_and_pruning(brain):
    """Test offline dream state replay, hypothesis generation, and edge pruning."""
    # Seed working memory with a query
    brain.query("What treats hypertension?")

    # Run dream consolidation cycle
    stats = brain.dream(cycles=5)

    assert stats is not None
    assert stats.replays_conducted >= 1
    assert isinstance(stats.hypotheses_generated, int)
    assert isinstance(stats.edges_pruned, int)


def test_curiosity_exploration(brain):
    """Test detection of curiosity gaps / isolated concepts in the graph."""
    gaps = brain.explore_curiosity(top_k=3)

    assert isinstance(gaps, list)
    for gap in gaps:
        assert gap.node_id in brain.graph.nodes
        assert len(gap.suggested_query) > 0
        assert gap.isolation_score > 0.0
