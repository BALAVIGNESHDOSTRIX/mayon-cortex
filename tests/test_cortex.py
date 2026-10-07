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
Cortex-Graph Core Tests
========================
Tests for the fundamental components: NDU, activation wave,
path ranker, language ingestion, sentence assembly, and the
full brain pipeline.
"""

import time
import numpy as np
import pytest
import sys
import os

# Add parent packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.core.ndu import NodeDecisionUnit, NDUDecision
from mayon_cortex.dynamics.salience import SalienceWeighter
from mayon_cortex.executive.uncertainty import UncertaintyGate
from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.executive.self_correction import SelfCorrector


# ── NDU Tests ──

class TestNDU:
    def setup_method(self):
        self.config = CortexConfig()
        self.ndu = NodeDecisionUnit(self.config)

    def test_extract_features_shape(self):
        """Feature vector must be exactly 52 dimensions."""
        query_vec = np.random.randn(384).astype(np.float32)
        node_vec = np.random.randn(384).astype(np.float32)

        features = self.ndu.extract_features(
            query_vec=query_vec, node_vec=node_vec,
            edge_confidence=0.9, relation_type="treats",
            node_level=1, node_sector="medical", query_sector="medical",
            access_count=5, hop_depth=1, max_hops=5,
            out_degree=3, max_degree=10,
            node_timestamp=time.time() - 3600, current_time=time.time(),
            path_confidence=0.8, is_phrase_node=False,
            neighbor_relevance=0.6, is_visited=False, is_dead_end=False,
        )

        assert features.shape == (52,)
        assert features.dtype == np.float32

    def test_heuristic_prediction(self):
        """Untrained NDU uses heuristic fallback."""
        assert not self.ndu._trained

        # High similarity + high confidence → should follow
        features = np.zeros(52, dtype=np.float32)
        features[0] = 0.9   # high cosine similarity
        features[1] = 0.95  # high edge confidence
        features[46] = 0.9  # high path confidence

        decision = self.ndu.predict(features)
        assert isinstance(decision, NDUDecision)
        assert decision.follow_prob > 0.5

    def test_visited_node_backtracks(self):
        """Visited nodes should trigger backtrack."""
        features = np.zeros(52, dtype=np.float32)
        features[0] = 0.9
        features[1] = 0.9
        features[46] = 0.9
        features[49] = 1.0  # is_visited = True

        decision = self.ndu.predict(features)
        assert decision.backtrack_prob > 0.5
        assert decision.follow_prob < 0.2

    def test_record_and_buffer(self):
        """Recording outcomes should accumulate in buffer."""
        features = np.random.randn(52).astype(np.float32)
        self.ndu.record_outcome(features, True)
        self.ndu.record_outcome(features, False)
        assert len(self.ndu._training_buffer_X) == 2
        assert len(self.ndu._training_buffer_y) == 2


# ── Salience Tests ──

class TestSalience:
    def test_safety_edges_higher(self):
        """Safety-critical edges must have higher salience."""
        weighter = SalienceWeighter()
        safe = weighter.compute("contraindicates", 0.8)
        routine = weighter.compute("is-a", 0.8)
        assert safe > routine

    def test_is_safety_critical(self):
        weighter = SalienceWeighter()
        assert weighter.is_safety_critical("contraindicates")
        assert not weighter.is_safety_critical("is-a")

    def test_urgency_labels(self):
        weighter = SalienceWeighter()
        assert weighter.get_urgency_label(9.0) == "CRITICAL"
        assert weighter.get_urgency_label(0.5) == "ROUTINE"


# ── Uncertainty Tests ──

class TestUncertainty:
    def test_no_paths_is_uncertain(self):
        """No paths = uncertain."""
        gate = UncertaintyGate()
        assessment = gate.evaluate([], np.zeros(384), None)
        assert assessment.is_uncertain
        assert assessment.confidence == 0.0

    def test_format_response(self):
        gate = UncertaintyGate()
        assessment = gate.evaluate([], np.zeros(384), None)
        text = gate.format_uncertain_response(assessment)
        assert "don't have" in text.lower() or "evidence" in text.lower()


# ── Working Memory Tests ──

class TestWorkingMemory:
    def test_push_and_context(self):
        wm = WorkingMemory()
        vec = np.random.randn(384).astype(np.float32)
        wm.push("test question", vec, "test answer")
        assert wm.size == 1

        ctx = wm.get_context_vector()
        assert ctx is not None
        assert ctx.shape == (384,)

    def test_augment_query(self):
        wm = WorkingMemory()
        vec = np.random.randn(384).astype(np.float32)
        wm.push("test", vec, "answer")

        new_vec = np.random.randn(384).astype(np.float32)
        augmented = wm.augment_query(new_vec)
        assert augmented.shape == (384,)
        # Augmented should be different from original
        assert not np.allclose(new_vec, augmented)

    def test_max_items_eviction(self):
        config = CortexConfig(wm_max_items=3)
        wm = WorkingMemory(config)
        for i in range(5):
            wm.push(f"q{i}", np.random.randn(384).astype(np.float32), f"a{i}")
        assert wm.size == 3  # oldest 2 evicted

    def test_clear(self):
        wm = WorkingMemory()
        wm.push("q", np.zeros(384), "a")
        wm.clear()
        assert wm.is_empty


# ── Self-Correction Tests ──

class TestSelfCorrection:
    def test_no_contradiction_clean(self):
        corrector = SelfCorrector()
        # Fake paths with no conflicts
        from mayon_cortex.dynamics.activation_wave import ActivatedPath
        p1 = ActivatedPath(
            node_ids=["a", "b"], edges=[("a", "treats", "b")],
            confidence=0.9, relevance=0.8, hop_count=1,
        )
        p2 = ActivatedPath(
            node_ids=["c", "d"], edges=[("c", "causes", "d")],
            confidence=0.8, relevance=0.7, hop_count=1,
        )
        clean, contradictions = corrector.correct([p1, p2])
        assert len(clean) == 2
        assert len(contradictions) == 0

    def test_opposing_relations_detected(self):
        corrector = SelfCorrector()
        from mayon_cortex.dynamics.activation_wave import ActivatedPath
        p1 = ActivatedPath(
            node_ids=["drug", "disease"],
            edges=[("drug", "treats", "disease")],
            confidence=0.9, relevance=0.8, hop_count=1,
        )
        p2 = ActivatedPath(
            node_ids=["drug", "disease"],
            edges=[("drug", "contraindicates", "disease")],
            confidence=0.7, relevance=0.6, hop_count=1,
        )
        clean, contradictions = corrector.correct([p1, p2])
        assert len(contradictions) == 1
        # Safety (contraindicates) should win
        assert len(clean) == 1
        assert clean[0].edges[0][1] == "contraindicates"


from mayon_cortex.cortex import CortexGraph, CortexResponse, DecisionResponse
from mayon_cortex.learning.bootstrap import bootstrap_minimal
import tempfile
import shutil


# ── Full Cortex End-to-End Tests ──

class TestCortexEndToEnd:
    def test_cortex_init_and_stats(self):
        brain = CortexGraph()
        stats = brain.stats()
        assert stats["graph_nodes"] == 0
        assert stats["queries_processed"] == 0

    def test_cortex_ingest_and_query(self):
        brain = CortexGraph()
        # Ingest facts
        brain.ingest("Lisinopril is used to treat hypertension.", sector="medical")
        brain.ingest("Hypertension causes increased risk of stroke.", sector="medical")
        
        assert brain.graph.num_nodes > 0
        
        # Query the brain
        resp = brain.query("What treats hypertension?")
        assert isinstance(resp, CortexResponse)
        assert resp.text is not None
        assert len(resp.text) > 0
        assert brain.stats()["queries_processed"] == 1

    def test_cortex_decide(self):
        brain = CortexGraph()
        brain.ingest("Warfarin is an anticoagulant that interacts with aspirin causing severe bleeding.", sector="medical")
        brain.ingest("Aspirin is prohibited when active bleeding risk is high.", sector="legal")
        
        decision = brain.decide("Should a patient taking aspirin be prescribed warfarin?")
        assert isinstance(decision, DecisionResponse)
        assert decision.text is not None

    def test_cortex_teach_and_assess(self):
        brain = CortexGraph()
        # Teach infant phase
        brain.teach(phase=1)
        assert brain.graph.num_nodes > 0
        
        # Assess brain
        assessment = brain.assess()
        assert "overall_accuracy" in assessment
        assert "sector_accuracy" in assessment
        assert "recommendations" in assessment

    def test_cortex_save_and_load(self):
        brain = CortexGraph()
        brain.ingest("Water boils at 100 degrees Celsius.", sector="science")
        nodes_before = brain.graph.num_nodes
        
        temp_dir = tempfile.mkdtemp()
        try:
            brain.save(temp_dir)
            loaded_brain = CortexGraph.load(temp_dir)
            assert loaded_brain.graph.num_nodes == nodes_before
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_broca_decoder_articulation(self):
        from mayon_cortex.language.broca import BrocaDecoder, BrocaConfig
        decoder = BrocaDecoder(BrocaConfig(hidden_dim=128, num_layers=2, num_heads=2, ffn_dim=256))
        
        # Verify micro footprint
        size_mb = decoder.param_size_mb()
        assert size_mb < 15.0  # Micro parameter footprint
        
        # Test generation conditioned on graph vectors
        graph_vecs = np.random.randn(3, 384).astype(np.float32)
        out = decoder.generate_from_graph(graph_vecs, prompt_text="Hypertension treatment", max_tokens=10)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_cortex_articulate_broca(self):
        brain = CortexGraph()
        brain.ingest("Lisinopril is an ACE inhibitor that treats hypertension.", sector="medical")
        out = brain.articulate_broca("What treats hypertension?")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_xgboost_ndu_and_ranker_training(self):
        config = CortexConfig()
        ndu = NodeDecisionUnit(config)
        # Buffer positive and negative decisions
        for _ in range(25):
            ndu.record_outcome(np.random.randn(52).astype(np.float32), True)
            ndu.record_outcome(np.random.randn(52).astype(np.float32), False)
        
        trained = ndu.train(force=True)
        assert trained
        assert ndu._trained

        # Path ranker training
        from mayon_cortex.reasoning.path_ranker import PathRanker
        ranker = PathRanker(config)
        for i in range(35):
            ranker.record_outcome(np.random.randn(15).astype(np.float32), reward=float(i % 2))
        
        r_trained = ranker.train(force=True)
        assert r_trained
        assert ranker._trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


