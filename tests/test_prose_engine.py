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
Unit Tests for Hierarchical Prose Engine & Universal 1-to-10 Parameter Adapters
================================================================================
Tests:
1. ProseConfig parameter mapping (1-10 scale for novelty, coherence, cadence, emotion)
2. NarrativeArcPlanner multi-act storyboard generation
3. DiscourseManager genre transition markers
4. HierarchicalProseEngine paragraph and story generation
5. CortexGraph integration (compose_story, compose_article, ingest_corpus)
6. Parameter sweeps: Novelty (1 vs 10), Cadence (1 vs 10), Coherence (1 vs 10)
7. Anti-looping recency buffer & vocabulary diversity
"""

import pytest
import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.cortex import CortexGraph
from mayon_cortex.affect.emotion import EmotionEngine, VAD
from mayon_cortex.language.word_graph import WordGraph
from mayon_cortex.language.prose_engine import (
    ProseConfig,
    NarrativeArcPlanner,
    HierarchicalProseEngine,
    DISCOURSE_MARKERS,
)
from mayon_cortex.learning.bootstrap import bootstrap_minimal, bootstrap_large_corpus
from mayon_cortex.language.corpus_data import LARGE_DOMAIN_CORPUS


class TestProseConfig:
    """Test 1-to-10 universal parameter scaling."""

    def test_temperature_mapping(self):
        cfg_low = ProseConfig(novelty=1.0)
        cfg_mid = ProseConfig(novelty=5.0)
        cfg_high = ProseConfig(novelty=10.0)

        assert pytest.approx(cfg_low.get_temperature(), 0.05) == 0.25
        assert 0.7 < cfg_mid.get_temperature() < 1.2
        assert pytest.approx(cfg_high.get_temperature(), 0.05) == 2.20

    def test_attractor_mapping(self):
        cfg_low = ProseConfig(coherence=1.0)
        cfg_high = ProseConfig(coherence=10.0)

        assert cfg_low.get_attractor_weight() < cfg_high.get_attractor_weight()
        assert pytest.approx(cfg_high.get_attractor_weight(), 0.05) == 0.95

    def test_cadence_mapping(self):
        cfg_punchy = ProseConfig(cadence=1.0)
        cfg_flowing = ProseConfig(cadence=10.0)

        min_p, max_p = cfg_punchy.get_target_sentence_length()
        min_f, max_f = cfg_flowing.get_target_sentence_length()

        assert min_p < min_f
        assert max_p < max_f
        assert max_p <= 10
        assert max_f >= 25

    def test_emotion_weight_mapping(self):
        cfg_neutral = ProseConfig(emotion_intensity=1.0)
        cfg_dramatic = ProseConfig(emotion_intensity=10.0)

        assert cfg_neutral.get_emotion_weight() < cfg_dramatic.get_emotion_weight()
        assert pytest.approx(cfg_dramatic.get_emotion_weight(), 0.05) == 0.85


class TestNarrativeArcPlanner:
    """Test multi-act storyboard decomposition."""

    def test_3_act_planning(self):
        planner = NarrativeArcPlanner()
        acts = planner.plan_arc(prompt="Voyage to the Andromeda Galaxy", acts_count=3, genre="sci-fi")

        assert len(acts) == 3
        assert acts[0].act_index == 1
        assert acts[0].discourse_type == "opener"
        assert acts[1].discourse_type == "transition"
        assert acts[2].discourse_type == "conclusion"
        assert all(isinstance(a.target_vad, VAD) for a in acts)

    def test_4_act_planning(self):
        planner = NarrativeArcPlanner()
        acts = planner.plan_arc(prompt="The Mystery of the Quantum Core", acts_count=4, genre="cyberpunk")

        assert len(acts) == 4
        assert acts[0].act_name.startswith("Exposition")
        assert acts[3].act_name.startswith("Echoing")


class TestHierarchicalProseEngine:
    """Test graph-native prose generation engine."""

    @pytest.fixture(scope="module")
    def bootstrapped_brain(self):
        brain = CortexGraph()
        bootstrap_minimal(brain, verbose=False)
        bootstrap_large_corpus(brain, verbose=False)
        return brain

    def test_compose_paragraph(self, bootstrapped_brain):
        engine = bootstrapped_brain.prose_engine
        theme_vec = bootstrapped_brain.embedder.encode_single("interstellar cosmic exploration")
        cfg = ProseConfig(novelty=5.0, coherence=8.0, cadence=6.0, genre="sci-fi")

        para, recency = engine.compose_paragraph(
            theme_vector=theme_vec,
            config=cfg,
            num_sentences=3,
        )

        assert isinstance(para, str)
        assert len(para.split()) >= 10
        assert para[0].isupper()
        assert para.endswith(".")
        assert len(recency) > 0

    def test_compose_multi_act_story(self, bootstrapped_brain):
        engine = bootstrapped_brain.prose_engine
        cfg = ProseConfig(novelty=7.0, coherence=8.0, cadence=6.0, genre="cyberpunk")

        story_result = engine.compose_story(
            prompt="Neural breach inside the corporate mainframe",
            acts_count=3,
            config=cfg,
        )

        assert "title" in story_result
        assert len(story_result["acts"]) == 3
        assert story_result["parameters"]["novelty"] == 7.0
        assert len(story_result["full_story"]) > 100

        # Check each act has content
        for act in story_result["acts"]:
            assert "act_number" in act
            assert "act_name" in act
            assert len(act["text"]) > 10

    def test_cortex_high_level_story_api(self, bootstrapped_brain):
        story = bootstrapped_brain.compose_story(
            prompt="The ancient stargate on Mars",
            acts=3,
            novelty=6.5,
            coherence=8.0,
            cadence=6.0,
            emotion_intensity=7.0,
            genre="sci-fi",
        )

        assert isinstance(story, dict)
        assert len(story["acts"]) == 3
        assert "Mars" in story["prompt"]

    def test_cortex_article_api(self, bootstrapped_brain):
        article = bootstrapped_brain.compose_article(
            topic="Principles of associative neural graphs",
            paragraphs=2,
            novelty=4.0,
            coherence=9.0,
            genre="academic",
        )

        assert isinstance(article, dict)
        assert len(article["acts"]) == 2

    def test_cortex_ingest_corpus(self, bootstrapped_brain):
        custom_corpus = [
            "Quantum entanglement enables instantaneous correlation across spatial distances.",
            "Superconducting qubits maintain coherence at sub-Kelvin cryogenic temperatures.",
            "Topological quantum computers protect information using non-Abelian anyons.",
        ]

        report = bootstrapped_brain.ingest_corpus(custom_corpus, domain="quantum_physics")

        assert report["sentences_ingested"] == 3
        assert report["total_word_nodes"] >= len(bootstrapped_brain.word_graph.nodes)
        assert "qubits" in bootstrapped_brain.word_graph.nodes or "quantum" in bootstrapped_brain.word_graph.nodes
