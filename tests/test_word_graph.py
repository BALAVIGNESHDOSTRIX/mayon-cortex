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
Tests for Word Graph & Graph-Native Generation (Pillar 1)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator, WordSelector, guess_pos


class TestWordGraph:
    def setup_method(self):
        self.config = CortexConfig()
        self.emotion = EmotionEngine()
        self.word_graph = WordGraph(self.config, emotion_engine=self.emotion)
        self.generator = WordGraphGenerator(self.word_graph, emotion_engine=self.emotion)

    def test_pos_tagger(self):
        assert guess_pos("running") == "VERB_GER"
        assert guess_pos("walked") == "VERB_PAST"
        assert guess_pos("quickly") == "ADV"
        assert guess_pos("beautiful") == "ADJ"
        assert guess_pos("the") == "DET"
        assert guess_pos("dog") == "NOUN"
        assert guess_pos(".") == "PUNCT"

    def test_ingest_sentence_tokens(self):
        tokens = ["the", "brave", "hero", "protects", "the", "peaceful", "kingdom", "."]
        self.word_graph.add_sentence_tokens(tokens=tokens, sector="story")

        assert "hero" in self.word_graph.nodes
        assert "kingdom" in self.word_graph.nodes
        assert self.word_graph.nodes["the"].frequency >= 2

        # Check sequential transition
        hero_node = self.word_graph.nodes["hero"]
        assert "protects" in hero_node.next_words

        # Check skip-gram transition
        assert "the" in hero_node.skip_1

    def test_12_feature_word_selector(self):
        selector = WordSelector()
        dim = 384
        thought_vec = np.random.randn(dim).astype(np.float32)
        thought_vec /= np.linalg.norm(thought_vec)

        self.word_graph.add_sentence_tokens(
            tokens=["sun", "shines", "brightly", "over", "earth", "."],
            sector="science"
        )
        cand_node = self.word_graph.nodes["sun"]

        feats = selector.extract_features(
            candidate_word="sun",
            candidate_node=cand_node,
            prev_word=None,
            prev_node=None,
            generated_so_far=[],
            thought_vec=thought_vec,
            proof_vectors=[thought_vec],
            context_vad=VAD(valence=0.5, arousal=0.5, dominance=0.5),
            target_sector="science",
            position=0,
            max_position=20,
        )

        assert feats.shape == (12,)
        score = selector.score(feats)
        assert isinstance(score, float)

    def test_beam_search_generation(self):
        # Ingest story knowledge
        sentences = [
            ["arthur", "is", "a", "legendary", "hero", "."],
            ["the", "hero", "defeats", "the", "dangerous", "dragon", "."],
            ["the", "dragon", "lived", "in", "the", "dark", "forest", "."],
            ["peace", "returned", "to", "the", "golden", "kingdom", "."],
        ]
        for s in sentences:
            self.word_graph.add_sentence_tokens(tokens=s, sector="story")

        thought_vec = self.word_graph.nodes["hero"].vector
        generated = self.generator.generate(
            thought_vector=thought_vec,
            max_tokens=15,
            beam_width=5,
            target_sector="story",
            seed_words=["arthur", "the"],
        )

        assert len(generated) > 0
        assert isinstance(generated, str)
