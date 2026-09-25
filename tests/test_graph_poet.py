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
Tests for GraphPoetEngine: 100% Graph-Native Poetry & Creative Generation (Zero Transformers).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex import CortexGraph
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.language.word_graph import (
    WordGraph,
    GraphPoetEngine,
    count_syllables,
    extract_rhyme_key,
    get_stress_pattern,
)


class TestGraphPoet:
    def setup_method(self):
        self.config = CortexConfig()
        self.emotion = EmotionEngine()
        self.word_graph = WordGraph(self.config, emotion_engine=self.emotion)
        self.poet = GraphPoetEngine(self.word_graph, emotion_engine=self.emotion)

    def test_syllable_counter(self):
        assert count_syllables("light") == 1
        assert count_syllables("night") == 1
        assert count_syllables("ocean") == 2
        assert count_syllables("silent") == 2
        assert count_syllables("eternity") == 4
        assert count_syllables("universe") == 3
        assert count_syllables("the") == 1
        assert count_syllables("beautiful") == 3

    def test_rhyme_key_extractor(self):
        assert extract_rhyme_key("light") == "AY_T"
        assert extract_rhyme_key("night") == "AY_T"
        assert extract_rhyme_key("bright") == "AY_T"
        assert extract_rhyme_key("deep") == "IY_P"
        assert extract_rhyme_key("sleep") == "IY_P"
        assert extract_rhyme_key("fire") == "AY_R"
        assert extract_rhyme_key("desire") == "AY_R"
        assert extract_rhyme_key("gold") == "OW_L_D"
        assert extract_rhyme_key("cold") == "OW_L_D"

    def test_rhyme_bucket_lookup(self):
        rhymes_light = self.word_graph.get_rhyming_words("light")
        assert len(rhymes_light) > 0
        assert "night" in rhymes_light or "bright" in rhymes_light

        rhymes_deep = self.word_graph.get_rhyming_words("deep")
        assert "sleep" in rhymes_deep or "keep" in rhymes_deep

    def test_find_poetic_rhyme_pair(self):
        dim = 384
        theme_vec = np.random.randn(dim).astype(np.float32)
        theme_vec /= np.linalg.norm(theme_vec)

        w1, w2 = self.word_graph.find_poetic_rhyme_pair(theme_vec)
        assert w1 != w2
        assert extract_rhyme_key(w1) == extract_rhyme_key(w2)

    def test_compose_line_with_syllable_budget(self):
        dim = 384
        theme_vec = np.random.randn(dim).astype(np.float32)
        theme_vec /= np.linalg.norm(theme_vec)

        line = self.poet.compose_line(
            theme_vector=theme_vec,
            target_syllables=8,
            end_word="night",
            temperature=0.4,
        )

        assert len(line) > 0
        assert line.lower().endswith("night")
        # Verify line starts capitalized
        assert line[0].isupper()

    def test_compose_poem_couplet_form(self):
        dim = 384
        theme_vec = np.random.randn(dim).astype(np.float32)
        theme_vec /= np.linalg.norm(theme_vec)

        poem = self.poet.compose_poem(
            theme_vector=theme_vec,
            form="couplet",
            mood="heroic",
            lines_count=4,
            meter="tetrameter",
            temperature=0.5,
        )

        lines = poem.strip().split("\n")
        assert len(lines) == 4

        # In couplets, line 0 and line 1 rhyme
        end_word_0 = lines[0].rstrip(",.").split()[-1].lower()
        end_word_1 = lines[1].rstrip(",.").split()[-1].lower()
        assert extract_rhyme_key(end_word_0) == extract_rhyme_key(end_word_1)

        # Line 2 and line 3 rhyme
        end_word_2 = lines[2].rstrip(",.").split()[-1].lower()
        end_word_3 = lines[3].rstrip(",.").split()[-1].lower()
        assert extract_rhyme_key(end_word_2) == extract_rhyme_key(end_word_3)

    def test_compose_poem_quatrain_form(self):
        dim = 384
        theme_vec = np.random.randn(dim).astype(np.float32)
        theme_vec /= np.linalg.norm(theme_vec)

        poem = self.poet.compose_poem(
            theme_vector=theme_vec,
            form="quatrain",
            mood="serene",
            lines_count=4,
            meter="tetrameter",
            temperature=0.5,
        )

        lines = poem.strip().split("\n")
        assert len(lines) == 4

        # ABAB rhyme scheme: line 0 and line 2 rhyme
        end_word_0 = lines[0].rstrip(",.").split()[-1].lower()
        end_word_2 = lines[2].rstrip(",.").split()[-1].lower()
        assert extract_rhyme_key(end_word_0) == extract_rhyme_key(end_word_2)

        # Line 1 and line 3 rhyme
        end_word_1 = lines[1].rstrip(",.").split()[-1].lower()
        end_word_3 = lines[3].rstrip(",.").split()[-1].lower()
        assert extract_rhyme_key(end_word_1) == extract_rhyme_key(end_word_3)

    def test_cortex_brain_integration(self):
        brain = CortexGraph()
        
        # Test direct poem composition
        poem = brain.compose_poem(
            theme="stars and eternity",
            form="couplet",
            mood="epic",
            lines=4,
        )
        assert isinstance(poem, str)
        assert len(poem.split("\n")) == 4

        # Test prose composition
        prose = brain.compose_prose("The ancient kingdom was bathed in starlight", style="epic")
        assert isinstance(prose, str)
        assert len(prose) > 0
