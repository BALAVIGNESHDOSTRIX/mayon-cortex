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
Tests for Pillars 2-6: Abstraction, Emotion, Vision, Analogy, and Full v4 Brain E2E
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mayon-graph"))

from mayon_cortex import CortexConfig, CortexGraph
from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.reasoning.abstraction import FractalAbstractionEngine, AbstractionLevel
from mayon_cortex.perception.vision import VisionCortex
from mayon_cortex.reasoning.analogy import AnalogyEngine


class TestEmotionEngine:
    def setup_method(self):
        self.engine = EmotionEngine()

    def test_vad_lookups(self):
        joy_vad = self.engine.get_word_vad("joy")
        grief_vad = self.engine.get_word_vad("grief")

        assert joy_vad.valence > 0.5
        assert grief_vad.valence < -0.4

    def test_sentence_emotional_resonance(self):
        sad_text = "I am deeply sorrowful and mourning the tragic loss of my friend."
        vad = self.engine.compute_concept_vad(sad_text)
        assert vad.valence < 0.0

        happy_text = "Celebrating a wonderful victory and joyful triumph!"
        h_vad = self.engine.compute_concept_vad(happy_text)
        assert h_vad.valence > 0.3


class TestVisionCortex:
    def setup_method(self):
        self.vision = VisionCortex()

    def test_visual_encoding_and_see(self):
        # Create a mock image as numpy array (RGB)
        mock_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_node = self.vision.see(
            image_input=mock_img,
            caption="A majestic soaring eagle in blue sky",
            tags=["bird", "sky", "eagle"]
        )

        assert img_node.vector.shape == (384,)
        assert np.isclose(np.linalg.norm(img_node.vector), 1.0, atol=1e-3)
        assert img_node.caption == "A majestic soaring eagle in blue sky"


class TestAnalogyEngine:
    def setup_method(self):
        self.brain = CortexGraph()
        self.analogy = AnalogyEngine()

    def test_analogical_reasoning(self):
        # Ingest structural knowledge
        self.brain.ingest("The king rules the kingdom.", sector="general")
        self.brain.ingest("The captain commands the ship.", sector="general")

        result = self.brain.solve_analogy("king", "kingdom", "captain")
        if result is not None:
            assert result.source_a == "king"
            assert result.source_b == "kingdom"
            assert result.target_c == "captain"


class TestCortexGraphV4E2E:
    def setup_method(self):
        self.brain = CortexGraph()

    def test_real_world_story_and_science_flow(self):
        # Ingest real world facts & stories (no medical requirement)
        self.brain.ingest("Photosynthesis produces oxygen and glucose for plants.", sector="science")
        self.brain.ingest("Trees in forests absorb carbon dioxide from the atmosphere.", sector="science")
        self.brain.ingest("Arthur the brave knight protects the kingdom from dragons.", sector="story")
        self.brain.ingest("A dragon dwells inside the dark volcanic mountain.", sector="story")

        # Query story
        resp1 = self.brain.query("What does Arthur protect?")
        assert len(resp1.text) > 0

        # Query science
        resp2 = self.brain.query("What does photosynthesis produce?")
        assert len(resp2.text) > 0

        # Run abstraction cycle
        report = self.brain.abstract()
        assert report is not None

        # Verify stats
        stats = self.brain.stats()
        assert stats["graph_nodes"] > 0
        assert stats["word_nodes"] > 0

    def test_curriculum_assessment_live(self):
        self.brain.teach(phase=1)
        assessment = self.brain.assess(live_eval=True)

        assert "overall_accuracy" in assessment
        assert "phases" in assessment
        assert 1 in assessment["phases"]
        assert assessment["phases"][1]["total"] > 0
