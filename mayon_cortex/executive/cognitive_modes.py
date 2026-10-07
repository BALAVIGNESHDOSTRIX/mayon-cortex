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
Cognitive Modes — The Dual-Mode Brain Architecture
=====================================================
Phase 1.5 of the Brain-Like Intelligence upgrade.

The human brain uses the SAME neurons traversed DIFFERENTLY:
  - Task-Positive Network (TPN): focused, analytical, factual
  - Default Mode Network (DMN): wandering, creative, recombinative

This module implements the mode switch (prefrontal cortex) that changes
how the SAME knowledge graph is traversed depending on the task.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CognitiveMode(Enum):
    """The five cognitive operating modes."""
    FACTUAL = "factual"             # TPN — strict, evidence-based, zero hallucination
    CREATIVE = "creative"           # DMN — relaxed, cross-domain, recombinative
    PROBLEM_SOLVING = "solving"     # Goal-directed search + decomposition
    JUDGING = "judging"             # Precedent matching + explanation
    EXPLORATORY = "exploratory"     # Curiosity-driven connection finding


@dataclass
class ModeConfig:
    """
    How the SAME graph traversal changes per mode.
    
    These parameters modulate the ActivationWave, NDU, and FeedforwardCascade
    to produce radically different behaviors from the same graph.
    """
    ndu_follow_threshold: float     # How easily neurons fire
    max_hops: int                   # How deep to explore
    cross_sector_allowed: bool      # Can we jump across domains?
    random_jump_probability: float  # Chance of spontaneous association
    confidence_floor: float         # Minimum confidence to consider an edge
    blend_probability: float        # Chance of conceptual blending mid-traversal
    constraint_strictness: float    # How strictly to enforce logical rules (0-1)
    noise_level: float              # Neural noise for creativity (0-1)
    beam_width: int                 # Beam width for generation

    @classmethod
    def factual(cls) -> "ModeConfig":
        """Strict, evidence-based traversal. Zero hallucination."""
        return cls(
            ndu_follow_threshold=0.25,
            max_hops=5,
            cross_sector_allowed=False,
            random_jump_probability=0.0,
            confidence_floor=0.3,
            blend_probability=0.0,
            constraint_strictness=1.0,
            noise_level=0.0,
            beam_width=8,
        )

    @classmethod
    def creative(cls) -> "ModeConfig":
        """Relaxed, exploratory traversal. Enables creative recombination."""
        return cls(
            ndu_follow_threshold=0.08,
            max_hops=10,
            cross_sector_allowed=True,
            random_jump_probability=0.15,
            confidence_floor=0.05,
            blend_probability=0.20,
            constraint_strictness=0.3,
            noise_level=0.3,
            beam_width=12,
        )

    @classmethod
    def problem_solving(cls) -> "ModeConfig":
        """Moderate traversal. Goal-directed but allows lateral thinking."""
        return cls(
            ndu_follow_threshold=0.15,
            max_hops=8,
            cross_sector_allowed=True,
            random_jump_probability=0.05,
            confidence_floor=0.15,
            blend_probability=0.10,
            constraint_strictness=0.7,
            noise_level=0.1,
            beam_width=10,
        )

    @classmethod
    def judging(cls) -> "ModeConfig":
        """Precedent-focused traversal. Finds similar cases."""
        return cls(
            ndu_follow_threshold=0.20,
            max_hops=6,
            cross_sector_allowed=False,
            random_jump_probability=0.0,
            confidence_floor=0.2,
            blend_probability=0.0,
            constraint_strictness=0.8,
            noise_level=0.0,
            beam_width=8,
        )

    @classmethod
    def exploratory(cls) -> "ModeConfig":
        """Wide, curiosity-driven traversal. Finds unexpected connections."""
        return cls(
            ndu_follow_threshold=0.10,
            max_hops=12,
            cross_sector_allowed=True,
            random_jump_probability=0.10,
            confidence_floor=0.05,
            blend_probability=0.15,
            constraint_strictness=0.4,
            noise_level=0.2,
            beam_width=15,
        )


# Intent detection patterns
_FACTUAL_PATTERNS = [
    r"what (?:is|are|was|were)\b",
    r"who (?:is|are|was|were)\b",
    r"when (?:did|does|is|was)\b",
    r"where (?:is|are|was|were)\b",
    r"how (?:many|much|often)\b",
    r"define\b",
    r"explain\b",
    r"describe\b",
    r"list\b",
    r"tell me about\b",
]

_CREATIVE_PATTERNS = [
    r"write\b",
    r"compose\b",
    r"create\b",
    r"imagine\b",
    r"generate\b",
    r"make (?:a|an|me)\b",
    r"poem\b",
    r"story\b",
    r"creative\b",
    r"fiction\b",
    r"invent\b",
    r"dream\b",
]

_PROBLEM_PATTERNS = [
    r"solve\b",
    r"calculate\b",
    r"compute\b",
    r"how (?:to|do|can|should)\b",
    r"fix\b",
    r"debug\b",
    r"plan\b",
    r"step.by.step\b",
    r"formula\b",
    r"equation\b",
    r"if .+ then\b",
]

_JUDGING_PATTERNS = [
    r"should\b",
    r"is (?:it|this) (?:right|wrong|correct|acceptable)\b",
    r"judge\b",
    r"evaluate\b",
    r"assess\b",
    r"verdict\b",
    r"ruling\b",
    r"decision\b",
    r"fair\b",
    r"justified\b",
    r"appropriate\b",
]

_EXPLORATORY_PATTERNS = [
    r"what if\b",
    r"connect(?:ion|ions)\b",
    r"relat(?:e|ed|ionship)\b",
    r"pattern\b",
    r"discover\b",
    r"explore\b",
    r"hidden\b",
    r"unexpected\b",
    r"between .+ and\b",
    r"common .+ between\b",
    r"similar(?:ity|ities)\b",
]


class CognitiveModeController:
    """
    The prefrontal cortex mode switch.
    
    Detects task type from query intent and switches how the graph
    is traversed. Same brain, different cognitive mode.
    """

    def __init__(self, default_mode: str = "factual"):
        self.default_mode = default_mode

    def detect_mode(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> CognitiveMode:

        """
        Detect the appropriate cognitive mode from query intent.
        
        Uses pattern matching on the query text to classify intent:
          - Questions about facts → FACTUAL
          - Creative requests → CREATIVE
          - Problem-solving → PROBLEM_SOLVING
          - Judgment/evaluation → JUDGING
          - Connection-finding → EXPLORATORY
        """
        q_lower = query.lower().strip()

        # Score each mode
        scores = {
            CognitiveMode.FACTUAL: self._score_patterns(q_lower, _FACTUAL_PATTERNS),
            CognitiveMode.CREATIVE: self._score_patterns(q_lower, _CREATIVE_PATTERNS),
            CognitiveMode.PROBLEM_SOLVING: self._score_patterns(q_lower, _PROBLEM_PATTERNS),
            CognitiveMode.JUDGING: self._score_patterns(q_lower, _JUDGING_PATTERNS),
            CognitiveMode.EXPLORATORY: self._score_patterns(q_lower, _EXPLORATORY_PATTERNS),
        }

        # Default bias toward factual (most common)
        scores[CognitiveMode.FACTUAL] += 0.5

        # Pick highest score
        best_mode = max(scores, key=scores.get)

        # Context can override
        if context:
            ctx_lower = context.lower()
            if "creative" in ctx_lower or "poem" in ctx_lower:
                return CognitiveMode.CREATIVE
            if "solve" in ctx_lower or "calculate" in ctx_lower:
                return CognitiveMode.PROBLEM_SOLVING

        return best_mode

    def get_mode_config(self, mode: CognitiveMode) -> ModeConfig:
        """Get traversal parameters for the detected mode."""
        config_map = {
            CognitiveMode.FACTUAL: ModeConfig.factual,
            CognitiveMode.CREATIVE: ModeConfig.creative,
            CognitiveMode.PROBLEM_SOLVING: ModeConfig.problem_solving,
            CognitiveMode.JUDGING: ModeConfig.judging,
            CognitiveMode.EXPLORATORY: ModeConfig.exploratory,
        }
        factory = config_map.get(mode, ModeConfig.factual)
        return factory()

    def get_output_label(self, mode: CognitiveMode) -> str:
        """
        Label output appropriately to maintain zero-hallucination guarantee.
        
        FACTUAL output is presented as evidence-based fact.
        CREATIVE output is explicitly labeled as generated/imagined.
        """
        labels = {
            CognitiveMode.FACTUAL: "Based on evidence",
            CognitiveMode.CREATIVE: "Creative generation (imagined, not factual)",
            CognitiveMode.PROBLEM_SOLVING: "Solution",
            CognitiveMode.JUDGING: "Judgment (based on learned precedents)",
            CognitiveMode.EXPLORATORY: "Exploration (speculative connections)",
        }
        return labels.get(mode, "Response")

    def _score_patterns(self, text: str, patterns: list) -> float:
        """Score how many patterns match in the text."""
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text):
                score += 1.0
        return score


# Aliases for unified naming
CognitiveModeEngine = CognitiveModeController
CognitiveParameters = ModeConfig

