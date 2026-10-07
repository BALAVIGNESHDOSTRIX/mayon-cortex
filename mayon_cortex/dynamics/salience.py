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
Salience Weighter — The Brain's Amygdala
=========================================
Assigns urgency/importance weights to graph evidence.

A death risk is NOT the same importance as a cost saving.
The salience weighter ensures life-critical information always
gets maximum attention, just like the amygdala flags danger signals
with higher priority than routine stimuli.

Usage:
    weighter = SalienceWeighter()
    weighted_score = weighter.compute(edge, base_score)
    # contraindicates edge: 0.8 * 10.0 = 8.0  (URGENT)
    # costs edge:           0.8 * 1.5  = 1.2   (routine)
"""

from typing import Dict, List, Optional

from mayon_cortex.core.config import SALIENCE_WEIGHTS, CortexConfig


class SalienceWeighter:
    """
    Assigns urgency multipliers to evidence based on relation type.
    
    Brain analogy: The amygdala processes incoming stimuli and flags
    high-importance signals (danger, reward) for priority processing.
    A snake on the path gets 100x the attention of a pretty flower.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.weights = dict(SALIENCE_WEIGHTS)

    def compute(self, relation_type: str, base_score: float) -> float:
        """
        Weight a score by the salience of its relation type.

        Args:
            relation_type: The edge relation type (e.g. "contraindicates")
            base_score: The raw confidence/relevance score

        Returns:
            Salience-weighted score
        """
        multiplier = self.weights.get(relation_type, 1.0)
        return base_score * multiplier

    def compute_path_salience(
        self,
        relation_types: List[str],
        confidences: List[float],
    ) -> float:
        """
        Compute aggregate salience for an entire path.
        The path's salience is dominated by its highest-salience edge
        (one danger signal makes the whole path urgent).

        Args:
            relation_types: List of relation types along the path
            confidences: Confidence scores for each edge

        Returns:
            Path salience score (higher = more urgent/important)
        """
        if not relation_types:
            return 0.0

        max_salience = 0.0
        total_salience = 0.0

        for rel, conf in zip(relation_types, confidences):
            edge_salience = self.compute(rel, conf)
            max_salience = max(max_salience, edge_salience)
            total_salience += edge_salience

        # Dominated by highest (like amygdala hijack) but consider average too
        avg_salience = total_salience / len(relation_types)
        return 0.7 * max_salience + 0.3 * avg_salience

    def is_safety_critical(self, relation_type: str) -> bool:
        """Check if a relation type is safety-critical (salience >= 5.0)."""
        return self.weights.get(relation_type, 1.0) >= 5.0

    def get_urgency_label(self, salience_score: float) -> str:
        """Human-readable urgency label for a salience score."""
        if salience_score >= 8.0:
            return "CRITICAL"
        elif salience_score >= 5.0:
            return "HIGH"
        elif salience_score >= 3.0:
            return "MODERATE"
        elif salience_score >= 1.5:
            return "LOW"
        else:
            return "ROUTINE"
