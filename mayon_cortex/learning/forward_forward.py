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
Forward-Forward Learning — Goodness-Based Node Training Without Backprop
==========================================================================
Phase Ω of the Brain-Like Intelligence upgrade.

Hinton (2022): Each node learns INDEPENDENTLY using two forward passes:
  Pass 1: Show REAL data → compute "goodness" (high = good match)
  Pass 2: Show NEGATIVE data → compute "goodness" (should be low)
  Update: Adjust to increase goodness for real, decrease for negative

No backward pass. No global loss function. Each node is autonomous.
This is biologically plausible — how real neurons likely learn.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GoodnessRecord:
    """Record of goodness scores for a node across training."""
    node_id: str
    positive_goodness_history: List[float] = field(default_factory=list)
    negative_goodness_history: List[float] = field(default_factory=list)
    threshold: float = 0.5
    learning_rate: float = 0.03

    @property
    def separation(self) -> float:
        """How well the node separates positive from negative."""
        if not self.positive_goodness_history or not self.negative_goodness_history:
            return 0.0
        avg_pos = np.mean(self.positive_goodness_history[-10:])
        avg_neg = np.mean(self.negative_goodness_history[-10:])
        return avg_pos - avg_neg


# Aliases
GoodnessProfile = GoodnessRecord


class ForwardForwardLearner:

    """
    Forward-Forward learning for graph nodes.
    
    Each node in the graph learns to distinguish "good" (real) activations
    from "bad" (noise/negative) activations WITHOUT backpropagation.
    
    This replaces the tree-based NDU training with biologically plausible
    local learning:
    
    Old (NDU):
      Collect global examples → train decision trees → apply globally
      Problem: requires batch training, not biologically plausible
    
    New (Forward-Forward):
      Each node independently computes goodness(activation)
      Each node independently adjusts to maximize positive/negative separation
      Result: emergent global intelligence from local decisions
    """

    def __init__(
        self,
        learning_rate: float = 0.03,
        goodness_threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.goodness_threshold = goodness_threshold
        self.node_records: Dict[str, GoodnessRecord] = {}
        self._node_weights: Dict[str, np.ndarray] = {}

    def compute_goodness(self, activations: np.ndarray) -> float:
        """
        Goodness = sum of squared activations.
        
        This is the local "energy" metric that each node uses to evaluate
        how well the current activation matches learned patterns.
        
        Higher goodness → more "real" / recognized pattern
        Lower goodness → more "noise" / unfamiliar pattern
        """
        return float(np.sum(activations ** 2))

    def train_node_positive(
        self,
        node_id: str,
        positive_activation: np.ndarray,
    ) -> float:
        """
        Forward pass 1: Train on POSITIVE (real) data.
        
        Record goodness and adjust weights to increase it.
        Returns the goodness score.
        """
        record = self._ensure_record(node_id)
        weights = self._ensure_weights(node_id, len(positive_activation))

        # Compute weighted activation
        weighted = positive_activation * weights
        goodness = self.compute_goodness(weighted)
        record.positive_goodness_history.append(goodness)

        # Adjust weights to increase goodness for positive
        if goodness < record.threshold:
            # Strengthen dimensions that were active
            gradient = 2.0 * weighted * positive_activation
            weights += self.learning_rate * gradient
            self._node_weights[node_id] = weights

        return goodness

    def train_node_negative(
        self,
        node_id: str,
        negative_activation: np.ndarray,
    ) -> float:
        """
        Forward pass 2: Train on NEGATIVE (noise/corrupt) data.
        
        Record goodness and adjust weights to decrease it.
        Returns the goodness score (should be LOW for well-trained nodes).
        """
        record = self._ensure_record(node_id)
        weights = self._ensure_weights(node_id, len(negative_activation))

        # Compute weighted activation
        weighted = negative_activation * weights
        goodness = self.compute_goodness(weighted)
        record.negative_goodness_history.append(goodness)

        # Adjust weights to decrease goodness for negative
        if goodness > record.threshold:
            # Weaken dimensions that were active on negative data
            gradient = 2.0 * weighted * negative_activation
            weights -= self.learning_rate * gradient
            self._node_weights[node_id] = weights

        return goodness

    def is_positive(self, node_id: str, activation: np.ndarray) -> Tuple[bool, float]:
        """
        Classify an activation as positive (real) or negative (noise).
        
        Returns (is_positive, goodness_score).
        """
        weights = self._ensure_weights(node_id, len(activation))
        weighted = activation * weights
        goodness = self.compute_goodness(weighted)
        record = self._ensure_record(node_id)
        return goodness > record.threshold, goodness

    def generate_negatives(
        self,
        positive_data: np.ndarray,
        method: str = "shuffle",
    ) -> np.ndarray:
        """
        Generate negative examples by corrupting real data.
        
        Methods:
          - "shuffle": randomly permute dimensions
          - "noise": add Gaussian noise
          - "mask": randomly zero out dimensions
          - "swap": swap pairs of dimensions
        """
        neg = positive_data.copy()

        if method == "shuffle":
            np.random.shuffle(neg)

        elif method == "noise":
            noise = np.random.randn(*neg.shape) * 0.5
            neg = neg + noise

        elif method == "mask":
            mask = np.random.random(neg.shape) > 0.3  # Drop 30%
            neg = neg * mask

        elif method == "swap":
            for i in range(0, len(neg) - 1, 2):
                neg[i], neg[i + 1] = neg[i + 1], neg[i]

        return neg

    def train_from_graph(
        self,
        graph: Any,
        num_epochs: int = 5,
        negative_method: str = "shuffle",
    ) -> Dict[str, float]:
        """
        Train all graph nodes using Forward-Forward.
        
        For each node:
          1. Use the node's own vector as POSITIVE example
          2. Generate a corrupted version as NEGATIVE example
          3. Train the node to distinguish them
          
        Returns dict of node_id → separation score.
        """
        separations: Dict[str, float] = {}

        for nid, node in graph.nodes.items():
            vec = node.vector
            if vec is None or np.linalg.norm(vec) < 1e-8:
                continue

            for _ in range(num_epochs):
                # Positive pass
                self.train_node_positive(nid, vec)

                # Negative pass
                neg = self.generate_negatives(vec, method=negative_method)
                self.train_node_negative(nid, neg)

            record = self.node_records.get(nid)
            if record:
                separations[nid] = record.separation

        return separations

    def evaluate_node_quality(self, node_id: str) -> Dict[str, float]:
        """Get training quality metrics for a node."""
        record = self.node_records.get(node_id)
        if not record:
            return {"separation": 0.0, "avg_pos_goodness": 0.0, "avg_neg_goodness": 0.0}

        avg_pos = (
            float(np.mean(record.positive_goodness_history[-10:]))
            if record.positive_goodness_history else 0.0
        )
        avg_neg = (
            float(np.mean(record.negative_goodness_history[-10:]))
            if record.negative_goodness_history else 0.0
        )

        return {
            "separation": record.separation,
            "avg_pos_goodness": avg_pos,
            "avg_neg_goodness": avg_neg,
            "threshold": record.threshold,
            "total_positive_samples": len(record.positive_goodness_history),
            "total_negative_samples": len(record.negative_goodness_history),
        }

    # ── Internal ──

    def _ensure_record(self, node_id: str) -> GoodnessRecord:
        """Get or create a goodness record for a node."""
        if node_id not in self.node_records:
            self.node_records[node_id] = GoodnessRecord(
                node_id=node_id,
                threshold=self.goodness_threshold,
                learning_rate=self.learning_rate,
            )
        return self.node_records[node_id]

    def _ensure_weights(self, node_id: str, dim: int) -> np.ndarray:
        """Get or create weight vector for a node."""
        if node_id not in self._node_weights:
            self._node_weights[node_id] = np.ones(dim, dtype=np.float32)
        w = self._node_weights[node_id]
        if len(w) != dim:
            self._node_weights[node_id] = np.ones(dim, dtype=np.float32)
        return self._node_weights[node_id]
