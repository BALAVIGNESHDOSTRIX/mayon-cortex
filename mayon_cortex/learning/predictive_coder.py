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
Predictive Coder — Hierarchical Prediction Error Learning
============================================================
Phase Ω of the Brain-Like Intelligence upgrade.

The brain's ACTUAL learning algorithm (strong neuroscience evidence):
  Each cortical layer PREDICTS what it will see → computes error → adjusts locally.
  Mathematically equivalent to backpropagation at convergence!

For cortex-graph:
  Each node PREDICTS what its neighbors will activate.
  Prediction error = actual - expected.
  Error drives learning: reduce error = better predictions = better reasoning.

This is biologically plausible and runs entirely with LOCAL operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class PredictionRecord:
    """Record of predictions and errors for a node."""
    node_id: str
    prediction_weights: np.ndarray      # Weights for predicting neighbors
    prediction_errors: List[float] = field(default_factory=list)
    cumulative_error: float = 0.0
    update_count: int = 0
    learning_rate: float = 0.01


# Aliases
PredictionError = PredictionRecord


class PredictiveCoder:

    """
    Hierarchical predictive coding for the knowledge graph.
    
    How it works:
    1. Each node PREDICTS what its connected neighbors' vectors should be
    2. When actual neighbor activation arrives, compute prediction error
    3. Error is used to:
       a. Update prediction weights (reduce future error) — LOCAL
       b. Signal surprise to higher-level nodes — enables abstraction
    4. Over time, nodes become "specialized" predictors of their neighborhood
    
    Key insight: Nodes with LOW prediction error have WELL-UNDERSTOOD connections.
    Nodes with HIGH prediction error are at the FRONTIER of knowledge (interesting!).
    """

    def __init__(self, config: Optional[CortexConfig] = None, learning_rate: float = 0.01):
        self.config = config or CortexConfig()
        self.learning_rate = learning_rate
        self.records: Dict[str, PredictionRecord] = {}
        self._initialized: Set[str] = set()


    def predict_neighbors(
        self,
        node_id: str,
        graph: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Predict what each neighbor's activation SHOULD be.
        
        Uses learned prediction weights to generate expected vectors
        for each connected neighbor.
        
        Returns dict of neighbor_id → predicted_vector.
        """
        node = graph.nodes.get(node_id)
        if not node:
            return {}

        record = self._ensure_record(node_id, len(node.vector))
        predictions: Dict[str, np.ndarray] = {}

        # Predict each neighbor
        for eid in graph._out_edges.get(node_id, []):
            if eid not in graph.edges:
                continue
            edge = graph.edges[eid]
            target = graph.nodes.get(edge.target_id)
            if not target:
                continue

            # Linear prediction: predicted = node_vec × prediction_weights
            predicted = node.vector * record.prediction_weights * edge.confidence
            predictions[edge.target_id] = predicted

        return predictions

    def compute_errors(
        self,
        node_id: str,
        predictions: Dict[str, np.ndarray],
        graph: Any,
    ) -> Dict[str, float]:
        """
        Compute prediction error for each neighbor.
        
        Error = ||actual_vector - predicted_vector||² / dim
        
        Low error → well-predicted → well-understood connection
        High error → surprising → frontier of knowledge
        """
        errors: Dict[str, float] = {}

        for neighbor_id, predicted in predictions.items():
            actual_node = graph.nodes.get(neighbor_id)
            if not actual_node:
                continue

            actual = actual_node.vector
            error_vec = actual - predicted
            mse = float(np.mean(error_vec ** 2))
            errors[neighbor_id] = mse

        return errors

    def learn(
        self,
        node_id: str,
        graph: Any,
    ) -> float:
        """
        One learning step: predict → compute error → update weights.
        
        This is the complete predictive coding cycle for a single node.
        Returns average prediction error.
        """
        node = graph.nodes.get(node_id)
        if not node:
            return 0.0

        record = self._ensure_record(node_id, len(node.vector))

        # Step 1: Predict
        predictions = self.predict_neighbors(node_id, graph)
        if not predictions:
            return 0.0

        # Step 2: Compute errors
        errors = self.compute_errors(node_id, predictions, graph)
        if not errors:
            return 0.0

        # Step 3: Update prediction weights to reduce error
        total_error = 0.0
        for neighbor_id, error in errors.items():
            actual = graph.nodes[neighbor_id].vector
            predicted = predictions[neighbor_id]
            error_vec = actual - predicted

            # Update weights: move predictions toward actual
            # This is the LOCAL learning rule (no backprop)
            weight_update = record.learning_rate * error_vec * node.vector
            record.prediction_weights += weight_update

            total_error += error

        avg_error = total_error / len(errors)
        record.prediction_errors.append(avg_error)
        record.cumulative_error += avg_error
        record.update_count += 1

        return avg_error

    def learn_all(
        self,
        graph: Any,
        num_epochs: int = 3,
    ) -> Dict[str, float]:
        """
        Run predictive coding learning on ALL nodes.
        
        Each node independently learns to predict its neighbors.
        Returns dict of node_id → final average error.
        """
        final_errors: Dict[str, float] = {}

        for _ in range(num_epochs):
            for nid in list(graph.nodes.keys()):
                error = self.learn(nid, graph)
                final_errors[nid] = error

        return final_errors

    def find_surprises(
        self,
        graph: Any,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most surprising (highest error) connections.
        
        High prediction error = the system DOESN'T UNDERSTAND this connection.
        These are the most interesting areas to explore or where more data is needed.
        
        Returns list of (source_id, target_id, error) sorted by error descending.
        """
        surprises: List[Tuple[str, str, float]] = []

        for nid in graph.nodes:
            predictions = self.predict_neighbors(nid, graph)
            errors = self.compute_errors(nid, predictions, graph)

            for neighbor_id, error in errors.items():
                surprises.append((nid, neighbor_id, error))

        surprises.sort(key=lambda x: x[2], reverse=True)
        return surprises[:top_k]

    def find_well_understood(
        self,
        graph: Any,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most well-understood (lowest error) connections.
        
        Low error = the system PERFECTLY PREDICTS this connection.
        These are the most reliable knowledge edges.
        """
        understood: List[Tuple[str, str, float]] = []

        for nid in graph.nodes:
            predictions = self.predict_neighbors(nid, graph)
            errors = self.compute_errors(nid, predictions, graph)

            for neighbor_id, error in errors.items():
                understood.append((nid, neighbor_id, error))

        understood.sort(key=lambda x: x[2])
        return understood[:top_k]

    def get_node_quality(self, node_id: str) -> Dict[str, Any]:
        """Get prediction quality metrics for a node."""
        record = self.records.get(node_id)
        if not record:
            return {"status": "not_trained"}

        recent_errors = record.prediction_errors[-10:]
        avg_recent = float(np.mean(recent_errors)) if recent_errors else 0.0
        trend = "improving" if len(recent_errors) > 1 and recent_errors[-1] < recent_errors[0] else "stable"

        return {
            "avg_recent_error": avg_recent,
            "total_updates": record.update_count,
            "cumulative_error": record.cumulative_error,
            "trend": trend,
        }

    def strengthen_edges(
        self,
        graph: Any,
        error_threshold: float = 0.1,
    ) -> int:
        """
        Strengthen edges with low prediction error (Hebbian reinforcement).
        
        If a node can PREDICT its neighbor well (low error),
        the connection is reliable → boost confidence.
        
        Returns number of edges strengthened.
        """
        count = 0

        for nid in graph.nodes:
            predictions = self.predict_neighbors(nid, graph)
            errors = self.compute_errors(nid, predictions, graph)

            for neighbor_id, error in errors.items():
                if error < error_threshold:
                    # Find and strengthen the edge
                    for eid in graph._out_edges.get(nid, []):
                        if eid in graph.edges and graph.edges[eid].target_id == neighbor_id:
                            edge = graph.edges[eid]
                            old_conf = edge.confidence
                            edge.confidence = min(1.0, old_conf + 0.02)
                            count += 1
                            break

        return count

    # ── Internal ──

    def _ensure_record(self, node_id: str, dim: int) -> PredictionRecord:
        """Get or create prediction record for a node."""
        if node_id not in self.records:
            self.records[node_id] = PredictionRecord(
                node_id=node_id,
                prediction_weights=np.ones(dim, dtype=np.float32) * 0.5,
                learning_rate=0.01,
            )
        return self.records[node_id]
