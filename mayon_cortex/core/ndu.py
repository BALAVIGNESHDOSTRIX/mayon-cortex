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
Node Decision Unit (NDU) — The Neuron Firing Rule
===================================================
A lightweight gradient-boosted tree model that makes a local decision
at each active graph node during wavefront expansion.

Brain analogy: Each neuron independently decides whether to fire (propagate
activation) based on local inputs — its own state, the incoming signal
strength, and the type of synaptic connection.

Input:  52-feature vector computed from local node + edge + query context
Output: {follow_prob, halt_prob, backtrack_prob, relevance_score}

Before enough training data accumulates, uses a heuristic fallback
(cosine similarity + edge confidence) to make reasonable decisions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class NDUDecision:
    """Output of a single node decision."""
    follow_prob: float      # P(propagate activation through this edge)
    halt_prob: float        # P(stop here — this node is the answer)
    backtrack_prob: float   # P(this path is wrong, go back)
    relevance_score: float  # How relevant this node is to the query


class NodeDecisionUnit:
    """
    The neuron firing rule.

    A lightweight tree-based model that decides at each graph node:
    - Should activation propagate through this edge? (follow)
    - Should reasoning stop here? (halt)
    - Should we backtrack? (backtrack)

    Uses scikit-learn GradientBoostingClassifier internally.
    Falls back to heuristics until enough training data is collected.

    52 features per decision (see extract_features for full list).
    """

    def __init__(self, config: CortexConfig):
        self.config = config
        self.model = None
        self._trained = False
        self._training_buffer_X: List[np.ndarray] = []
        self._training_buffer_y: List[int] = []  # 1 = good follow, 0 = bad follow

    def extract_features(
        self,
        query_vec: np.ndarray,
        node_vec: np.ndarray,
        edge_confidence: float,
        relation_type: str,
        node_level: int,
        node_sector: str,
        query_sector: str,
        access_count: int,
        hop_depth: int,
        max_hops: int,
        out_degree: int,
        max_degree: int,
        node_timestamp: float,
        current_time: float,
        path_confidence: float,
        is_phrase_node: bool,
        neighbor_relevance: float,
        is_visited: bool,
        is_dead_end: bool,
        relation_types_list: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract a 52-dimensional feature vector for this node+edge decision.

        Features 1-2:   Semantic similarity and edge strength
        Features 3-40:  One-hot relation type encoding (38 types)
        Features 41-52: Structural, temporal, and meta features
        """
        # Default relation type list (38 universal + 4 language + 6 decision = 48,
        # but we use top 38 for encoding, rest as flags)
        if relation_types_list is None:
            relation_types_list = [
                "is-a", "part-of", "has-property", "same-as", "differs-from",
                "causes", "prevents", "enables", "derived-from", "transforms-into",
                "located-in", "adjacent-to", "precedes", "created-by",
                "uses", "returns", "calls", "inherits", "implements",
                "diagnoses", "treats", "contraindicates", "manifests-as", "synthesizes",
                "interacts-with", "metabolized-by", "inhibits", "induces",
                "monitored-by", "has-severity",
                "regulates", "cites", "prohibits", "funds",
                "implies", "proves", "contradicts", "answered-by",
            ]

        features = np.zeros(self.config.ndu_num_features, dtype=np.float32)

        # 1. Query-node cosine similarity
        q_norm = np.linalg.norm(query_vec)
        n_norm = np.linalg.norm(node_vec)
        if q_norm > 0 and n_norm > 0:
            features[0] = float(np.dot(query_vec, node_vec) / (q_norm * n_norm))
        else:
            features[0] = 0.0

        # 2. Edge confidence (synaptic strength)
        features[1] = edge_confidence

        # 3-40. One-hot relation type encoding (38 dimensions)
        rel_lower = relation_type.lower()
        for i, rt in enumerate(relation_types_list[:38]):
            if rel_lower == rt.lower():
                features[2 + i] = 1.0
                break

        # 41. Node level (FACT=0, CONCEPT=1, ABSTRACT=2, PHRASE=3)
        features[40] = node_level / 3.0

        # 42. Sector match (same brain region?)
        features[41] = 1.0 if node_sector == query_sector else 0.0

        # 43. Access count (reinforcement strength) — log-scaled
        features[42] = np.log1p(access_count)

        # 44. Hop depth (distance from stimulus origin)
        features[43] = hop_depth / max(max_hops, 1)

        # 45. Out-degree (hub or leaf neuron?) — normalized
        features[44] = out_degree / max(max_degree, 1)

        # 46. Temporal recency (recent memory boost)
        age = current_time - node_timestamp
        features[45] = 1.0 / (1.0 + age / 86400.0)  # decay over days

        # 47. Cumulative path confidence
        features[46] = path_confidence

        # 48. Is phrase node (language vs knowledge)
        features[47] = 1.0 if is_phrase_node else 0.0

        # 49. Neighbor mean relevance (surrounding context quality)
        features[48] = neighbor_relevance

        # 50. Already visited (avoid loops)
        features[49] = 1.0 if is_visited else 0.0

        # 51. Dead end (no further edges)
        features[50] = 1.0 if is_dead_end else 0.0

        # 52. Sector is safety-critical
        safety_sectors = {"medical", "legal", "finance"}
        features[51] = 1.0 if node_sector in safety_sectors else 0.0

        return features

    def predict(self, features: np.ndarray) -> NDUDecision:
        """
        Predict whether to follow, halt, or backtrack at this node.

        Uses trained tree model if available, otherwise falls back to
        heuristic rules based on cosine similarity and edge confidence.
        """
        if self._trained and self.model is not None:
            return self._predict_tree(features)
        else:
            return self._predict_heuristic(features)

    def _predict_heuristic(self, features: np.ndarray) -> NDUDecision:
        """
        Heuristic fallback for untrained model.
        Uses a simple rule: follow if cosine similarity × edge confidence
        exceeds threshold, halt if similarity is very high, backtrack if visited.
        """
        cos_sim = features[0]       # query-node similarity
        edge_conf = features[1]     # edge confidence
        hop_depth = features[43]    # how deep into graph
        is_visited = features[49]   # already visited?
        is_dead_end = features[50]  # no more edges?
        path_conf = features[46]    # cumulative path confidence

        # Combined signal: weighted combination of semantic relevance and synaptic edge strength
        signal = 0.4 * max(cos_sim, 0.0) + 0.6 * edge_conf * max(path_conf, 0.2)

        # Follow probability: higher signal = more likely to follow
        follow_prob = min(1.0, max(0.1, signal))

        # Halt probability: high when similarity is very high and deep enough
        halt_prob = max(cos_sim, 0.0) ** 2 * min(hop_depth * 2, 1.0) * 0.5

        # Backtrack probability: high when visited or signal is weak
        backtrack_prob = 0.0
        if is_visited > 0.5:
            backtrack_prob = 0.9
            follow_prob = 0.05
        elif signal < 0.15:
            backtrack_prob = 0.4
            follow_prob *= 0.3

        if is_dead_end > 0.5:
            halt_prob = max(halt_prob, 0.6)
            follow_prob = 0.0

        # Relevance score: raw signal strength
        relevance = signal

        return NDUDecision(
            follow_prob=follow_prob,
            halt_prob=halt_prob,
            backtrack_prob=backtrack_prob,
            relevance_score=relevance,
        )

    def _predict_tree(self, features: np.ndarray) -> NDUDecision:
        """Predict using the trained gradient-boosted tree model."""
        X = features.reshape(1, -1)
        # Probability of class 1 (good follow decision)
        proba = self.model.predict_proba(X)[0]
        follow_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Derive halt and backtrack from features + follow probability
        cos_sim = features[0]
        hop_depth = features[43]
        is_visited = features[49]

        halt_prob = (1.0 - follow_prob) * cos_sim * min(hop_depth * 2, 1.0)
        backtrack_prob = (1.0 - follow_prob) * (1.0 - cos_sim)
        if is_visited > 0.5:
            backtrack_prob = max(backtrack_prob, 0.8)
            follow_prob *= 0.1

        relevance = follow_prob * cos_sim

        return NDUDecision(
            follow_prob=follow_prob,
            halt_prob=halt_prob,
            backtrack_prob=backtrack_prob,
            relevance_score=relevance,
        )

    def record_outcome(self, features: np.ndarray, was_good: bool):
        """
        Record the outcome of a decision for future training.
        was_good=True if following this edge led to a correct/useful answer.
        """
        self._training_buffer_X.append(features.copy())
        self._training_buffer_y.append(1 if was_good else 0)

        # Cap buffer size
        if len(self._training_buffer_X) > self.config.max_training_buffer:
            self._training_buffer_X = self._training_buffer_X[-self.config.max_training_buffer:]
            self._training_buffer_y = self._training_buffer_y[-self.config.max_training_buffer:]

    def train(self, force: bool = False):
        """
        Train (or retrain) the tree model on accumulated data.
        Called periodically by the OnlineLearner.
        """
        n_samples = len(self._training_buffer_X)
        if n_samples < self.config.ndu_min_samples_to_train and not force:
            return False

        # Need at least both classes
        if len(set(self._training_buffer_y)) < 2:
            return False

        X = np.array(self._training_buffer_X)
        y = np.array(self._training_buffer_y)

        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.ndu_n_estimators,
                max_depth=self.config.ndu_max_depth,
                learning_rate=self.config.ndu_learning_rate,
                subsample=0.8,
                random_state=42,
                eval_metric="logloss",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.ndu_n_estimators,
                max_depth=self.config.ndu_max_depth,
                learning_rate=self.config.ndu_learning_rate,
                subsample=0.8,
                random_state=42,
            )

        self.model.fit(X, y)
        self._trained = True
        return True

    def save(self, path: str):
        """Save NDU model and training buffer to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "trained": self._trained,
            "buffer_X": self._training_buffer_X,
            "buffer_y": self._training_buffer_y,
        }
        with open(save_dir / "ndu_state.pkl", "wb") as f:
            pickle.dump(state, f)

        if self._trained and self.model is not None:
            with open(save_dir / "ndu_model.pkl", "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path: str):
        """Load NDU model and training buffer from disk."""
        save_dir = Path(path)

        state_file = save_dir / "ndu_state.pkl"
        if state_file.exists():
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            self._trained = state["trained"]
            self._training_buffer_X = state["buffer_X"]
            self._training_buffer_y = state["buffer_y"]

        model_file = save_dir / "ndu_model.pkl"
        if model_file.exists():
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            self._trained = True
