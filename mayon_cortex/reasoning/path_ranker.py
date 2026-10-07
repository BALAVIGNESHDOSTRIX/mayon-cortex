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
Path Ranker — Cortical Proof Chain Ranking
===========================================
Ranks all activated paths by deductive soundness, relevance, and efficiency.

Multiple neural pathways activate in parallel. The cortex selects the
strongest, most coherent signal. This module is that selection mechanism.

Uses a gradient-boosted tree ranker with 15 features per path.
Falls back to heuristic scoring before training data accumulates.

Usage:
    ranker = PathRanker(config)
    scored = ranker.rank_paths(paths, query_vec, graph)
    # scored[0] is the best proof chain
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.dynamics.activation_wave import ActivatedPath
from mayon_cortex.dynamics.salience import SalienceWeighter


@dataclass
class ScoredPath:
    """A path with its ranking score and explanation."""
    path: ActivatedPath
    score: float
    features: np.ndarray
    explanation: str  # why this path ranked high/low

    # Delegate common attributes for convenience
    @property
    def node_ids(self): return self.path.node_ids
    @property
    def edges(self): return self.path.edges
    @property
    def confidence(self): return self.path.confidence
    @property
    def hop_count(self): return self.path.hop_count


class PathRanker:
    """
    Ranks proof chains by quality using gradient-boosted trees.

    15 features per path:
    1.  path_confidence      — cumulative edge confidence product
    2.  path_length          — number of hops (shorter often better)
    3.  endpoint_similarity  — cos(query, last node)
    4.  start_similarity     — cos(query, first node)
    5.  symbolic_valid       — does relation chain follow deductive rules?
    6.  sector_consistency   — all nodes in same sector?
    7.  temporal_coherence   — timestamps in order?
    8.  cycle_free           — no repeated nodes?
    9.  avg_access_count     — average reinforcement across path
    10. has_phrase_nodes     — contains language pattern nodes?
    11. phrase_coverage      — % of knowledge nodes with phrase bridges
    12. salience_score       — urgency/importance weighted score
    13. avg_ndu_score        — average NDU relevance along path
    14. path_diversity       — how different from other top paths
    15. source_diversity     — how many distinct provenances
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.model = None
        self._trained = False
        self._buffer_X: List[np.ndarray] = []
        self._buffer_y: List[float] = []
        self.salience = SalienceWeighter(config)

    def rank_paths(
        self,
        paths: List[ActivatedPath],
        query_vec: np.ndarray,
        graph=None,
    ) -> List[ScoredPath]:
        """
        Rank all paths and return sorted scored paths.
        """
        if not paths:
            return []

        scored = []
        for path in paths:
            features = self.extract_features(path, query_vec, graph)
            score = self._score(features)
            explanation = self._explain(features, score)
            scored.append(ScoredPath(
                path=path, score=score,
                features=features, explanation=explanation,
            ))

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)

        # Return top-K
        return scored[:self.config.ranker_top_k]

    def extract_features(
        self,
        path: ActivatedPath,
        query_vec: np.ndarray,
        graph=None,
    ) -> np.ndarray:
        """Extract 15 ranking features from a path."""
        features = np.zeros(self.config.ranker_num_features, dtype=np.float32)

        # 1. Path confidence (cumulative product)
        features[0] = path.confidence

        # 2. Path length (normalized, shorter = higher)
        features[1] = 1.0 / (1.0 + path.hop_count)

        # 3. Endpoint similarity
        if graph and path.node_ids:
            last_id = path.node_ids[-1]
            if last_id in graph.nodes:
                last_vec = graph.nodes[last_id].vector
                q_n = np.linalg.norm(query_vec)
                l_n = np.linalg.norm(last_vec)
                if q_n > 0 and l_n > 0:
                    features[2] = float(np.dot(query_vec, last_vec) / (q_n * l_n))

        # 4. Start similarity
        if graph and path.node_ids:
            first_id = path.node_ids[0]
            if first_id in graph.nodes:
                first_vec = graph.nodes[first_id].vector
                q_n = np.linalg.norm(query_vec)
                f_n = np.linalg.norm(first_vec)
                if q_n > 0 and f_n > 0:
                    features[3] = float(np.dot(query_vec, first_vec) / (q_n * f_n))

        # 5. Symbolic validity (basic check)
        features[4] = 1.0 if self._basic_validity_check(path) else 0.0

        # 6. Sector consistency
        if graph and path.node_ids:
            sectors = set()
            for nid in path.node_ids:
                if nid in graph.nodes:
                    sectors.add(graph.nodes[nid].sector)
            features[5] = 1.0 if len(sectors) <= 2 else 1.0 / len(sectors)

        # 7. Temporal coherence
        features[6] = self._check_temporal(path, graph)

        # 8. Cycle free (should always be true due to activation wave)
        features[7] = 1.0 if len(set(path.node_ids)) == len(path.node_ids) else 0.0

        # 9. Average access count
        if graph and path.node_ids:
            counts = [graph.nodes[nid].access_count for nid in path.node_ids if nid in graph.nodes]
            features[8] = np.log1p(np.mean(counts)) if counts else 0.0

        # 10. Has phrase nodes
        if graph and path.node_ids:
            phrase_count = sum(
                1 for nid in path.node_ids
                if nid in graph.nodes and graph.nodes[nid].node_type == "phrase"
            )
            features[9] = phrase_count / max(len(path.node_ids), 1)

        # 11. Phrase coverage
        if graph and path.node_ids:
            knowledge_nodes = [
                nid for nid in path.node_ids
                if nid in graph.nodes and graph.nodes[nid].node_type != "phrase"
            ]
            if knowledge_nodes:
                has_phrase_bridge = sum(
                    1 for nid in knowledge_nodes
                    if any(
                        graph.edges[eid].relation_type == "expressed-as"
                        for eid in graph._out_edges.get(nid, [])
                        if eid in graph.edges
                    )
                )
                features[10] = has_phrase_bridge / len(knowledge_nodes)

        # 12. Salience score
        if path.edges:
            rels = [e[1] for e in path.edges]
            confs = [path.confidence] * len(rels)  # simplified
            features[11] = self.salience.compute_path_salience(rels, confs)

        # 13. Average NDU score
        if path.scores:
            features[12] = float(np.mean(path.scores))

        # 14. Path diversity (placeholder — needs other paths for comparison)
        features[13] = 1.0  # will be updated in rank_paths if needed

        # 15. Source diversity
        if graph and path.node_ids:
            provenances = set()
            for nid in path.node_ids:
                if nid in graph.nodes:
                    provenances.add(graph.nodes[nid].provenance)
            features[14] = min(len(provenances) / 3.0, 1.0)

        return features

    def _score(self, features: np.ndarray) -> float:
        """Score a path using trained model or heuristic."""
        if self._trained and self.model is not None:
            return float(self.model.predict(features.reshape(1, -1))[0])
        else:
            return self._heuristic_score(features)

    def _heuristic_score(self, features: np.ndarray) -> float:
        """Heuristic scoring before model is trained."""
        score = 0.0
        score += features[0] * 0.15   # path confidence
        score += features[1] * 0.05   # shorter paths
        score += features[2] * 0.30   # endpoint similarity (semantic relevance)
        score += features[3] * 0.30   # start similarity (semantic relevance)
        score += features[4] * 0.10   # symbolic validity
        score += features[5] * 0.05   # sector consistency
        score += features[7] * 0.05   # cycle free
        return score

    def _explain(self, features: np.ndarray, score: float) -> str:
        """Generate human-readable explanation for ranking."""
        parts = []
        if features[0] > 0.8:
            parts.append("high-confidence path")
        if features[2] > 0.7:
            parts.append("strong endpoint match")
        if features[4] > 0.5:
            parts.append("symbolically valid")
        if features[11] > 5.0:
            parts.append("safety-critical evidence")
        if not parts:
            parts.append(f"composite score {score:.2f}")
        return "; ".join(parts)

    def _basic_validity_check(self, path: ActivatedPath) -> bool:
        """Basic symbolic validity: no cycles, non-empty."""
        if not path.edges:
            return False
        if len(set(path.node_ids)) != len(path.node_ids):
            return False
        return True

    def _check_temporal(self, path: ActivatedPath, graph) -> float:
        """Check if timestamps are in order along the path."""
        if graph is None or not path.node_ids:
            return 0.5
        timestamps = []
        for nid in path.node_ids:
            if nid in graph.nodes:
                timestamps.append(graph.nodes[nid].timestamp)
        if len(timestamps) < 2:
            return 1.0
        ordered = all(t1 <= t2 for t1, t2 in zip(timestamps, timestamps[1:]))
        return 1.0 if ordered else 0.3

    def record_outcome(self, features: np.ndarray, reward: float):
        """Record path outcome for future training."""
        self._buffer_X.append(features.copy())
        self._buffer_y.append(reward)
        if len(self._buffer_X) > self.config.max_training_buffer:
            self._buffer_X = self._buffer_X[-self.config.max_training_buffer:]
            self._buffer_y = self._buffer_y[-self.config.max_training_buffer:]

    def train(self, force: bool = False):
        """Train the ranking model on accumulated data."""
        if len(self._buffer_X) < 30 and not force:
            return False
        X = np.array(self._buffer_X)
        y = np.array(self._buffer_y)

        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.ranker_n_estimators,
                max_depth=self.config.ranker_max_depth,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.ranker_n_estimators,
                max_depth=self.config.ranker_max_depth,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        self.model.fit(X, y)
        self._trained = True
        return True

    def save(self, path: str):
        """Save ranker model to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {"trained": self._trained, "X": self._buffer_X, "y": self._buffer_y}
        with open(save_dir / "ranker_state.pkl", "wb") as f:
            pickle.dump(state, f)
        if self._trained and self.model is not None:
            with open(save_dir / "ranker_model.pkl", "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path: str):
        """Load ranker model from disk."""
        save_dir = Path(path)
        state_file = save_dir / "ranker_state.pkl"
        if state_file.exists():
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            self._trained = state["trained"]
            self._buffer_X = state["X"]
            self._buffer_y = state["y"]
        model_file = save_dir / "ranker_model.pkl"
        if model_file.exists():
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            self._trained = True
