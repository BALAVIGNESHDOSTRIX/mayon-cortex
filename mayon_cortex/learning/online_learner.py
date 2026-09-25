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
Online Learner — Hebbian Plasticity
=====================================
The brain's learning system. Three learning signals:

1. Hebbian: "neurons that fire together wire together"
   — Strengthen paths that led to correct answers
2. Reinforcement: update NDU trees based on outcomes
   — Train decision models on accumulated experience
3. Growth: add new nodes when novel knowledge is confirmed
   — The graph grows with every successful interaction

No backpropagation. No gradient descent through billions of parameters.
Just: strengthen what works, weaken what doesn't, grow when needed.
"""

import time
from typing import Dict, List, Optional

import numpy as np

from mayon_cortex.core.config import CortexConfig


class OnlineLearner:
    """
    Brain-like plasticity system.

    Called after every query to:
    1. Strengthen used paths (Hebbian)
    2. Record outcomes for NDU/Ranker training
    3. Grow the graph with new knowledge
    4. Periodically retrain tree models
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self._query_count = 0

    def learn(
        self,
        scored_paths: List,
        reward: float,
        graph,
        ndu,
        ranker,
        query_vec: Optional[np.ndarray] = None,
        answer_text: Optional[str] = None,
        embedder=None,
        word_graph=None,
        abstraction_engine=None,
    ):
        """
        Complete learning cycle after a query.
        """
        self._query_count += 1

        # 1. Hebbian: strengthen used paths
        self._hebbian_reinforce(scored_paths, reward, graph)

        # 2. Word-Level Hebbian: strengthen word bigram transitions in WordGraph
        if word_graph is not None and answer_text and reward > 0.5:
            self._reinforce_words(answer_text, reward, word_graph)

        # 3. Record outcomes for NDU training
        self._record_ndu_outcomes(scored_paths, reward, ndu)

        # 4. Record outcomes for Ranker training
        self._record_ranker_outcomes(scored_paths, reward, ranker)

        # 5. Growth: add new knowledge if reward is high
        if reward >= self.config.consolidation_threshold:
            self._grow(query_vec, answer_text, scored_paths, graph, embedder)

        # 6. Periodic abstraction cycle
        trig_interval = getattr(self.config, "abstraction_trigger_interval", 50)
        if abstraction_engine is not None and self._query_count % trig_interval == 0:
            abstraction_engine.run_abstraction_cycle(graph)

        # 7. Periodic retraining
        if self._query_count % self.config.ndu_retrain_interval == 0:
            ndu.train()

        if self._query_count % self.config.ranker_retrain_interval == 0:
            ranker.train()

        # 8. Periodic decay (forgetting)
        if self._query_count % 100 == 0:
            graph.decay(
                factor=self.config.decay_factor,
                min_confidence=0.05,
            )

    def _reinforce_words(self, answer_text: str, reward: float, word_graph):
        """Strengthen word-level transitions in the word graph."""
        import re
        tokens = [t.lower() for t in re.findall(r"\w+", answer_text)]
        boost = 0.5 * reward
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i+1]
            if w1 in word_graph.nodes:
                node = word_graph.nodes[w1]
                node.next_words[w2] = node.next_words.get(w2, 0.0) + boost

    def reinforce(self, node_ids: List[str], graph, boost: float = 0.1):
        """
        Direct reinforcement of specific nodes.
        Used by the Teacher to strengthen learned knowledge.
        """
        for nid in node_ids:
            if nid in graph.nodes:
                graph.nodes[nid].access_count += 1
                graph.nodes[nid].confidence = min(
                    1.0, graph.nodes[nid].confidence + boost
                )

    def _hebbian_reinforce(self, scored_paths: List, reward: float, graph):
        """
        Hebbian rule: strengthen edges and nodes along successful paths.
        'Neurons that fire together wire together.'
        """
        if reward <= 0:
            return

        boost = self.config.hebbian_lr * reward

        for sp in scored_paths:
            path = sp.path if hasattr(sp, "path") else sp
            edges = path.edges if hasattr(path, "edges") else []
            node_ids = path.node_ids if hasattr(path, "node_ids") else []

            # Strengthen nodes
            for nid in node_ids:
                if nid in graph.nodes:
                    graph.nodes[nid].access_count += 1
                    graph.nodes[nid].confidence = min(
                        1.0, graph.nodes[nid].confidence + boost * 0.5
                    )

            # Strengthen edges
            for src, rel, tgt in edges:
                edge_id = f"{src}->{tgt}:{rel}"
                if edge_id in graph.edges:
                    graph.edges[edge_id].confidence = min(
                        1.0, graph.edges[edge_id].confidence + boost
                    )

    def _record_ndu_outcomes(self, scored_paths: List, reward: float, ndu):
        """Record feature→outcome pairs for NDU training."""
        for sp in scored_paths:
            if hasattr(sp, "features") and sp.features is not None:
                # Use path-level features as proxy for individual NDU decisions
                ndu.record_outcome(sp.features[:ndu.config.ndu_num_features], reward > 0.5)

    def _record_ranker_outcomes(self, scored_paths: List, reward: float, ranker):
        """Record path features → reward for ranker training."""
        for sp in scored_paths:
            if hasattr(sp, "features") and sp.features is not None:
                ranker.record_outcome(sp.features, reward)

    def _grow(
        self,
        query_vec: Optional[np.ndarray],
        answer_text: Optional[str],
        scored_paths: List,
        graph,
        embedder,
    ):
        """
        Grow the graph with new knowledge from a successful query.
        Only grows when reward > consolidation threshold.
        """
        if query_vec is None or answer_text is None or embedder is None:
            return

        # Check for near-duplicates before adding
        if graph.num_nodes > 0:
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                q_normed = query_vec / norm
                scores, indices = graph.faiss_index.search(
                    q_normed.reshape(1, -1).astype(np.float32), 1
                )
                if scores[0][0] > 0.92:  # Very similar node already exists
                    return

        # Add answer as a new fact node
        try:
            answer_vec = embedder.encode_single(answer_text)
            node = graph.add_node(
                vector=answer_vec,
                source_text=answer_text[:300],
                level=0,  # FACT
                sector="general",
                node_type="text",
                provenance="cortex_growth",
                confidence=0.7,
                metadata={"learned": True},
            )

            # Connect to the most relevant seed node
            if scored_paths:
                sp = scored_paths[0]
                path = sp.path if hasattr(sp, "path") else sp
                node_ids = path.node_ids if hasattr(path, "node_ids") else []
                if node_ids and node_ids[0] in graph.nodes:
                    try:
                        graph.add_edge(
                            source_id=node_ids[0],
                            target_id=node.id,
                            relation_type="answered-by",
                            confidence=0.6,
                            provenance="cortex_growth",
                        )
                    except ValueError:
                        pass
        except Exception:
            pass

    @property
    def query_count(self) -> int:
        return self._query_count
