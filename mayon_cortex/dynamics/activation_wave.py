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
Activation Wave — Parallel Neural Cascade
===========================================
The brain's spreading activation. When a stimulus arrives, it doesn't
process nodes one-by-one — it fires a wavefront of activation that
spreads through connected neurons in parallel.

Each active node runs its NDU (Node Decision Unit) independently to
decide: follow this edge? halt here? backtrack?

This replaces the sequential Transformer attention mechanism with
O(1) local decisions at each node, running in parallel across the graph.

Usage:
    wave = ActivationWave(config)
    result = wave.activate(query_vec, graph, ndu)
    # result.paths = all discovered proof chains
    # result.active_nodes = all nodes that fired
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.core.ndu import NodeDecisionUnit, NDUDecision


@dataclass
class ActivatedPath:
    """A single path discovered during wavefront expansion."""
    node_ids: List[str]                    # ordered node IDs in path
    edges: List[Tuple[str, str, str]]      # (source_id, relation_type, target_id)
    confidence: float                       # cumulative path confidence
    relevance: float                        # NDU relevance score at endpoint
    hop_count: int                          # number of hops
    scores: List[float] = field(default_factory=list)  # NDU scores per hop


@dataclass
class ActivationResult:
    """Result of a complete wavefront activation."""
    paths: List[ActivatedPath]              # all discovered proof chains
    active_node_ids: Set[str]               # all nodes that fired
    total_nodes_evaluated: int              # how many nodes the NDU evaluated
    seed_node_ids: List[str]                # initial FAISS seed nodes
    hops_used: int                          # actual hops before convergence


class ActivationWave:
    """
    Parallel wavefront expansion through the knowledge graph.

    Algorithm:
    1. FAISS retrieval → seed nodes (like V1 cortex recognizing the stimulus)
    2. For each seed, expand outward:
       - At each node, NDU decides: follow / halt / backtrack
       - All active nodes fire simultaneously (parallel wavefront)
    3. Collect all activated paths as proof chain candidates
    4. Respect budget: max_active_nodes, max_hops

    Supports optional sector filtering for cross-domain activation waves.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def activate(
        self,
        query_vec: np.ndarray,
        graph,       # MayonGraph
        ndu: NodeDecisionUnit,
        sector_filter: Optional[str] = None,
        max_hops: Optional[int] = None,
        seed_node_ids: Optional[List[str]] = None,
    ) -> ActivationResult:
        """
        Run parallel wavefront activation through the graph.

        Args:
            query_vec: Query concept vector (ℝ³⁸⁴)
            graph: MayonGraph instance
            ndu: Node Decision Unit for local decisions
            sector_filter: Optional sector to filter nodes (for cross-domain)
            max_hops: Override max hops (default from config)
            seed_node_ids: Override seed nodes (default from FAISS)

        Returns:
            ActivationResult with all discovered paths and metadata
        """
        max_hops = max_hops or self.config.wave_max_hops
        now = time.time()

        # Step 1: Get seed nodes via FAISS
        if seed_node_ids is None:
            seed_node_ids = self._get_seeds(query_vec, graph, sector_filter)

        if not seed_node_ids:
            return ActivationResult(
                paths=[], active_node_ids=set(),
                total_nodes_evaluated=0, seed_node_ids=[],
                hops_used=0,
            )

        # Determine query sector for feature extraction
        query_sector = self._detect_query_sector(query_vec, graph, seed_node_ids)

        # Compute max degree for normalization
        max_degree = max(
            (len(graph._out_edges.get(nid, [])) for nid in graph.nodes),
            default=1,
        )

        # Step 2: Wavefront expansion
        all_paths: List[ActivatedPath] = []
        active_set: Set[str] = set()
        total_evaluated = 0

        # Initialize frontiers: each seed starts its own path
        # frontier = list of (current_node_id, path_so_far, cumulative_confidence)
        frontiers: List[Tuple[str, List[str], List[Tuple[str, str, str]], float, List[float]]] = []
        for seed_id in seed_node_ids:
            frontiers.append((seed_id, [seed_id], [], 1.0, []))
            active_set.add(seed_id)

        actual_hops = 0
        for hop in range(max_hops):
            if not frontiers:
                break

            next_frontiers = []
            actual_hops = hop + 1

            for node_id, path_nodes, path_edges, path_conf, path_scores in frontiers:
                if node_id not in graph.nodes:
                    continue

                node = graph.nodes[node_id]

                # Get connected edges (both outgoing and incoming for bidirectional graph reasoning)
                candidate_transitions = []
                for edge_id in graph._out_edges.get(node_id, []):
                    if edge_id in graph.edges:
                        edge = graph.edges[edge_id]
                        if edge.is_active and edge.target_id in graph.nodes:
                            candidate_transitions.append((edge.target_id, edge, (node_id, edge.relation_type, edge.target_id)))

                for edge_id in graph._in_edges.get(node_id, []):
                    if edge_id in graph.edges:
                        edge = graph.edges[edge_id]
                        if edge.is_active and edge.source_id in graph.nodes:
                            candidate_transitions.append((edge.source_id, edge, (edge.source_id, edge.relation_type, node_id)))

                if not candidate_transitions:
                    # Dead end — record this path
                    if len(path_edges) > 0:
                        all_paths.append(ActivatedPath(
                            node_ids=list(path_nodes),
                            edges=list(path_edges),
                            confidence=path_conf,
                            relevance=path_scores[-1] if path_scores else 0.0,
                            hop_count=len(path_edges),
                            scores=list(path_scores),
                        ))
                    continue

                for target_id, edge, edge_tuple in candidate_transitions:
                    target_node = graph.nodes[target_id]

                    # Sector filter
                    if sector_filter and sector_filter.lower() not in ("all", "any"):
                        if target_node.sector != sector_filter.lower() and target_node.sector != "general":
                            continue

                    # Avoid cycles
                    if target_id in path_nodes:
                        continue

                    # NDU decision
                    neighbor_relevance = self._compute_neighbor_relevance(
                        query_vec, target_id, graph,
                    )

                    features = ndu.extract_features(
                        query_vec=query_vec,
                        node_vec=target_node.vector,
                        edge_confidence=edge.confidence,
                        relation_type=edge.relation_type,
                        node_level=int(target_node.level),
                        node_sector=target_node.sector,
                        query_sector=query_sector,
                        access_count=target_node.access_count,
                        hop_depth=hop + 1,
                        max_hops=max_hops,
                        out_degree=len(graph._out_edges.get(target_id, [])),
                        max_degree=max_degree,
                        node_timestamp=target_node.timestamp,
                        current_time=now,
                        path_confidence=path_conf * edge.confidence,
                        is_phrase_node=(target_node.node_type == "phrase"),
                        neighbor_relevance=neighbor_relevance,
                        is_visited=(target_id in active_set),
                        is_dead_end=(len(graph._out_edges.get(target_id, [])) == 0),
                    )

                    decision = ndu.predict(features)
                    total_evaluated += 1

                    # Follow decision
                    if decision.follow_prob >= self.config.ndu_follow_threshold:
                        new_path_nodes = path_nodes + [target_id]
                        new_path_edges = path_edges + [edge_tuple]
                        new_conf = path_conf * edge.confidence
                        new_scores = path_scores + [decision.relevance_score]

                        active_set.add(target_id)

                        # Record discovered path
                        all_paths.append(ActivatedPath(
                            node_ids=new_path_nodes,
                            edges=new_path_edges,
                            confidence=new_conf,
                            relevance=decision.relevance_score,
                            hop_count=len(new_path_edges),
                            scores=new_scores,
                        ))

                        # Continue expanding if not explicitly halted
                        if decision.halt_prob < self.config.wave_halt_threshold:
                            next_frontiers.append((
                                target_id, new_path_nodes, new_path_edges,
                                new_conf, new_scores,
                            ))

                    # Backtrack — don't follow, record current path if non-empty
                    elif decision.backtrack_prob > 0.5 and len(path_edges) > 0:
                        all_paths.append(ActivatedPath(
                            node_ids=list(path_nodes),
                            edges=list(path_edges),
                            confidence=path_conf,
                            relevance=decision.relevance_score,
                            hop_count=len(path_edges),
                            scores=list(path_scores),
                        ))

                # Budget check
                if len(next_frontiers) > self.config.wave_max_active:
                    # Keep only the most promising frontiers (by path confidence)
                    next_frontiers.sort(key=lambda x: x[3], reverse=True)
                    next_frontiers = next_frontiers[:self.config.wave_max_active]

            frontiers = next_frontiers

        # Record any remaining active frontiers as paths
        for node_id, path_nodes, path_edges, path_conf, path_scores in frontiers:
            if path_edges:
                all_paths.append(ActivatedPath(
                    node_ids=path_nodes,
                    edges=path_edges,
                    confidence=path_conf,
                    relevance=path_scores[-1] if path_scores else 0.0,
                    hop_count=len(path_edges),
                    scores=path_scores,
                ))

        return ActivationResult(
            paths=all_paths,
            active_node_ids=active_set,
            total_nodes_evaluated=total_evaluated,
            seed_node_ids=seed_node_ids,
            hops_used=actual_hops,
        )

    def _get_seeds(
        self, query_vec: np.ndarray, graph, sector_filter: Optional[str]
    ) -> List[str]:
        """Get seed nodes via FAISS ANN search."""
        if graph.num_nodes == 0:
            return []

        query_vec = query_vec.astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(self.config.wave_seed_top_k * 3, graph.num_nodes)
        scores, indices = graph.faiss_index.search(query_vec.reshape(1, -1), k)
        scores, indices = scores[0], indices[0]

        seeds = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            node_id = graph._faiss_idx_to_id.get(int(idx))
            if node_id is None or node_id not in graph.nodes:
                continue

            node = graph.nodes[node_id]

            # Sector filter
            if sector_filter and sector_filter.lower() not in ("all", "any"):
                if node.sector != sector_filter.lower() and node.sector != "general":
                    continue

            seeds.append(node_id)
            if len(seeds) >= self.config.wave_seed_top_k:
                break

        return seeds

    def _detect_query_sector(
        self, query_vec: np.ndarray, graph, seed_ids: List[str]
    ) -> str:
        """Detect the most likely sector for the query based on seed nodes."""
        sector_counts: Dict[str, int] = {}
        for sid in seed_ids:
            if sid in graph.nodes:
                sec = graph.nodes[sid].sector
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        if sector_counts:
            return max(sector_counts, key=sector_counts.get)
        return "general"

    def _compute_neighbor_relevance(
        self, query_vec: np.ndarray, node_id: str, graph
    ) -> float:
        """Compute mean cosine similarity of a node's neighbors to the query."""
        neighbor_ids = graph.get_successors(node_id)
        if not neighbor_ids:
            return 0.0

        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return 0.0

        sims = []
        for nid in neighbor_ids[:5]:  # Cap at 5 neighbors for efficiency
            if nid in graph.nodes:
                nvec = graph.nodes[nid].vector
                n_norm = np.linalg.norm(nvec)
                if n_norm > 0:
                    sims.append(float(np.dot(query_vec, nvec) / (q_norm * n_norm)))

        return float(np.mean(sims)) if sims else 0.0
