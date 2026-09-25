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
Analogical Reasoning Engine — Subgraph Isomorphism & Vector Analogy
===================================================================
Pillar 5 of Cortex-Graph v4.

"A is to B as C is to ?"
Analogical reasoning is what separates genuine conceptual understanding from
rote pattern matching.

This engine operates via dual mechanisms:
1. Structural Isomorphism: Identifies the typed graph relations connecting A to B,
   and searches for an identical relational structure originating from C.
2. Vector-Space Analogy: Calculates relational vector displacement (B - A)
   and searches for candidates matching (C + B - A) in ℝ³⁸⁴.
3. Cross-Domain Mapping: Maps structural roles across different sectors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class AnalogyResult:
    """The result of an analogical query."""
    source_a: str
    source_b: str
    target_c: str
    predicted_d: str
    relation_type: str
    confidence: float
    structural_match: bool
    vector_similarity: float
    explanation: str


class AnalogyEngine:
    """
    Solves A:B :: C:? queries using structural graph search + vector displacement.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def solve(
        self,
        a: str,
        b: str,
        c: str,
        graph,
        sector_hint: Optional[str] = None,
    ) -> Optional[AnalogyResult]:
        """
        Solves the analogy: A is to B as C is to [D].
        Example: solve('king', 'kingdom', 'captain', graph) -> 'ship'
        """
        if graph is None or not hasattr(graph, "nodes") or not graph.nodes:
            return None

        # Resolve entity nodes in graph
        a_id = self._resolve_node_id(a, graph)
        b_id = self._resolve_node_id(b, graph)
        c_id = self._resolve_node_id(c, graph)

        if not a_id or not b_id or not c_id:
            return None

        # Step 1: Discover relation between A and B
        rel_type, direct_edge = self._find_relation(a_id, b_id, graph)

        # Retrieve vectors
        vec_a = self._get_vector(a_id, graph)
        vec_b = self._get_vector(b_id, graph)
        vec_c = self._get_vector(c_id, graph)

        target_vec = None
        if vec_a is not None and vec_b is not None and vec_c is not None:
            displacement = vec_b - vec_a
            target_vec = vec_c + displacement
            target_vec /= (np.linalg.norm(target_vec) + 1e-8)

        # Step 2: Structural candidate search from C
        candidates: List[Tuple[str, float, bool]] = []

        if rel_type and hasattr(graph, "_out_edges"):
            for eid in graph._out_edges.get(c_id, []):
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    if edge.relation_type == rel_type and edge.target_id != c_id:
                        candidates.append((edge.target_id, 0.95, True))

            for eid in graph._in_edges.get(c_id, []) if hasattr(graph, "_in_edges") else []:
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    if edge.relation_type == rel_type and edge.source_id != c_id:
                        candidates.append((edge.source_id, 0.90, True))

        # Step 3: Vector search candidates
        if target_vec is not None:
            for nid, node in graph.nodes.items():
                if nid in {a_id, b_id, c_id}:
                    continue
                nvec = getattr(node, "vector", None)
                if nvec is not None:
                    sim = float(np.dot(target_vec, nvec) / (np.linalg.norm(nvec) + 1e-8))
                    if sim > 0.3:
                        candidates.append((nid, sim, False))

        if not candidates:
            return None

        # Step 4: Rank candidates
        scored_candidates: Dict[str, float] = {}
        for nid, score, is_struct in candidates:
            bonus = 0.5 if is_struct else 0.0
            scored_candidates[nid] = scored_candidates.get(nid, 0.0) + score + bonus

        best_id = max(scored_candidates.keys(), key=lambda k: scored_candidates[k])
        best_name = self._get_node_text(best_id, graph)
        final_conf = min(0.99, scored_candidates[best_id] / 1.5)

        is_struct_match = any(nid == best_id and is_s for nid, _, is_s in candidates)
        v_sim = 0.0
        if target_vec is not None:
            best_vec = self._get_vector(best_id, graph)
            if best_vec is not None:
                v_sim = float(np.dot(target_vec, best_vec))

        rel_desc = rel_type if rel_type else "relates-to"
        explanation = (
            f"Because '{a}' {rel_desc} '{b}', similarly '{c}' {rel_desc} '{best_name}' "
            f"(structural_match={is_struct_match}, vector_sim={v_sim:.2f})."
        )

        return AnalogyResult(
            source_a=a,
            source_b=b,
            target_c=c,
            predicted_d=best_name,
            relation_type=rel_desc,
            confidence=final_conf,
            structural_match=is_struct_match,
            vector_similarity=v_sim,
            explanation=explanation,
        )

    def _resolve_node_id(self, name_or_id: str, graph) -> Optional[str]:
        """Find matching node id in graph."""
        clean = name_or_id.lower().strip()
        if clean in graph.nodes:
            return clean

        for nid, node in graph.nodes.items():
            text = getattr(node, "source_text", "").lower().strip()
            if text == clean or clean in text.split():
                return nid
        return None

    def _find_relation(self, a_id: str, b_id: str, graph) -> Tuple[Optional[str], bool]:
        """Find relation between A and B in graph."""
        if hasattr(graph, "_out_edges"):
            for eid in graph._out_edges.get(a_id, []):
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    if edge.target_id == b_id:
                        return edge.relation_type, True

            for eid in graph._out_edges.get(b_id, []):
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    if edge.target_id == a_id:
                        return edge.relation_type, True

        return None, False

    def _get_vector(self, node_id: str, graph) -> Optional[np.ndarray]:
        """Fetch node embedding vector from graph."""
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            v = getattr(node, "vector", None)
            if v is not None:
                norm = np.linalg.norm(v)
                return v / (norm + 1e-8)
        return None

    def _get_node_text(self, node_id: str, graph) -> str:
        """Get display name of node."""
        if node_id in graph.nodes:
            return getattr(graph.nodes[node_id], "source_text", node_id)
        return node_id
