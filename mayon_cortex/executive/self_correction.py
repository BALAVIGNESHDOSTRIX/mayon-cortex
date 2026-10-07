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
Self-Correction — The Brain's Error Monitor
=============================================
Mid-reasoning contradiction detection and backtracking.

When the brain notices "wait, this contradicts what I found earlier",
it backtracks and picks the consistent subset. This module implements
the anterior cingulate cortex — the brain's error detection system.

Usage:
    corrector = SelfCorrector(config)
    clean_paths = corrector.correct(all_paths, graph)
    # Removes contradictory paths, keeps consistent subset
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from mayon_cortex.core.config import CortexConfig, OPPOSING_RELATIONS


@dataclass
class Contradiction:
    """A detected contradiction between two evidence paths."""
    path_a_index: int
    path_b_index: int
    relation_a: str        # e.g. "treats"
    relation_b: str        # e.g. "contraindicates"
    entity: str            # the node where contradiction occurs
    severity: float        # 0-1, how severe the contradiction is
    resolution: str        # which path was kept and why


class SelfCorrector:
    """
    Detects contradictions in activated evidence and resolves them.

    Strategy:
    1. Scan all path pairs for opposing relations on same entities
    2. Flag contradictions with severity scores
    3. Resolve by keeping the higher-confidence path
    4. If safety-critical, ALWAYS keep the cautionary path

    Brain analogy: The anterior cingulate cortex monitors for errors
    and conflicts. When it detects "this doesn't make sense", it triggers
    re-evaluation and course correction.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.opposing = dict(OPPOSING_RELATIONS)
        # Build reverse mapping too
        self._all_opposing = {}
        for k, v in OPPOSING_RELATIONS.items():
            self._all_opposing[k] = v
            self._all_opposing[v] = k

    def detect_contradictions(
        self,
        paths: List,
        graph=None,
    ) -> List[Contradiction]:
        """
        Scan all path pairs for contradictions.

        A contradiction occurs when:
        - Two paths share a common node (same entity discussed)
        - One path has relation R1 and another has R2 where R1 opposes R2
          (e.g. "treats" vs "contraindicates")
        """
        contradictions = []

        for i, path_a in enumerate(paths):
            for j, path_b in enumerate(paths):
                if j <= i:
                    continue

                # Extract (source, relation, target) triples from each path
                edges_a = self._extract_edges(path_a)
                edges_b = self._extract_edges(path_b)

                for src_a, rel_a, tgt_a in edges_a:
                    for src_b, rel_b, tgt_b in edges_b:
                        # Check if they discuss the same entity pair
                        same_pair = (
                            (src_a == src_b and tgt_a == tgt_b) or
                            (src_a == tgt_b and tgt_a == src_b)
                        )
                        if not same_pair:
                            continue

                        # Check if relations are opposing
                        if self._are_opposing(rel_a, rel_b):
                            entity = src_a if graph is None else self._get_text(src_a, graph)
                            contradictions.append(Contradiction(
                                path_a_index=i,
                                path_b_index=j,
                                relation_a=rel_a,
                                relation_b=rel_b,
                                entity=entity,
                                severity=self._compute_severity(rel_a, rel_b),
                                resolution="",
                            ))

        return contradictions

    def correct(
        self,
        paths: List,
        graph=None,
    ) -> Tuple[List, List[Contradiction]]:
        """
        Detect contradictions and remove conflicting paths.

        Resolution strategy:
        1. If one path is safety-critical (contraindicates, causes),
           ALWAYS keep the cautionary path (better safe than sorry)
        2. Otherwise, keep the higher-confidence path
        3. Return the clean (non-contradictory) set of paths

        Returns:
            (clean_paths, detected_contradictions)
        """
        if len(paths) < 2:
            return paths, []

        contradictions = self.detect_contradictions(paths, graph)
        if not contradictions:
            return paths, []

        # Determine which paths to remove
        paths_to_remove: Set[int] = set()

        for contradiction in contradictions:
            idx_a = contradiction.path_a_index
            idx_b = contradiction.path_b_index

            if idx_a in paths_to_remove or idx_b in paths_to_remove:
                continue  # Already handled

            # Safety-critical: always keep the cautionary one
            safety_rels = {"contraindicates", "causes", "prevents", "prohibits", "risk-factor-for"}
            a_is_safety = contradiction.relation_a in safety_rels
            b_is_safety = contradiction.relation_b in safety_rels

            if a_is_safety and not b_is_safety:
                paths_to_remove.add(idx_b)
                contradiction.resolution = f"Kept cautionary path ({contradiction.relation_a}), removed optimistic path ({contradiction.relation_b})"
            elif b_is_safety and not a_is_safety:
                paths_to_remove.add(idx_a)
                contradiction.resolution = f"Kept cautionary path ({contradiction.relation_b}), removed optimistic path ({contradiction.relation_a})"
            else:
                # Both safety or neither — keep higher confidence
                conf_a = self._get_path_confidence(paths[idx_a])
                conf_b = self._get_path_confidence(paths[idx_b])
                if conf_a >= conf_b:
                    paths_to_remove.add(idx_b)
                    contradiction.resolution = f"Kept higher-confidence path A ({conf_a:.2f} vs {conf_b:.2f})"
                else:
                    paths_to_remove.add(idx_a)
                    contradiction.resolution = f"Kept higher-confidence path B ({conf_b:.2f} vs {conf_a:.2f})"

        clean_paths = [p for i, p in enumerate(paths) if i not in paths_to_remove]
        return clean_paths, contradictions

    def _are_opposing(self, rel_a: str, rel_b: str) -> bool:
        """Check if two relations are opposing."""
        a_lower = rel_a.lower()
        b_lower = rel_b.lower()
        return (
            self._all_opposing.get(a_lower) == b_lower or
            self._all_opposing.get(b_lower) == a_lower
        )

    def _compute_severity(self, rel_a: str, rel_b: str) -> float:
        """Compute contradiction severity (0-1). Safety contradictions are highest."""
        safety_rels = {"contraindicates", "causes", "prevents", "prohibits"}
        if rel_a in safety_rels or rel_b in safety_rels:
            return 1.0
        elif rel_a in {"implies", "contradicts"} or rel_b in {"implies", "contradicts"}:
            return 0.8
        else:
            return 0.5

    def _extract_edges(self, path) -> List[Tuple[str, str, str]]:
        """Extract (source, relation, target) triples from a path object."""
        edges = []
        if hasattr(path, "edges"):
            for edge in path.edges:
                if hasattr(edge, "source_id"):
                    edges.append((edge.source_id, edge.relation_type, edge.target_id))
                elif isinstance(edge, tuple) and len(edge) >= 3:
                    edges.append((edge[0], edge[1], edge[2]))
        elif hasattr(path, "traversed_edges"):
            edges = list(path.traversed_edges)
        return edges

    def _get_path_confidence(self, path) -> float:
        """Get overall confidence of a path."""
        if hasattr(path, "confidence"):
            return path.confidence
        if hasattr(path, "score"):
            return path.score
        return 0.5

    def _get_text(self, node_id: str, graph) -> str:
        """Get human-readable text for a node."""
        if graph and hasattr(graph, "nodes") and node_id in graph.nodes:
            return graph.nodes[node_id].source_text[:50]
        return node_id
