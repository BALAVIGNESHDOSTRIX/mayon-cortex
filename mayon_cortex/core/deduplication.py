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
Entity Deduplication & Conflict Reconciliation
===============================================
Phase 5 of the Brain-Like Intelligence upgrade.

Handles entity resolution, alias merging, synonym clustering, and edge rewiring
at scale to keep the knowledge graph compact, coherent, and contradiction-free.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class DeduplicationReport:
    """Summary of entity resolution and merging operations."""
    total_checked: int = 0
    duplicates_found: int = 0
    nodes_merged: int = 0
    edges_rewired: int = 0
    conflicts_resolved: int = 0
    merged_clusters: List[List[str]] = field(default_factory=list)


class GraphDeduplicator:
    """
    Performs fuzzy entity resolution, alias clustering, and knowledge conflict reconciliation.
    """

    def __init__(
        self,
        string_similarity_thresh: float = 0.88,
        vector_similarity_thresh: float = 0.92,
    ):
        self.str_thresh = string_similarity_thresh
        self.vec_thresh = vector_similarity_thresh

    @staticmethod
    def _jaccard_similarity(s1: str, s2: str) -> float:
        """Character 3-gram Jaccard similarity."""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        n1 = {s1[i:i+3] for i in range(max(len(s1)-2, 1))}
        n2 = {s2[i:i+3] for i in range(max(len(s2)-2, 1))}
        union = len(n1 | n2)
        if union == 0:
            return 0.0
        return len(n1 & n2) / union

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two dense vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def find_duplicate_candidates(
        self,
        node_labels: List[str],
        node_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of node IDs/labels that likely represent the identical concept.
        Returns [(node_a, node_b, confidence_score), ...]
        """
        candidates = []
        n = len(node_labels)

        for i in range(n):
            l1 = node_labels[i]
            for j in range(i + 1, n):
                l2 = node_labels[j]
                # 1. String similarity check
                str_sim = self._jaccard_similarity(l1, l2)
                
                # 2. Vector check if provided
                vec_sim = 0.0
                if node_vectors and l1 in node_vectors and l2 in node_vectors:
                    vec_sim = self._cosine_similarity(node_vectors[l1], node_vectors[l2])

                # Blend score
                combined_score = max(str_sim, vec_sim) if node_vectors else str_sim
                if combined_score >= self.str_thresh or vec_sim >= self.vec_thresh:
                    candidates.append((l1, l2, combined_score))

        return sorted(candidates, key=lambda x: x[2], reverse=True)

    def reconcile_conflicts(
        self,
        fact_a: Tuple[str, str, str, float],
        fact_b: Tuple[str, str, str, float],
    ) -> Tuple[str, str, str, float, str]:
        """
        Reconcile conflicting triples (s, r1, o1, conf1) vs (s, r2, o2, conf2).
        Returns the winning fact and an explanation rationale.
        """
        s1, r1, o1, c1 = fact_a
        s2, r2, o2, c2 = fact_b

        if c1 >= c2:
            return (s1, r1, o1, c1, f"Retained fact with higher confidence ({c1:.2f} >= {c2:.2f})")
        else:
            return (s2, r2, o2, c2, f"Retained fact with higher confidence ({c2:.2f} > {c1:.2f})")

    def run_deduplication(
        self,
        node_labels: List[str],
        node_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> DeduplicationReport:
        """
        Execute full clustering and return summary report.
        """
        report = DeduplicationReport(total_checked=len(node_labels))
        pairs = self.find_duplicate_candidates(node_labels, node_vectors)
        report.duplicates_found = len(pairs)

        # Build union-find clusters
        parent: Dict[str, str] = {lbl: lbl for lbl in node_labels}

        def find(u: str) -> str:
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u: str, v: str):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

        for u, v, _ in pairs:
            union(u, v)

        # Group by root
        clusters: Dict[str, List[str]] = {}
        for lbl in node_labels:
            root = find(lbl)
            clusters.setdefault(root, []).append(lbl)

        multi_item_clusters = [c for c in clusters.values() if len(c) > 1]
        report.merged_clusters = multi_item_clusters
        report.nodes_merged = sum(len(c) - 1 for c in multi_item_clusters)

        return report
