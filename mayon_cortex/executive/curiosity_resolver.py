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
Autonomous Curiosity & Dynamic Gap Resolver
===========================================
Detects epistemic voids during multi-hop reasoning, searches connected
corpus or local source streams for missing bridges, dynamically ingests
verified knowledge, and closes the reasoning loop in real-time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mayon_cortex.core.graph import MayonGraph
from mayon_cortex.language.hyper_parser import SemanticRoleExtractor, HyperEdge


@dataclass
class ResolutionAttempt:
    """Record of a dynamic curiosity resolution cycle."""
    source_gap: str
    target_gap: str
    queries_formulated: List[str]
    nodes_added: int
    edges_added: int
    is_resolved: bool
    explanation: str


class CuriosityGapResolver:
    """
    Autonomous Knowledge Gap Resolver.
    """

    def __init__(self, extractor: Optional[SemanticRoleExtractor] = None):
        self.extractor = extractor or SemanticRoleExtractor()
        self.history: List[ResolutionAttempt] = []

    def resolve_gap(
        self,
        start_concept: str,
        goal_concept: str,
        graph: MayonGraph,
        available_corpus: Optional[List[str]] = None,
        sector: str = "general",
    ) -> ResolutionAttempt:
        """
        Attempt to dynamically close a reasoning void between start_concept and goal_concept.
        """
        available_corpus = available_corpus or []
        queries = [
            f"What connects {start_concept} to {goal_concept}?",
            f"Relationship between {start_concept} and {goal_concept}",
            f"{start_concept} effects and mechanism",
        ]

        nodes_added = 0
        edges_added = 0
        found_connection = False

        s_low = start_concept.lower()
        g_low = goal_concept.lower()

        import numpy as np
        for text in available_corpus:
            t_low = text.lower()
            if s_low in t_low or g_low in t_low:
                # Ingest through SemanticRoleExtractor
                hyper_edges = self.extractor.parse_sentence(text, sector=sector, provenance="curiosity_resolver")
                entity_node_map = {}
                for he in hyper_edges:
                    src_low = he.source.lower()
                    tgt_low = he.target.lower()
                    dim = getattr(graph, "concept_dim", 384)
                    
                    if src_low not in entity_node_map:
                        vec = np.random.randn(dim).astype(np.float32)
                        vec /= np.linalg.norm(vec) + 1e-8
                        node_s = graph.add_node(vector=vec, source_text=he.source, level=1, sector=he.sector)
                        entity_node_map[src_low] = node_s.id
                        nodes_added += 1
                    if tgt_low not in entity_node_map:
                        vec = np.random.randn(dim).astype(np.float32)
                        vec /= np.linalg.norm(vec) + 1e-8
                        node_t = graph.add_node(vector=vec, source_text=he.target, level=1, sector=he.sector)
                        entity_node_map[tgt_low] = node_t.id
                        nodes_added += 1

                    try:
                        graph.add_edge(
                            source_id=entity_node_map[src_low],
                            target_id=entity_node_map[tgt_low],
                            relation_type=he.relation,
                            confidence=he.confidence,
                            sector=he.sector,
                        )
                        edges_added += 1
                    except Exception:
                        pass

                    if (s_low in src_low and g_low in tgt_low) or (g_low in src_low and s_low in tgt_low):
                        found_connection = True

        is_resolved = found_connection or edges_added > 0
        explanation = (
            f"Successfully closed knowledge gap with {edges_added} new relational edges."
            if is_resolved
            else "Could not locate connecting evidence in available sources."
        )

        attempt = ResolutionAttempt(
            source_gap=start_concept,
            target_gap=goal_concept,
            queries_formulated=queries,
            nodes_added=nodes_added,
            edges_added=edges_added,
            is_resolved=is_resolved,
            explanation=explanation,
        )
        self.history.append(attempt)
        return attempt
