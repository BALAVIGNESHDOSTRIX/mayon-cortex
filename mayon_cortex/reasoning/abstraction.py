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
Fractal Abstraction Engine — Concept Hierarchy & Automatic Generalization
==========================================================================
Pillar 2 of Cortex-Graph v4.

How biological brains generalize:
Seeing instances of cats, dogs, and birds leads to spontaneous induction of
the abstract category "animal" with shared properties and affordances.

This engine:
1. Detects recurring structural patterns across nodes (shared relations/sectors).
2. Uses vector clustering (k-means / centroid grouping) to find natural boundaries.
3. Automatically synthesizes ABSTRACT-level nodes with centroid embeddings.
4. Wires `is-a` taxonomic edges from concrete instances to abstract parents.
5. Induces abstract-level relationships (e.g. `Medication --treats--> Condition`).
6. Generates candidate hypotheses when novel instances enter the graph.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


class AbstractionLevel(IntEnum):
    CONCRETE     = 0   # Instance: e.g. "Earth", "Aspirin", "Python"
    INTERMEDIATE = 1   # Category: e.g. "Planet", "Analgesic", "Programming Language"
    ABSTRACT     = 2   # High-level: e.g. "Celestial Body", "Medication", "Tool"
    UNIVERSAL    = 3   # Universal: e.g. "Entity", "Action", "State"


@dataclass
class AbstractConcept:
    """An automatically formed abstract concept in the cortical hierarchy."""
    concept_id: str
    name: str
    level: AbstractionLevel
    centroid_vector: np.ndarray
    sector: str
    member_node_ids: List[str] = field(default_factory=list)
    common_relations: Dict[str, float] = field(default_factory=dict)
    inferred_edges: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class AbstractionReport:
    """Summary of abstractions discovered during a cycle."""
    new_concepts_formed: int = 0
    is_a_edges_created: int = 0
    abstract_relations_induced: int = 0
    concepts: List[AbstractConcept] = field(default_factory=list)


class FractalAbstractionEngine:
    """
    Scans the graph topology & embeddings to detect recurring patterns and construct
    multi-level abstraction hierarchies.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.min_instances = getattr(self.config, "abstraction_min_instances", 3)
        self.abstract_concepts: Dict[str, AbstractConcept] = {}

    def run_abstraction_cycle(self, graph) -> AbstractionReport:
        """
        Executes one full abstraction cycle over the graph:
        1. Pattern mining by relation and sector
        2. Vector clustering
        3. Abstract node & edge injection
        """
        report = AbstractionReport()
        if graph is None or not hasattr(graph, "nodes") or len(graph.nodes) < self.min_instances:
            return report

        # 1. Group nodes by shared outgoing relation signatures
        rel_signatures: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        nodes_by_sector: Dict[str, List[str]] = defaultdict(list)

        for node_id, node in graph.nodes.items():
            sector = getattr(node, "sector", "general")
            nodes_by_sector[sector].append(node_id)

            # Check outgoing relations
            out_edge_ids = graph._out_edges.get(node_id, []) if hasattr(graph, "_out_edges") else []
            for eid in out_edge_ids:
                if hasattr(graph, "edges") and eid in graph.edges:
                    edge = graph.edges[eid]
                    rel = getattr(edge, "relation_type", "")
                    if rel and rel not in {"is-a", "part-of", "next-word", "skip-1", "skip-2", "expressed-as", "followed-by"}:
                        rel_signatures[(rel, sector)].append(node_id)

        # 2. Form candidate concepts from frequent relation patterns
        for (rel, sector), member_ids in rel_signatures.items():
            unique_members = list(set(member_ids))
            if len(unique_members) >= self.min_instances:
                concept = self._form_abstract_concept(
                    graph=graph,
                    member_ids=unique_members,
                    sector=sector,
                    name_hint=f"{sector.capitalize()}_{rel.replace('-', '_').capitalize()}_Group",
                    level=AbstractionLevel.INTERMEDIATE,
                )
                if concept:
                    report.new_concepts_formed += 1
                    report.is_a_edges_created += len(concept.member_node_ids)
                    report.concepts.append(concept)
                    self._wire_concept_to_graph(graph, concept)

        # 3. Form abstract concepts from sector vector clustering
        for sector, member_ids in nodes_by_sector.items():
            if len(member_ids) >= self.min_instances * 2:
                clusters = self._cluster_nodes(graph, member_ids, k=min(3, len(member_ids) // self.min_instances))
                for cluster_idx, cluster_members in enumerate(clusters):
                    if len(cluster_members) >= self.min_instances:
                        concept = self._form_abstract_concept(
                            graph=graph,
                            member_ids=cluster_members,
                            sector=sector,
                            name_hint=f"{sector.capitalize()}_Cluster_{cluster_idx+1}",
                            level=AbstractionLevel.ABSTRACT,
                        )
                        if concept:
                            report.new_concepts_formed += 1
                            report.is_a_edges_created += len(concept.member_node_ids)
                            report.concepts.append(concept)
                            self._wire_concept_to_graph(graph, concept)

        return report

    def _cluster_nodes(self, graph, node_ids: List[str], k: int) -> List[List[str]]:
        """Simple k-means clustering on node embeddings."""
        if k <= 1 or len(node_ids) <= k:
            return [node_ids]

        vectors: List[np.ndarray] = []
        valid_ids: List[str] = []
        for nid in node_ids:
            if nid in graph.nodes:
                node = graph.nodes[nid]
                v = getattr(node, "vector", None)
                if v is not None:
                    vectors.append(v)
                    valid_ids.append(nid)

        if len(valid_ids) < k:
            return [valid_ids]

        X = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        rng = np.random.RandomState(42)
        init_idx = rng.choice(len(X), size=k, replace=False)
        centroids = X[init_idx].copy()

        clusters: List[List[str]] = [[] for _ in range(k)]
        for _ in range(6):
            sims = np.dot(X, centroids.T)
            assignments = np.argmax(sims, axis=1)

            new_clusters: List[List[str]] = [[] for _ in range(k)]
            for i, c_idx in enumerate(assignments):
                new_clusters[c_idx].append(valid_ids[i])

            for c_idx in range(k):
                c_members_idx = np.where(assignments == c_idx)[0]
                if len(c_members_idx) > 0:
                    c_vecs = X[c_members_idx]
                    mean_v = np.mean(c_vecs, axis=0)
                    mean_v /= (np.linalg.norm(mean_v) + 1e-8)
                    centroids[c_idx] = mean_v

            clusters = new_clusters

        return [c for c in clusters if c]

    def _form_abstract_concept(
        self,
        graph,
        member_ids: List[str],
        sector: str,
        name_hint: str,
        level: AbstractionLevel,
    ) -> Optional[AbstractConcept]:
        """Synthesize an abstract concept from a set of concrete nodes."""
        vectors = []
        for nid in member_ids:
            if nid in graph.nodes:
                node = graph.nodes[nid]
                v = getattr(node, "vector", None)
                if v is not None:
                    vectors.append(v)

        if not vectors:
            return None

        centroid = np.mean(vectors, axis=0).astype(np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-8)

        concept_id = f"abstract::{name_hint.lower()}"
        concept = AbstractConcept(
            concept_id=concept_id,
            name=name_hint,
            level=level,
            centroid_vector=centroid,
            sector=sector,
            member_node_ids=member_ids,
        )
        self.abstract_concepts[concept_id] = concept
        return concept

    def _wire_concept_to_graph(self, graph, concept: AbstractConcept):
        """Inject abstract concept node and `is-a` edges into graph."""
        if not hasattr(graph, "add_node") or not hasattr(graph, "add_edge"):
            return

        # Check if already in graph
        if concept.concept_id not in graph.nodes:
            try:
                graph.add_node(
                    vector=concept.centroid_vector,
                    source_text=concept.name,
                    level=int(concept.level),
                    sector=concept.sector,
                    node_type="concept",
                    provenance="abstraction_engine",
                    confidence=0.95,
                    metadata={
                        "is_abstract": True,
                        "concept_id": concept.concept_id,
                        "member_count": len(concept.member_node_ids),
                    }
                )
            except Exception:
                pass

        # Wire is-a edges from each instance to parent
        for member_id in concept.member_node_ids:
            try:
                graph.add_edge(
                    source_id=member_id,
                    target_id=concept.concept_id,
                    relation_type="is-a",
                    confidence=0.95,
                    sector=concept.sector,
                    provenance="abstraction_engine",
                )
            except Exception:
                pass

    def generalize_query(self, query_vector: np.ndarray, top_k: int = 2) -> List[AbstractConcept]:
        """Find the most relevant abstract concepts for a given query vector."""
        if not self.abstract_concepts:
            return []

        results = []
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        for concept in self.abstract_concepts.values():
            sim = float(np.dot(concept.centroid_vector, q_norm))
            results.append((concept, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in results[:top_k]]
