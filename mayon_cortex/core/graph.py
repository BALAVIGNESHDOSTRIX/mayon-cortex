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
Mayon Graph Engine
==================
The core knowledge graph that stores concept nodes and typed relation edges.
Supports hierarchical levels, sector taxonomy, multimodal/image nodes, temporal versioning,
and FAISS-backed ANN search.

This is the "growing" part of Mayon-Net — knowledge accumulates here
while the Nucleus stays frozen.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.vector_index import (
    BaseVectorIndex,
    VectorIndexFactory,
    USearchVectorIndex,
    FaissVectorIndex,
    NumpyVectorIndex,
)


class LightweightGraphView:
    """
    High-performance native adjacency view replacing NetworkX.
    Preserves full API compatibility with NetworkX DiGraph:
    - successors(u)
    - predecessors(u)
    - out_edges(u, data=False)
    - in_edges(u, data=False)
    - has_edge(u, v)
    - add_node(n)
    - add_edge(u, v, **data)
    - remove_edge(u, v)
    - u in graph
    - graph[u][v]
    """

    def __init__(self, mayon_graph: "MayonGraph"):
        self._m = mayon_graph

    def successors(self, node_id: str) -> List[str]:
        return self._m.get_successors(node_id)

    def predecessors(self, node_id: str) -> List[str]:
        return self._m.get_predecessors(node_id)

    def out_edges(self, node_id: str, data: bool = False):
        return self._m.get_out_edges_data(node_id, data=data)

    def in_edges(self, node_id: str, data: bool = False):
        return self._m.get_in_edges_data(node_id, data=data)

    def has_edge(self, u: str, v: str) -> bool:
        return self._m.has_edge(u, v)

    def add_node(self, node_id: str):
        if node_id not in self._m._out_edges:
            self._m._out_edges[node_id] = []
        if node_id not in self._m._in_edges:
            self._m._in_edges[node_id] = []

    def add_edge(self, u: str, v: str, **data):
        edge_id = data.get("edge_id", f"{u}->{v}:{data.get('relation_type', 'connected_to')}")
        if u not in self._m._out_edges:
            self._m._out_edges[u] = []
        if v not in self._m._in_edges:
            self._m._in_edges[v] = []
        if edge_id not in self._m._out_edges[u]:
            self._m._out_edges[u].append(edge_id)
        if edge_id not in self._m._in_edges[v]:
            self._m._in_edges[v].append(edge_id)

    def remove_edge(self, u: str, v: str):
        self._m.remove_edge_between(u, v)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._m.nodes or node_id in self._m._out_edges

    def __getitem__(self, u: str) -> Dict[str, Dict]:
        # Emulate nx graph[u][v] -> edge_data
        result = {}
        for eid in self._m._out_edges.get(u, []):
            if eid in self._m.edges:
                edge = self._m.edges[eid]
                result[edge.target_id] = {
                    "relation_type": edge.relation_type,
                    "confidence": edge.confidence,
                    "edge_id": eid,
                }
        return result




class GraphLevel(IntEnum):
    """Hierarchical levels in the Mayon graph."""
    FACT = 0       # Concrete facts: "list.append() adds an element"
    CONCEPT = 1    # Concepts: "Python list methods"
    ABSTRACT = 2   # Abstract categories: "Data structures"
    PHRASE = 3     # Language phrase patterns for generation


@dataclass
class ConceptNode:
    """A node in the Mayon graph (Text, Image, or Concept)."""
    id: str
    vector: np.ndarray          # Concept embedding (dim=concept_dim)
    level: GraphLevel           # Hierarchical level
    source_text: str            # Original text or caption this node was created from
    timestamp: float            # When this node was created
    provenance: str             # Where this knowledge came from
    sector: str = "general"     # Domain: "general", "code", "medical", "legal", "finance", "science"
    node_type: str = "text"     # "text", "image", "code", "table"
    confidence: float = 1.0     # How confident we are in this node
    access_count: int = 0       # How often this node has been retrieved (reinforcement)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "level": self.level.name,
            "source_text": self.source_text,
            "sector": self.sector,
            "node_type": self.node_type,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class RelationEdge:
    """A typed, directed edge between two concept nodes."""
    source_id: str
    target_id: str
    relation_type: str          # "is-a", "part-of", "calls", "treats", "regulates", "implies", etc.
    confidence: float           # How confident we are in this relation
    provenance: str             # Where this relation came from
    timestamp: float            # When this edge was created
    sector: str = "general"     # Sector classification
    valid_from: Optional[float] = None   # Temporal validity start
    valid_to: Optional[float] = None     # Temporal validity end (None = still valid)
    superseded_by: Optional[str] = None  # If this edge was replaced
    metadata: Dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if this edge is currently valid (not expired or superseded)."""
        now = time.time()
        if self.superseded_by is not None:
            return False
        if self.valid_to is not None and now > self.valid_to:
            return False
        return True


@dataclass
class Subgraph:
    """A retrieved subgraph — the result of a Mayon query."""
    nodes: List[ConceptNode]
    edges: List[RelationEdge]
    query_vector: np.ndarray
    confidence: float           # Overall confidence of this subgraph
    hop_depth: int              # How many hops from the query

    @property
    def node_vectors(self) -> np.ndarray:
        """Stack all node vectors into a matrix for the Binder."""
        if not self.nodes:
            return np.zeros((0, self.query_vector.shape[0]))
        return np.stack([n.vector for n in self.nodes])

    @property
    def adjacency(self) -> np.ndarray:
        """Build adjacency matrix for GNN processing."""
        n = len(self.nodes)
        if n == 0:
            return np.zeros((0, 0))
        id_to_idx = {node.id: i for i, node in enumerate(self.nodes)}
        adj = np.zeros((n, n))
        for edge in self.edges:
            if edge.source_id in id_to_idx and edge.target_id in id_to_idx:
                i, j = id_to_idx[edge.source_id], id_to_idx[edge.target_id]
                adj[i, j] = edge.confidence
        return adj

    @property
    def relation_types(self) -> List[str]:
        """Get all relation types in this subgraph."""
        return list(set(e.relation_type for e in self.edges))

    def provenance_chain(self) -> List[str]:
        """Get the full provenance chain for explainability."""
        provenances = []
        for node in self.nodes:
            provenances.append(f"Node[{node.id[:8]} | {node.sector}]: {node.source_text[:60]} ({node.provenance})")
        for edge in self.edges:
            provenances.append(
                f"Edge[{edge.source_id[:8]}→{edge.target_id[:8]}]: "
                f"{edge.relation_type} ({edge.provenance})"
            )
        return provenances


class MayonGraph:
    """
    The Mayon: an external knowledge graph that grows without bound.

    Key design principles:
    - Knowledge lives here, not in the Nucleus weights.
    - Nodes are concept embeddings from the frozen Perceiver or CLIP vision encoder.
    - Edges are typed relations with confidence, sector, and provenance.
    - Retrieval is ANN search + graph traversal, not attention over tokens.
    - Supports hierarchical levels, multi-sector filtering, and temporal versioning.
    """

    def __init__(self, concept_dim: int = 384, ann_index_type: str = "auto"):
        self.concept_dim = concept_dim
        self.ann_index_type = ann_index_type

        # Core graph
        self._out_edges: Dict[str, List[str]] = {}
        self._in_edges: Dict[str, List[str]] = {}
        self.graph = LightweightGraphView(self)
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[str, RelationEdge] = {}

        # High-performance polymorphic vector index
        self._build_index()

        # ID → index mapping
        self._id_to_faiss_idx: Dict[str, int] = {}
        self._faiss_idx_to_id: Dict[int, str] = {}
        self._next_faiss_idx: int = 0

        # Level-specific indices for hierarchical search
        self.level_indices: Dict[GraphLevel, List[str]] = {
            level: [] for level in GraphLevel
        }

        # Sector indices for scoped domain lookup
        self.sector_indices: Dict[str, List[str]] = {}

    def _build_index(self):
        """Build or rebuild the polymorphic vector index (USearch, FAISS, or NumPy)."""
        self.vector_index: BaseVectorIndex = VectorIndexFactory.create_index(
            dim=self.concept_dim,
            backend=self.ann_index_type,
            metric="cos",
        )

    @property
    def faiss_index(self) -> BaseVectorIndex:
        """Backward-compatibility proxy for vector_index."""
        return self.vector_index

    @faiss_index.setter
    def faiss_index(self, value: BaseVectorIndex):
        self.vector_index = value

    def add_node(
        self,
        vector: np.ndarray,
        source_text: str,
        level: GraphLevel = GraphLevel.FACT,
        sector: str = "general",
        node_type: str = "text",
        provenance: str = "unknown",
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> ConceptNode:
        """
        Add a new concept node to the Mayon (Text or Image).
        """
        node_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Normalize vector for cosine similarity via inner product
        vector = vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Ensure GraphLevel enum
        if isinstance(level, int) and not isinstance(level, GraphLevel):
            level = GraphLevel(level)

        node = ConceptNode(
            id=node_id,
            vector=vector,
            level=level,
            source_text=source_text,
            timestamp=now,
            sector=sector.lower(),
            node_type=node_type,
            provenance=provenance,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Store in all data structures
        self.nodes[node_id] = node
        if node_id not in self._out_edges:
            self._out_edges[node_id] = []
        if node_id not in self._in_edges:
            self._in_edges[node_id] = []
        self.level_indices[level].append(node_id)

        # Update sector index
        sec_key = sector.lower()
        if sec_key not in self.sector_indices:
            self.sector_indices[sec_key] = []
        self.sector_indices[sec_key].append(node_id)

        # Add to Vector Index
        self.vector_index.add(vector.reshape(-1), key=self._next_faiss_idx)
        self._id_to_faiss_idx[node_id] = self._next_faiss_idx
        self._faiss_idx_to_id[self._next_faiss_idx] = node_id
        self._next_faiss_idx += 1

        return node

    def add_nodes_bulk(
        self,
        vectors: np.ndarray,
        source_texts: List[str],
        levels: Optional[List[GraphLevel]] = None,
        sectors: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        provenances: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
        metadata_list: Optional[List[Dict]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[ConceptNode]:
        """
        Add multiple concept nodes in a single high-performance vectorized batch.
        Performs vectorized BLAS normalization and a single batch vector index addition.
        """
        N = len(source_texts)
        if N == 0:
            return []

        if len(vectors) != N:
            raise ValueError(f"Length mismatch: {len(vectors)} vectors vs {N} source_texts")

        # 1. Vectorized normalization in one operation
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized_vectors = vectors / norms

        # 2. Single batch addition to Vector index
        start_faiss_idx = self._next_faiss_idx
        keys = list(range(start_faiss_idx, start_faiss_idx + N))
        self.vector_index.add_batch(normalized_vectors, keys=keys)
        self._next_faiss_idx += N

        # 3. Fast node creation and indexing
        now = time.time()
        created_nodes = []
        for i in range(N):
            nid = node_ids[i] if node_ids and i < len(node_ids) else str(uuid.uuid4())[:12]
            lvl = levels[i] if levels and i < len(levels) else GraphLevel.FACT
            sec = (sectors[i] if sectors and i < len(sectors) else "general").lower()
            ntype = node_types[i] if node_types and i < len(node_types) else "text"
            prov = provenances[i] if provenances and i < len(provenances) else "unknown"
            conf = confidences[i] if confidences and i < len(confidences) else 1.0
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}

            node = ConceptNode(
                id=nid,
                vector=normalized_vectors[i],
                level=lvl,
                source_text=source_texts[i],
                timestamp=now,
                sector=sec,
                node_type=ntype,
                provenance=prov,
                confidence=conf,
                metadata=meta,
            )

            self.nodes[nid] = node
            if nid not in self._out_edges:
                self._out_edges[nid] = []
            if nid not in self._in_edges:
                self._in_edges[nid] = []

            self.level_indices[lvl].append(nid)
            if sec not in self.sector_indices:
                self.sector_indices[sec] = []
            self.sector_indices[sec].append(nid)

            faiss_idx = start_faiss_idx + i
            self._id_to_faiss_idx[nid] = faiss_idx
            self._faiss_idx_to_id[faiss_idx] = nid

            created_nodes.append(node)

        return created_nodes

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float = 1.0,
        sector: str = "general",
        provenance: str = "unknown",
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> RelationEdge:
        """Add a typed relation edge between two concept nodes."""
        if source_id not in self.nodes:
            raise ValueError(f"Source node {source_id} not found in Mayon")
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id} not found in Mayon")

        edge_id = f"{source_id}->{target_id}:{relation_type}"
        now = time.time()

        # Check for existing edge (conflict resolution)
        existing = self._find_edge(source_id, target_id, relation_type)
        if existing is not None:
            return self._resolve_conflict(existing, confidence, provenance)

        edge = RelationEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            sector=sector.lower(),
            provenance=provenance,
            timestamp=now,
            valid_from=valid_from or now,
            valid_to=valid_to,
            metadata=metadata or {},
        )

        self.edges[edge_id] = edge
        if source_id not in self._out_edges:
            self._out_edges[source_id] = []
        if edge_id not in self._out_edges[source_id]:
            self._out_edges[source_id].append(edge_id)

        if target_id not in self._in_edges:
            self._in_edges[target_id] = []
        if edge_id not in self._in_edges[target_id]:
            self._in_edges[target_id].append(edge_id)

        return edge

    def add_edges_bulk(
        self,
        edges_data: List[Dict[str, Any]],
    ) -> List[RelationEdge]:
        """
        Add multiple typed relation edges in batch with conflict resolution.
        """
        now = time.time()
        created_edges = []
        for item in edges_data:
            source_id = item["source_id"]
            target_id = item["target_id"]
            relation_type = item["relation_type"]
            confidence = item.get("confidence", 1.0)
            sector = item.get("sector", "general").lower()
            provenance = item.get("provenance", "unknown")
            valid_from = item.get("valid_from")
            valid_to = item.get("valid_to")
            metadata = item.get("metadata", {})

            if source_id not in self.nodes or target_id not in self.nodes:
                continue

            edge_id = f"{source_id}->{target_id}:{relation_type}"
            existing = self._find_edge(source_id, target_id, relation_type)
            if existing is not None:
                resolved = self._resolve_conflict(existing, confidence, provenance)
                created_edges.append(resolved)
                continue

            edge = RelationEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
                sector=sector,
                provenance=provenance,
                timestamp=now,
                valid_from=valid_from or now,
                valid_to=valid_to,
                metadata=metadata,
            )

            self.edges[edge_id] = edge
            if source_id not in self._out_edges:
                self._out_edges[source_id] = []
            if edge_id not in self._out_edges[source_id]:
                self._out_edges[source_id].append(edge_id)

            if target_id not in self._in_edges:
                self._in_edges[target_id] = []
            if edge_id not in self._in_edges[target_id]:
                self._in_edges[target_id].append(edge_id)

            created_edges.append(edge)

        return created_edges

    def get_successors(self, node_id: str) -> List[str]:
        """Get all target node IDs reachable from node_id via active edges."""
        res = []
        seen = set()
        for eid in self._out_edges.get(node_id, []):
            if eid in self.edges:
                edge = self.edges[eid]
                if edge.is_active and edge.target_id not in seen:
                    seen.add(edge.target_id)
                    res.append(edge.target_id)
        return res

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all source node IDs that have active edges pointing to node_id."""
        res = []
        seen = set()
        for eid in self._in_edges.get(node_id, []):
            if eid in self.edges:
                edge = self.edges[eid]
                if edge.is_active and edge.source_id not in seen:
                    seen.add(edge.source_id)
                    res.append(edge.source_id)
        return res

    def get_out_edges_data(self, node_id: str, data: bool = False):
        """Get outgoing edges from node_id, matching NetworkX out_edges format."""
        results = []
        for eid in self._out_edges.get(node_id, []):
            if eid in self.edges:
                edge = self.edges[eid]
                if edge.is_active:
                    if data:
                        results.append((edge.source_id, edge.target_id, {
                            "relation_type": edge.relation_type,
                            "confidence": edge.confidence,
                            "edge_id": eid,
                        }))
                    else:
                        results.append((edge.source_id, edge.target_id))
        return results

    def get_in_edges_data(self, node_id: str, data: bool = False):
        """Get incoming edges to node_id, matching NetworkX in_edges format."""
        results = []
        for eid in self._in_edges.get(node_id, []):
            if eid in self.edges:
                edge = self.edges[eid]
                if edge.is_active:
                    if data:
                        results.append((edge.source_id, edge.target_id, {
                            "relation_type": edge.relation_type,
                            "confidence": edge.confidence,
                            "edge_id": eid,
                        }))
                    else:
                        results.append((edge.source_id, edge.target_id))
        return results

    def has_edge(self, u: str, v: str) -> bool:
        """Check if an active edge exists from u to v."""
        for eid in self._out_edges.get(u, []):
            if eid in self.edges:
                edge = self.edges[eid]
                if edge.target_id == v and edge.is_active:
                    return True
        return False

    def remove_edge_between(self, u: str, v: str):
        """Remove all edges between u and v."""
        eids_to_remove = []
        for eid in self._out_edges.get(u, []):
            if eid in self.edges and self.edges[eid].target_id == v:
                eids_to_remove.append(eid)
        for eid in eids_to_remove:
            if eid in self.edges:
                edge = self.edges.pop(eid)
                if edge.source_id in self._out_edges and eid in self._out_edges[edge.source_id]:
                    self._out_edges[edge.source_id].remove(eid)
                if edge.target_id in self._in_edges and eid in self._in_edges[edge.target_id]:
                    self._in_edges[edge.target_id].remove(eid)


    def traverse(
        self,
        query_vector: np.ndarray,
        relation_type: Optional[str] = None,
        top_k: int = 10,
        confidence_threshold: float = 0.3,
        max_hops: int = 1,
        level: Optional[GraphLevel] = None,
        sector: Optional[str] = None,
    ) -> Subgraph:
        """
        Query the Mayon: find relevant nodes and their connected subgraph.
        """
        if len(self.nodes) == 0:
            return Subgraph(
                nodes=[], edges=[], query_vector=query_vector,
                confidence=0.0, hop_depth=0,
            )

        # Normalize query vector
        query_vector = query_vector.astype(np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Step 1: ANN search for seed nodes
        # Retrieve extra if sector filter is applied
        k = min(top_k * 3 if sector else top_k, max(1, len(self.nodes)))
        scores, indices = self.vector_index.search(query_vector.reshape(1, -1), k)
        scores, indices = scores[0], indices[0]

        # Map vector indices to node IDs
        seed_ids = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            node_id = self._faiss_idx_to_id.get(int(idx))
            if node_id is None or node_id not in self.nodes:
                continue
            node = self.nodes[node_id]

            # Filter by level
            if level is not None and node.level != level:
                continue

            # Filter by sector (if specified and not "all")
            if sector and sector.lower() not in ("all", "any"):
                if node.sector != sector.lower() and node.sector != "general":
                    continue

            node.metadata["vector_similarity"] = float(score)
            seed_ids.append(node_id)
            if len(seed_ids) >= top_k:
                break

        # Step 2: Expand via graph traversal (BFS up to max_hops)
        collected_nodes: Dict[str, ConceptNode] = {}
        collected_edges: List[RelationEdge] = []
        frontier: List[str] = list(dict.fromkeys(seed_ids))
        visited: Set[str] = set()

        for hop in range(max_hops + 1):
            next_frontier: List[str] = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.access_count += 1
                    collected_nodes[node_id] = node

                # Follow outgoing edges
                if node_id in self._out_edges:
                    for edge_id in self._out_edges[node_id]:
                        if edge_id in self.edges:
                            edge = self.edges[edge_id]
                            if relation_type and edge.relation_type != relation_type:
                                continue
                            if edge.confidence < confidence_threshold:
                                continue
                            if not edge.is_active:
                                continue
                            collected_edges.append(edge)
                            neighbor_id = edge.target_id
                            if neighbor_id not in visited and neighbor_id in self.nodes:
                                neighbor_node = self.nodes[neighbor_id]
                                if "vector_similarity" not in neighbor_node.metadata:
                                    neighbor_node.metadata["vector_similarity"] = float(
                                        node.metadata.get("vector_similarity", 0.5) * edge.confidence
                                    )
                                if neighbor_id not in next_frontier:
                                    next_frontier.append(neighbor_id)

                # Also follow incoming edges (bidirectional traversal)
                if node_id in self._in_edges:
                    for edge_id in self._in_edges[node_id]:
                        if edge_id in self.edges:
                            edge = self.edges[edge_id]
                            if relation_type and edge.relation_type != relation_type:
                                continue
                            if edge.confidence < confidence_threshold:
                                continue
                            if not edge.is_active:
                                continue
                            collected_edges.append(edge)
                            neighbor_id = edge.source_id
                            if neighbor_id not in visited and neighbor_id in self.nodes:
                                if neighbor_id not in next_frontier:
                                    next_frontier.append(neighbor_id)

            frontier = [x for x in next_frontier if x not in visited]

        node_list = list(collected_nodes.values())
        if node_list:
            avg_confidence = np.mean([n.confidence for n in node_list])
        else:
            avg_confidence = 0.0

        return Subgraph(
            nodes=node_list,
            edges=collected_edges,
            query_vector=query_vector,
            confidence=float(avg_confidence),
            hop_depth=max_hops,
        )

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        sector: Optional[str] = None,
        level: Optional[GraphLevel] = None,
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Direct semantic vector similarity search returning (ConceptNode, score) pairs.
        """
        if len(self.nodes) == 0:
            return []

        # Normalize query vector
        query_vector = query_vector.astype(np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        k = min(top_k * 3 if sector else top_k, max(1, len(self.nodes)))
        scores, indices = self.vector_index.search(query_vector.reshape(1, -1), k)
        scores, indices = scores[0], indices[0]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            node_id = self._faiss_idx_to_id.get(int(idx))
            if node_id is None or node_id not in self.nodes:
                continue
            node = self.nodes[node_id]

            if level is not None and node.level != level:
                continue
            if sector and sector.lower() not in ("all", "any"):
                if node.sector != sector.lower() and node.sector != "general":
                    continue

            results.append((node, float(score)))
            if len(results) >= top_k:
                break

        return results

    def decay(self, factor: float = 0.99, min_confidence: float = 0.05):
        """
        Apply temporal decay to all edge confidences.
        Edges below min_confidence are pruned (forgotten).
        """
        to_remove = []
        for edge_id, edge in self.edges.items():
            edge.confidence *= factor
            if edge.confidence < min_confidence:
                to_remove.append(edge_id)

        for edge_id in to_remove:
            edge = self.edges.pop(edge_id)
            if edge.source_id in self._out_edges and edge_id in self._out_edges[edge.source_id]:
                self._out_edges[edge.source_id].remove(edge_id)
            if edge.target_id in self._in_edges and edge_id in self._in_edges[edge.target_id]:
                self._in_edges[edge.target_id].remove(edge_id)

        return len(to_remove)

    def _find_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> Optional[RelationEdge]:
        edge_id = f"{source_id}->{target_id}:{relation_type}"
        return self.edges.get(edge_id)

    def _resolve_conflict(
        self,
        existing: RelationEdge,
        new_confidence: float,
        new_provenance: str,
    ) -> RelationEdge:
        total = existing.confidence + new_confidence
        if total > 0:
            existing.confidence = (
                existing.confidence * existing.confidence
                + new_confidence * new_confidence
            ) / total
        existing.provenance = f"{existing.provenance}; {new_provenance}"
        existing.timestamp = time.time()
        return existing

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Retrieve a concept node by its ID."""
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[RelationEdge]:
        """Get all outgoing relation edges from a node."""
        return [self.edges[eid] for eid in self._out_edges.get(node_id, []) if eid in self.edges and self.edges[eid].is_active]

    def get_edges_to(self, node_id: str) -> List[RelationEdge]:
        """Get all incoming relation edges to a node."""
        return [self.edges[eid] for eid in self._in_edges.get(node_id, []) if eid in self.edges and self.edges[eid].is_active]

    def sector_stats(self) -> Dict[str, Dict[str, int]]:
        """Get nodes and edges breakdown per sector."""
        stats = {}
        for node in self.nodes.values():
            sec = node.sector
            if sec not in stats:
                stats[sec] = {"nodes": 0, "edges": 0, "images": 0}
            stats[sec]["nodes"] += 1
            if node.node_type == "image":
                stats[sec]["images"] += 1

        for edge in self.edges.values():
            sec = edge.sector
            if sec not in stats:
                stats[sec] = {"nodes": 0, "edges": 0, "images": 0}
            stats[sec]["edges"] += 1

        return stats

    def stats(self) -> Dict:
        """Get full Mayon statistics."""
        level_counts = {
            level.name: len(ids) for level, ids in self.level_indices.items()
        }
        relation_counts: Dict[str, int] = {}
        for edge in self.edges.values():
            relation_counts[edge.relation_type] = (
                relation_counts.get(edge.relation_type, 0) + 1
            )
        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "vector_backend": self.vector_index.backend_name,
            "nodes_by_level": level_counts,
            "sector_stats": self.sector_stats(),
            "edges_by_relation": relation_counts,
        }

    def save(self, save_dir: str):
        """Save the entire Mayon graph (nodes, edges, vector index) to disk."""
        import json
        import pickle
        from pathlib import Path

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "concept_dim": self.concept_dim,
            "ann_index_type": self.ann_index_type,
            "vector_backend": self.vector_index.backend_name,
            "nodes": {
                nid: {
                    "id": n.id,
                    "level": int(n.level),
                    "source_text": n.source_text,
                    "sector": n.sector,
                    "node_type": n.node_type,
                    "timestamp": n.timestamp,
                    "provenance": n.provenance,
                    "confidence": n.confidence,
                    "access_count": n.access_count,
                    "metadata": n.metadata,
                }
                for nid, n in self.nodes.items()
            },
            "edges": {
                eid: {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation_type": e.relation_type,
                    "confidence": e.confidence,
                    "sector": e.sector,
                    "provenance": e.provenance,
                    "timestamp": e.timestamp,
                    "valid_from": e.valid_from,
                    "valid_to": e.valid_to,
                    "superseded_by": e.superseded_by,
                    "metadata": e.metadata,
                }
                for eid, e in self.edges.items()
            },
            "_id_to_faiss_idx": self._id_to_faiss_idx,
            "_faiss_idx_to_id": {str(k): v for k, v in self._faiss_idx_to_id.items()},
            "_next_faiss_idx": self._next_faiss_idx,
        }

        with open(path / "mayon_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        node_vectors = {nid: n.vector for nid, n in self.nodes.items()}
        with open(path / "vectors.pkl", "wb") as f:
            pickle.dump(node_vectors, f)

        # Save vector index
        try:
            if isinstance(self.vector_index, USearchVectorIndex):
                self.vector_index.save(str(path / "vector_index.usearch"))
            elif isinstance(self.vector_index, FaissVectorIndex):
                self.vector_index.save(str(path / "faiss.index"))
            else:
                self.vector_index.save(str(path / "vector_index.npz"))
        except Exception:
            pass

    @classmethod
    def load(cls, save_dir: str) -> "MayonGraph":
        """Load a Mayon graph from disk."""
        import json
        import pickle
        from pathlib import Path

        path = Path(save_dir)
        mayon_file = path / "mayon_data.json"
        if not mayon_file.exists() or os.path.getsize(mayon_file) < 10:
            raise ValueError(f"Mayon data file {mayon_file} is empty or missing")

        with open(mayon_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "concept_dim" not in data or "nodes" not in data:
            raise ValueError("Invalid or empty mayon_data.json")

        vec_file = path / "vectors.pkl"
        if not vec_file.exists():
            raise ValueError(f"Vectors file {vec_file} missing")

        with open(vec_file, "rb") as f:
            vectors = pickle.load(f)

        graph = cls(
            concept_dim=data["concept_dim"],
            ann_index_type=data.get("ann_index_type", "auto"),
        )

        for nid, ndata in data["nodes"].items():
            node = ConceptNode(
                id=ndata["id"],
                vector=vectors[nid],
                level=GraphLevel(ndata["level"]),
                source_text=ndata["source_text"],
                sector=ndata.get("sector", "general"),
                node_type=ndata.get("node_type", "text"),
                timestamp=ndata["timestamp"],
                provenance=ndata["provenance"],
                confidence=ndata["confidence"],
                access_count=ndata.get("access_count", 0),
                metadata=ndata.get("metadata", {}),
            )
            graph.nodes[nid] = node
            if nid not in graph._out_edges:
                graph._out_edges[nid] = []
            if nid not in graph._in_edges:
                graph._in_edges[nid] = []
            graph.level_indices[node.level].append(nid)

            sec = node.sector.lower()
            if sec not in graph.sector_indices:
                graph.sector_indices[sec] = []
            graph.sector_indices[sec].append(nid)

        for eid, edata in data["edges"].items():
            edge = RelationEdge(
                source_id=edata["source_id"],
                target_id=edata["target_id"],
                relation_type=edata["relation_type"],
                confidence=edata["confidence"],
                sector=edata.get("sector", "general"),
                provenance=edata["provenance"],
                timestamp=edata["timestamp"],
                valid_from=edata.get("valid_from"),
                valid_to=edata.get("valid_to"),
                superseded_by=edata.get("superseded_by"),
                metadata=edata.get("metadata", {}),
            )
            graph.edges[eid] = edge
            if edge.source_id not in graph._out_edges:
                graph._out_edges[edge.source_id] = []
            if eid not in graph._out_edges[edge.source_id]:
                graph._out_edges[edge.source_id].append(eid)
            if edge.target_id not in graph._in_edges:
                graph._in_edges[edge.target_id] = []
            if eid not in graph._in_edges[edge.target_id]:
                graph._in_edges[edge.target_id].append(eid)

        graph._id_to_faiss_idx = data.get("_id_to_faiss_idx", {})
        graph._faiss_idx_to_id = {
            int(k): v for k, v in data.get("_faiss_idx_to_id", {}).items()
        }
        graph._next_faiss_idx = data.get("_next_faiss_idx", 0)

        # Restore vector index
        usearch_path = path / "vector_index.usearch"
        faiss_path = path / "faiss.index"
        npz_path = path / "vector_index.npz"

        loaded = False
        if usearch_path.exists() and isinstance(graph.vector_index, USearchVectorIndex):
            try:
                graph.vector_index.load(str(usearch_path))
                loaded = True
            except Exception:
                loaded = False

        if not loaded and faiss_path.exists() and isinstance(graph.vector_index, FaissVectorIndex):
            try:
                graph.vector_index.load(str(faiss_path))
                loaded = True
            except Exception:
                loaded = False

        if not loaded and npz_path.exists() and isinstance(graph.vector_index, NumpyVectorIndex):
            try:
                graph.vector_index.load(str(npz_path))
                loaded = True
            except Exception:
                loaded = False

        # Fallback: Populate vector index directly from node vectors if index file wasn't loaded
        if not loaded and len(graph.nodes) > 0:
            ordered_nodes = list(graph.nodes.values())
            vec_matrix = np.stack([n.vector for n in ordered_nodes])
            keys = [graph._id_to_faiss_idx.get(n.id, i) for i, n in enumerate(ordered_nodes)]
            graph.vector_index.add_batch(vec_matrix, keys=keys)

        return graph

    def __repr__(self) -> str:
        return (
            f"MayonGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"dim={self.concept_dim}, backend={self.vector_index.backend_name}, "
            f"sectors={list(self.sector_indices.keys())})"
        )

