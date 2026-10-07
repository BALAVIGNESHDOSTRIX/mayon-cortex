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
ImagineEngine (ImagineNet) — Cognitive Simulation & Synthesis Engine
====================================================================
Pillar 7 of Cortex-Graph: Brain-Like Generative Imagination.

Unlike traditional LLMs that rely on memorizing billions of parameters,
human cognition uses generative mental simulation:
1. MentalSandbox: Ephemeral "what-if" counterfactual simulation sandboxes.
2. ConceptBlender: Cross-domain conceptual blending (Bisociation) to invent new ideas.
3. DreamConsolidator: Offline sleep/REM memory replay, mutation, synaptic pruning, and abstraction.
4. CuriosityEngine: Epistemic surprise detection & knowledge-void exploration.
"""

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from mayon_cortex.core.graph import ConceptNode, GraphLevel, MayonGraph, RelationEdge

from mayon_cortex.dynamics.activation_wave import ActivationWave
from mayon_cortex.reasoning.analogy import AnalogyEngine
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.core.ndu import NodeDecisionUnit
from mayon_cortex.reasoning.path_ranker import PathRanker, ScoredPath


@dataclass
class SimulationResult:
    """The outcome of a counterfactual 'What-If' simulation."""
    scenario_name: str
    hypothetical_facts: List[str]
    target_query: str
    activated_paths: List[ScoredPath] = field(default_factory=list)
    inferred_consequences: List[str] = field(default_factory=list)
    confidence: float = 0.0
    emergent_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""


@dataclass
class EmergentConcept:
    """A novel concept synthesized by blending two distinct domains."""
    concept_id: str
    name: str
    source_concepts: Tuple[str, str]
    vector: np.ndarray
    blended_properties: List[str]
    synthesized_relations: List[Dict[str, Any]]
    explanation: str
    confidence: float


@dataclass
class DreamStats:
    """Telemetry from an offline dream / sleep consolidation cycle."""
    replays_conducted: int = 0
    hypotheses_generated: int = 0
    edges_pruned: int = 0
    abstractions_formed: int = 0
    synthetic_links_crystallized: int = 0


@dataclass
class CuriosityGap:
    """An identified gap or structural hole in the knowledge graph."""
    node_id: str
    concept_text: str
    sector: str
    degree: int
    isolation_score: float
    suggested_query: str


class MentalSandbox:
    """
    Isolated scratchpad branched from the main knowledge graph.
    Allows testing counterfactuals, disabling nodes, or injecting hypotheses
    without corrupting the main brain.
    """

    def __init__(self, base_graph: MayonGraph, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.base_graph = base_graph
        self.sandbox_graph = MayonGraph(concept_dim=base_graph.concept_dim)
        self._clone_subgraph()

    def _clone_subgraph(self, max_nodes: int = 1000):
        """Clone nodes and active edges into the isolated sandbox."""
        nodes_to_add = list(self.base_graph.nodes.values())[:max_nodes]
        if nodes_to_add:
            vectors = np.stack([n.vector for n in nodes_to_add])
            texts = [n.source_text for n in nodes_to_add]
            levels = [n.level for n in nodes_to_add]
            sectors = [n.sector for n in nodes_to_add]
            types = [n.node_type for n in nodes_to_add]
            provs = [f"sandbox_clone:{n.provenance}" for n in nodes_to_add]
            confs = [n.confidence for n in nodes_to_add]
            metas = [copy.deepcopy(n.metadata) for n in nodes_to_add]
            ids = [n.id for n in nodes_to_add]

            self.sandbox_graph.add_nodes_bulk(
                vectors=vectors,
                source_texts=texts,
                levels=levels,
                sectors=sectors,
                node_types=types,
                provenances=provs,
                confidences=confs,
                metadata_list=metas,
                node_ids=ids,
            )

        for eid, edge in self.base_graph.edges.items():
            if edge.source_id in self.sandbox_graph.nodes and edge.target_id in self.sandbox_graph.nodes:
                self.sandbox_graph.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation_type=edge.relation_type,
                    confidence=edge.confidence,
                    sector=edge.sector,
                    provenance=f"sandbox_clone:{edge.provenance}",
                    metadata=copy.deepcopy(edge.metadata),
                )

    def inject_fact(
        self,
        source_text: str,
        vector: np.ndarray,
        sector: str = "general",
        relation_to: Optional[Tuple[str, str]] = None, # (target_node_id, relation_type)
        confidence: float = 0.9,
    ) -> str:
        """Inject a hypothetical fact into the sandbox."""
        node = self.sandbox_graph.add_node(
            vector=vector,
            source_text=source_text,
            level=GraphLevel.FACT,
            sector=sector,
            confidence=confidence,
            provenance="mental_simulation",
        )
        node_id = node.id

        if relation_to:
            target_id, rel_type = relation_to
            if target_id in self.sandbox_graph.nodes:
                self.sandbox_graph.add_edge(
                    source_id=node_id,
                    target_id=target_id,
                    relation_type=rel_type,
                    confidence=confidence,
                    sector=sector,
                    provenance="mental_simulation",
                )
        return node_id

    def suppress_node(self, node_id: str):
        """Suppress/disable a node to test counterfactual ablation."""
        if node_id in self.sandbox_graph.nodes:
            self.sandbox_graph.nodes[node_id].confidence = 0.0
            # Remove incident edges
            out_eids = list(self.sandbox_graph._out_edges.get(node_id, []))
            for eid in out_eids:
                if eid in self.sandbox_graph.edges:
                    self.sandbox_graph.edges[eid].confidence = 0.0

    def simulate_wave(
        self,
        query_vec: np.ndarray,
        ndu: NodeDecisionUnit,
        wave: ActivationWave,
        ranker: PathRanker,
    ) -> List[ScoredPath]:
        """Run spreading activation inside the isolated sandbox."""
        activation = wave.activate(query_vec, self.sandbox_graph, ndu)
        scored = ranker.rank_paths(activation.paths, query_vec, self.sandbox_graph)
        return scored

    def commit_to_graph(
        self,
        target_graph: MayonGraph,
        node_ids: Optional[List[str]] = None,
    ) -> int:
        """Crystallize verified simulation nodes & edges into the permanent graph."""
        count = 0
        target_nids = set(node_ids) if node_ids else set(self.sandbox_graph.nodes.keys())
        for nid in target_nids:
            if nid in self.sandbox_graph.nodes and nid not in target_graph.nodes:
                n = self.sandbox_graph.nodes[nid]
                target_graph.add_nodes_bulk(
                    vectors=n.vector.reshape(1, -1),
                    source_texts=[n.source_text],
                    levels=[n.level],
                    sectors=[n.sector],
                    node_types=[n.node_type],
                    provenances=[n.provenance],
                    confidences=[n.confidence],
                    metadata_list=[copy.deepcopy(n.metadata)],
                    node_ids=[n.id],
                )
                count += 1

        for eid, edge in self.sandbox_graph.edges.items():
            if edge.source_id in target_graph.nodes and edge.target_id in target_graph.nodes:
                if not target_graph.has_edge(edge.source_id, edge.target_id):
                    target_graph.add_edge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        relation_type=edge.relation_type,
                        confidence=edge.confidence,
                        sector=edge.sector,
                        provenance=edge.provenance,
                        metadata=copy.deepcopy(edge.metadata),
                    )
        return count


class ConceptBlender:
    """
    Synthesizes emergent concepts from distant domains via structural isomorphism
    and vector-space interpolation (Bisociation).
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.analogy_engine = AnalogyEngine(self.config)

    def blend(
        self,
        concept_a: str,
        concept_b: str,
        graph: MayonGraph,
        embedder: Any,
        alpha: float = 0.5,
    ) -> Optional[EmergentConcept]:
        """Blend concept A and concept B into a synthesized emergent concept."""
        if not graph.nodes:
            return None

        # Resolve nodes
        node_a_id = self._find_node(concept_a, graph)
        node_b_id = self._find_node(concept_b, graph)

        if not node_a_id or not node_b_id:
            # Generate vectors on the fly if not explicit
            vec_a = embedder.encode_single(concept_a)
            vec_b = embedder.encode_single(concept_b)
            source_text_a = concept_a
            source_text_b = concept_b
        else:
            node_a = graph.nodes[node_a_id]
            node_b = graph.nodes[node_b_id]
            vec_a = node_a.vector
            vec_b = node_b.vector
            source_text_a = node_a.source_text
            source_text_b = node_b.source_text

        # Interpolate concept vector in ℝ³⁸⁴
        blended_vec = alpha * vec_a + (1.0 - alpha) * vec_b
        norm = np.linalg.norm(blended_vec)
        if norm > 1e-8:
            blended_vec = blended_vec / norm

        concept_name = f"Emergent({source_text_a} ⊗ {source_text_b})"
        concept_id = f"blend_{int(time.time()*1000)}"

        # Synthesize blended relations
        synthesized_rels = []
        blended_props = []

        # Inherit high-confidence relations from both parents
        if node_a_id and node_a_id in graph._out_edges:
            for eid in graph._out_edges[node_a_id][:3]:
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    target = graph.nodes.get(edge.target_id)
                    target_name = target.source_text if target else edge.target_id
                    synthesized_rels.append({
                        "relation": edge.relation_type,
                        "target": target_name,
                        "inherited_from": source_text_a,
                        "confidence": edge.confidence * 0.85,
                    })
                    blended_props.append(f"{edge.relation_type} {target_name}")

        if node_b_id and node_b_id in graph._out_edges:
            for eid in graph._out_edges[node_b_id][:3]:
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    target = graph.nodes.get(edge.target_id)
                    target_name = target.source_text if target else edge.target_id
                    synthesized_rels.append({
                        "relation": edge.relation_type,
                        "target": target_name,
                        "inherited_from": source_text_b,
                        "confidence": edge.confidence * 0.85,
                    })
                    blended_props.append(f"{edge.relation_type} {target_name}")

        explanation = (
            f"Synthesized concept '{concept_name}' by cross-pollinating '{source_text_a}' "
            f"with '{source_text_b}', inheriting {len(blended_props)} relational properties."
        )

        return EmergentConcept(
            concept_id=concept_id,
            name=concept_name,
            source_concepts=(source_text_a, source_text_b),
            vector=blended_vec.astype(np.float32),
            blended_properties=blended_props,
            synthesized_relations=synthesized_rels,
            explanation=explanation,
            confidence=0.80,
        )

    def _find_node(self, term: str, graph: MayonGraph) -> Optional[str]:
        term_clean = term.lower().strip()
        for nid, node in graph.nodes.items():
            if term_clean in node.source_text.lower():
                return nid
        return None


class DreamConsolidator:
    """
    Sleep / Dream State Engine.
    Performs offline replay of waking memory traces, generates counterfactual
    hypothesis links, and prunes noisy/stale synaptic connections.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def dream(
        self,
        graph: MayonGraph,
        working_memory: Any,
        abstraction_engine: Any,
        cycles: int = 10,
    ) -> DreamStats:
        """Run an offline dream consolidation cycle."""
        stats = DreamStats()
        if not graph.nodes:
            return stats

        all_nids = list(graph.nodes.keys())

        # 1. Episodic Replay from Working Memory
        recent_items = working_memory.buffer if hasattr(working_memory, "buffer") else (
            working_memory.history if hasattr(working_memory, "history") else []
        )
        for item in recent_items[-cycles:]:
            stats.replays_conducted += 1
            # Reinforce nodes used in recent successful memories
            for nid in getattr(item, "retrieved_node_ids", []):
                if nid in graph.nodes:
                    graph.nodes[nid].confidence = min(1.0, graph.nodes[nid].confidence + 0.02)
                    graph.nodes[nid].access_count += 1

        # 2. Dream State Hypotheses Generation (Creative Cross-Linking)
        if len(all_nids) >= 2:
            num_attempts = min(cycles * 2, 50)
            for _ in range(num_attempts):
                u, v = random.sample(all_nids, 2)
                if u == v or graph.has_edge(u, v):
                    continue

                node_u = graph.nodes[u]
                node_v = graph.nodes[v]

                # Measure semantic affinity
                sim = float(np.dot(node_u.vector, node_v.vector) / (
                    (np.linalg.norm(node_u.vector) * np.linalg.norm(node_v.vector)) + 1e-8
                ))

                # If high affinity but in different sectors, dream a connection
                if sim > 0.65 and node_u.sector != node_v.sector:
                    graph.add_edge(
                        source_id=u,
                        target_id=v,
                        relation_type="analogous-to",
                        confidence=sim * 0.75,
                        provenance="dream_consolidation",
                        sector="general",
                    )
                    stats.hypotheses_generated += 1
                    stats.synthetic_links_crystallized += 1

        # 3. Synaptic Pruning (Decay & Remove weak edges)
        prune_thresh = self.config.dream_prune_threshold
        edges_to_prune = []
        for eid, edge in list(graph.edges.items()):
            # Decay edge
            edge.confidence *= self.config.decay_factor
            if edge.confidence < prune_thresh and edge.provenance.startswith("dream"):
                edges_to_prune.append((edge.source_id, edge.target_id, eid))

        for u, v, eid in edges_to_prune:
            graph.remove_edge_between(u, v)
            stats.edges_pruned += 1

        # 4. Trigger Fractal Concept Abstraction
        if abstraction_engine:
            abstractions = abstraction_engine.run_abstraction_cycle(graph)
            stats.abstractions_formed = len(getattr(abstractions, "new_abstract_concepts", []))

        return stats


class CuriosityEngine:
    """
    Epistemic Surprise & Curiosity Engine.
    Detects knowledge gaps, isolated concept islands, and generates probes
    for active learning.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def identify_gaps(self, graph: MayonGraph, top_k: int = 5) -> List[CuriosityGap]:
        """Find concepts with high isolation or missing relational edges."""
        if not graph.nodes:
            return []

        gaps = []
        for nid, node in graph.nodes.items():
            if node.level == GraphLevel.PHRASE:
                continue

            out_degree = len(graph._out_edges.get(nid, []))
            in_degree = len(graph._in_edges.get(nid, []))
            total_degree = out_degree + in_degree

            # Isolation penalty
            isolation = 1.0 / (total_degree + 1.0)

            # High confidence node with zero or few connections = curiosity gap
            if total_degree <= 2 and node.confidence > 0.5:
                suggested_q = f"How does '{node.source_text}' relate to other concepts in {node.sector}?"
                gaps.append(CuriosityGap(
                    node_id=nid,
                    concept_text=node.source_text,
                    sector=node.sector,
                    degree=total_degree,
                    isolation_score=isolation,
                    suggested_query=suggested_q,
                ))

        gaps.sort(key=lambda g: g.isolation_score, reverse=True)
        return gaps[:top_k]


class ImagineEngine:
    """
    Unified Orchestrator for Pillar 7: Generative Imagination.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.blender = ConceptBlender(self.config)
        self.consolidator = DreamConsolidator(self.config)
        self.curiosity = CuriosityEngine(self.config)

    def what_if(
        self,
        base_graph: MayonGraph,
        hypothetical_fact: str,
        query: str,
        embedder: Any,
        ndu: NodeDecisionUnit,
        wave: ActivationWave,
        ranker: PathRanker,
        sector: str = "general",
    ) -> SimulationResult:
        """Run a counterfactual simulation in an isolated mental sandbox."""
        sandbox = MentalSandbox(base_graph, self.config)

        # Encode hypothetical fact and query
        hypo_vec = embedder.encode_single(hypothetical_fact)
        query_vec = embedder.encode_single(query)

        # Inject hypothesis into sandbox
        hypo_nid = sandbox.inject_fact(
            source_text=hypothetical_fact,
            vector=hypo_vec,
            sector=sector,
            confidence=0.95,
        )

        # Connect hypothesis to nearest relevant seed nodes in sandbox
        hypo_norm = hypo_vec.astype(np.float32)
        norm_val = np.linalg.norm(hypo_norm)
        if norm_val > 0:
            hypo_norm = hypo_norm / norm_val

        k = min(4, len(sandbox.sandbox_graph.nodes))
        if k > 1:
            scores, indices = sandbox.sandbox_graph.faiss_index.search(hypo_norm.reshape(1, -1), k)
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                node_id = sandbox.sandbox_graph._faiss_idx_to_id.get(int(idx))
                if node_id and node_id != hypo_nid and node_id in sandbox.sandbox_graph.nodes and score > 0.3:
                    sandbox.sandbox_graph.add_edge(
                        source_id=hypo_nid,
                        target_id=node_id,
                        relation_type="hypothetically-implies",
                        confidence=float(score),
                        sector=sector,
                        provenance="counterfactual_simulation",
                    )

        # Run spreading activation wave in sandbox
        scored_paths = sandbox.simulate_wave(query_vec, ndu, wave, ranker)

        consequences = []
        emergent_edges = []
        for p in scored_paths:
            for src, rel, tgt in p.path.edges:
                consequences.append(f"{src} -[{rel}]-> {tgt}")
                emergent_edges.append((src, rel, tgt))

        conf = scored_paths[0].score if scored_paths else 0.0
        summary = (
            f"Simulated scenario: '{hypothetical_fact}'. Under this condition, "
            f"found {len(scored_paths)} causal reasoning pathways for '{query}' with confidence {conf:.2f}."
        )

        return SimulationResult(
            scenario_name=f"WhatIf({hypothetical_fact})",
            hypothetical_facts=[hypothetical_fact],
            target_query=query,
            activated_paths=scored_paths,
            inferred_consequences=consequences[:10],
            confidence=conf,
            emergent_edges=emergent_edges[:10],
            summary=summary,
        )

    def blend(
        self,
        concept_a: str,
        concept_b: str,
        graph: MayonGraph,
        embedder: Any,
        alpha: float = 0.5,
    ) -> Optional[EmergentConcept]:
        """Synthesize emergent concept from two ideas."""
        return self.blender.blend(concept_a, concept_b, graph, embedder, alpha=alpha)

    def dream(
        self,
        graph: MayonGraph,
        working_memory: Any,
        abstraction_engine: Any,
        cycles: int = 10,
    ) -> DreamStats:
        """Run offline dream consolidation & synaptic pruning."""
        return self.consolidator.dream(graph, working_memory, abstraction_engine, cycles=cycles)

    def identify_gaps(self, graph: MayonGraph, top_k: int = 5) -> List[CuriosityGap]:
        """Find curiosity exploration targets."""
        return self.curiosity.identify_gaps(graph, top_k=top_k)
