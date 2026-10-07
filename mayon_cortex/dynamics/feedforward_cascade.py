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
Feedforward Cascade — Biologically-Accurate Forward-Only Neural Firing
========================================================================
Phase 1.5 of the Brain-Like Intelligence upgrade.

Like biological neurons: NO BACKPROPAGATION. Each node independently
decides to fire or not based on LOCAL inputs + mode context.
Creativity emerges from cascading forward decisions, not trained weights.

How it works:
  1. Stimulus arrives (query vector)
  2. Seed nodes activate (FAISS nearest)
  3. Each active node computes LOCAL energy from:
     - Incoming signal strength (cosine similarity)
     - Edge weight (synapse strength)
     - Mode context (factual=strict, creative=relaxed)
     - Noise injection (for creativity — like neural noise)
  4. If energy > threshold → FIRE → propagate to neighbors
  5. Fired nodes can trigger LATERAL INHIBITION (suppress similar neighbors)
  6. Cascade continues until energy dissipates
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.executive.cognitive_modes import ModeConfig


@dataclass
class NeuronState:
    """State of a single graph neuron during cascade."""
    node_id: str
    activation_energy: float
    fired: bool = False
    fire_time: int = -1
    inhibited_by: Optional[str] = None
    incoming_signal: Optional[np.ndarray] = None


@dataclass
class CascadePath:
    """A path traced through the cascade."""
    node_ids: List[str]
    edges: List[Tuple[str, str, str]]  # (source, relation, target)
    total_energy: float
    confidence: float
    fire_times: List[int]


@dataclass
class CascadeResult:
    """Result of a feedforward cascade."""
    paths: List[CascadePath]
    fired_nodes: List[str]
    total_nodes_evaluated: int
    cascade_steps: int
    active_node_ids: Set[str] = field(default_factory=set)
    blended_concepts: List[str] = field(default_factory=list)


class FeedforwardCascade:
    """
    Forward-only neural cascade through the knowledge graph.
    No backpropagation. No gradient descent. Pure local decisions.
    
    Each neuron independently computes:
      energy = cosine(node, signal) × edge_weight × mode_factor + noise
    
    If energy > threshold → FIRE → propagate to neighbors
    
    In FACTUAL mode: strict threshold, no noise → only strong paths
    In CREATIVE mode: low threshold, high noise → wide activation, novel combos
    """

    def __init__(self, concept_dim: int = 384):
        self.concept_dim = concept_dim

    def cascade(
        self,
        stimulus_vec: np.ndarray,
        graph: Any,
        mode_config: ModeConfig,
        max_steps: int = 15,
        seed_top_k: int = 10,
    ) -> CascadeResult:
        """
        Run a feedforward cascade through the graph.
        
        Args:
            stimulus_vec: Query vector (ℝ³⁸⁴)
            graph: MayonGraph instance
            mode_config: Cognitive mode parameters
            max_steps: Maximum cascade steps
            seed_top_k: Number of seed nodes from FAISS
            
        Returns:
            CascadeResult with all activated paths
        """
        if graph.num_nodes == 0:
            return CascadeResult(
                paths=[], fired_nodes=[], total_nodes_evaluated=0,
                cascade_steps=0,
            )

        # Step 1: Seed nodes via FAISS
        seed_ids = self._get_seeds(stimulus_vec, graph, seed_top_k)
        if not seed_ids:
            return CascadeResult(
                paths=[], fired_nodes=[], total_nodes_evaluated=0,
                cascade_steps=0,
            )

        # Initialize neuron states
        states: Dict[str, NeuronState] = {}
        for sid in seed_ids:
            energy = self._compute_firing_energy(
                graph.nodes[sid].vector, stimulus_vec,
                1.0,  # seed edge weight = 1.0
                mode_config,
            )
            states[sid] = NeuronState(
                node_id=sid,
                activation_energy=energy,
                fired=True,
                fire_time=0,
                incoming_signal=stimulus_vec,
            )

        fired_order: List[str] = list(seed_ids)
        all_paths: List[CascadePath] = []
        total_evaluated = len(seed_ids)
        blended: List[str] = []

        # Step 2: Cascade
        for step in range(1, max_steps + 1):
            # Find all nodes that fired in the previous step
            prev_fired = [
                nid for nid, s in states.items()
                if s.fired and s.fire_time == step - 1
            ]

            if not prev_fired:
                break  # Cascade has died out

            new_fires: List[str] = []

            for source_id in prev_fired:
                source_state = states[source_id]
                source_node = graph.nodes.get(source_id)
                if not source_node:
                    continue

                # Get outgoing edges
                out_edges = graph._out_edges.get(source_id, [])
                for eid in out_edges:
                    if eid not in graph.edges:
                        continue
                    edge = graph.edges[eid]
                    if not edge.is_active:
                        continue

                    target_id = edge.target_id
                    target_node = graph.nodes.get(target_id)
                    if not target_node:
                        continue

                    # Cross-sector check
                    if not mode_config.cross_sector_allowed:
                        if (source_node.sector != "general" and
                            target_node.sector != "general" and
                            source_node.sector != target_node.sector):
                            continue

                    # Already fired? Skip (avoid loops)
                    if target_id in states and states[target_id].fired:
                        continue

                    total_evaluated += 1

                    # Compute firing energy
                    signal = source_state.incoming_signal
                    if signal is None:
                        signal = stimulus_vec

                    energy = self._compute_firing_energy(
                        target_node.vector, signal, edge.confidence, mode_config
                    )

                    # Confidence floor check
                    if edge.confidence < mode_config.confidence_floor:
                        continue

                    # Fire decision
                    if energy > mode_config.ndu_follow_threshold:
                        # FIRE!
                        states[target_id] = NeuronState(
                            node_id=target_id,
                            activation_energy=energy,
                            fired=True,
                            fire_time=step,
                            incoming_signal=target_node.vector * energy,
                        )
                        new_fires.append(target_id)
                        fired_order.append(target_id)

                        # Record path segment
                        all_paths.append(CascadePath(
                            node_ids=[source_id, target_id],
                            edges=[(source_id, edge.relation_type, target_id)],
                            total_energy=source_state.activation_energy + energy,
                            confidence=edge.confidence * energy,
                            fire_times=[step - 1, step],
                        ))

            # Lateral inhibition
            if new_fires:
                self._lateral_inhibition(new_fires, states, graph)

            # Random spontaneous jump (creative mode)
            if mode_config.random_jump_probability > 0:
                if random.random() < mode_config.random_jump_probability:
                    all_nids = list(graph.nodes.keys())
                    rnd_id = random.choice(all_nids)
                    if rnd_id not in states or not states[rnd_id].fired:
                        rnd_energy = random.uniform(0.3, 0.7)
                        states[rnd_id] = NeuronState(
                            node_id=rnd_id,
                            activation_energy=rnd_energy,
                            fired=True,
                            fire_time=step,
                            incoming_signal=graph.nodes[rnd_id].vector * rnd_energy,
                        )
                        fired_order.append(rnd_id)

            # Concept blending (creative mode)
            if mode_config.blend_probability > 0 and len(new_fires) >= 2:
                if random.random() < mode_config.blend_probability:
                    a_id, b_id = random.sample(new_fires, 2)
                    a_node = graph.nodes.get(a_id)
                    b_node = graph.nodes.get(b_id)
                    if a_node and b_node:
                        blend_name = f"Blend({a_node.source_text[:20]} ⊗ {b_node.source_text[:20]})"
                        blended.append(blend_name)

        # Build complete paths from fired nodes
        complete_paths = self._build_complete_paths(fired_order, states, graph)
        all_paths.extend(complete_paths)

        # Sort paths by energy
        all_paths.sort(key=lambda p: p.total_energy, reverse=True)

        return CascadeResult(
            paths=all_paths[:20],  # Top 20 paths
            fired_nodes=fired_order,
            total_nodes_evaluated=total_evaluated,
            cascade_steps=max(
                (s.fire_time for s in states.values() if s.fired),
                default=0,
            ),
            active_node_ids=set(fired_order),
            blended_concepts=blended,
        )

    def _compute_firing_energy(
        self,
        node_vec: np.ndarray,
        incoming_signal: np.ndarray,
        edge_weight: float,
        mode_config: ModeConfig,
    ) -> float:
        """
        Local energy computation at each neuron.
        
        E = cosine(node, signal) × edge_weight + noise
        
        In creative mode: noise is high → weak edges can still cause firing
        In factual mode: noise is zero → only strong evidence activates
        """
        n_norm = np.linalg.norm(node_vec)
        s_norm = np.linalg.norm(incoming_signal)

        if n_norm < 1e-8 or s_norm < 1e-8:
            return 0.0

        cosine = float(np.dot(node_vec, incoming_signal) / (n_norm * s_norm))
        base_energy = max(0.0, cosine) * edge_weight

        # Mode-dependent noise injection (biological neural noise)
        noise = 0.0
        if mode_config.noise_level > 0:
            noise = np.random.normal(0, mode_config.noise_level * 0.3)

        return max(0.0, base_energy + noise)

    def _lateral_inhibition(
        self,
        newly_fired: List[str],
        states: Dict[str, NeuronState],
        graph: Any,
    ) -> None:
        """
        Biological lateral inhibition: when a neuron fires strongly,
        it SUPPRESSES similar nearby neurons.
        
        This prevents repetitive activation and produces DIVERSE patterns,
        leading to more creative and varied outputs.
        """
        if len(newly_fired) < 2:
            return

        # Find the strongest fire
        strongest_id = max(
            newly_fired,
            key=lambda nid: states[nid].activation_energy,
        )
        strongest_node = graph.nodes.get(strongest_id)
        if not strongest_node:
            return

        # Suppress very similar neighbors
        for nid in newly_fired:
            if nid == strongest_id:
                continue
            other_node = graph.nodes.get(nid)
            if not other_node:
                continue

            # Compute similarity
            sim = float(np.dot(strongest_node.vector, other_node.vector) / (
                np.linalg.norm(strongest_node.vector) * np.linalg.norm(other_node.vector) + 1e-8
            ))

            # If very similar (> 0.9), inhibit the weaker one
            if sim > 0.9 and states[nid].activation_energy < states[strongest_id].activation_energy:
                states[nid].inhibited_by = strongest_id
                states[nid].activation_energy *= 0.3  # Reduce but don't zero out

    def _get_seeds(
        self,
        query_vec: np.ndarray,
        graph: Any,
        top_k: int,
    ) -> List[str]:
        """Get seed nodes via FAISS nearest neighbor search."""
        if graph.num_nodes == 0:
            return []

        norm = np.linalg.norm(query_vec)
        if norm < 1e-8:
            return []

        q_normed = (query_vec / norm).reshape(1, -1).astype(np.float32)
        k = min(top_k, graph.num_nodes)
        scores, indices = graph.faiss_index.search(q_normed, k)

        id_list = list(graph.nodes.keys())
        seeds = []
        for idx in indices[0]:
            if 0 <= idx < len(id_list):
                seeds.append(id_list[idx])

        return seeds

    def _build_complete_paths(
        self,
        fired_order: List[str],
        states: Dict[str, NeuronState],
        graph: Any,
    ) -> List[CascadePath]:
        """Build longer paths by connecting consecutively fired nodes."""
        if len(fired_order) < 3:
            return []

        paths: List[CascadePath] = []

        # Group by fire_time
        time_groups: Dict[int, List[str]] = {}
        for nid in fired_order:
            if nid in states:
                t = states[nid].fire_time
                time_groups.setdefault(t, []).append(nid)

        # Build paths through consecutive time steps
        sorted_times = sorted(time_groups.keys())
        for i in range(len(sorted_times) - 1):
            t1 = sorted_times[i]
            t2 = sorted_times[i + 1]
            for n1 in time_groups[t1][:3]:
                for n2 in time_groups[t2][:3]:
                    # Check if there's an edge
                    for eid in graph._out_edges.get(n1, []):
                        if eid in graph.edges and graph.edges[eid].target_id == n2:
                            edge = graph.edges[eid]
                            e1 = states[n1].activation_energy
                            e2 = states[n2].activation_energy
                            paths.append(CascadePath(
                                node_ids=[n1, n2],
                                edges=[(n1, edge.relation_type, n2)],
                                total_energy=e1 + e2,
                                confidence=edge.confidence,
                                fire_times=[t1, t2],
                            ))

        return paths


# Aliases
CascadeActivation = CascadeResult
CascadeLayer = CascadePath

