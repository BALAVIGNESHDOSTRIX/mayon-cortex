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
Graph Reasoning Environment & Symbolic Verifier
===============================================
A Markov Decision Process (MDP) environment for training neural policy networks
(like Mayon-Net's Nucleus) to traverse knowledge graphs with verifiable deductive logic.

Core components:
- GraphAction: Discrete transition from current node to target neighbor via a typed relation.
- GraphState: The agent's current perceptual and topological state.
- SymbolicVerifier: Deterministic proof checker calculating formal rewards (GRPO-compatible).
- GraphReasoningEnv: Step-based RL environment with action masking, cycle prevention, and reward feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from mayon_cortex.core.graph import MayonGraph, ConceptNode, RelationEdge


@dataclass
class GraphAction:
    """Action taken by the reasoning policy: choose a neighbor node via relation edge."""
    source_id: str
    target_id: str
    relation_type: str
    target_text: str
    confidence: float = 1.0
    is_halt: bool = False


@dataclass
class GraphState:
    """The current state of an episode in the Graph Reasoning Environment."""
    query: str
    query_vector: np.ndarray
    current_node: ConceptNode
    visited_nodes: List[str] = field(default_factory=list)
    traversed_edges: List[Tuple[str, str, str]] = field(default_factory=list) # (source, relation, target)
    hop_count: int = 0
    done: bool = False


class SymbolicVerifier:
    """
    Deterministic Proof Verifier.
    Evaluates whether a traversed multi-hop path forms a valid, mathematically sound
    deductive chain that logically satisfies the query.
    """

    def __init__(self):
        # Known deductive rule schemas (Premise 1 + Premise 2 => Conclusion)
        self.deductive_rules = [
            # Transitive causality / pharmacology: A inhibits X, B metabolized_by X => A interacts_with B
            {"chain": ["inhibits", "metabolized_by"], "entails": "interacts_with"},
            {"chain": ["induces", "metabolized_by"], "entails": "decreases_efficacy"},
            {"chain": ["is-a", "treats"], "entails": "treats"},
            {"chain": ["part-of", "regulates"], "entails": "regulates"},
            {"chain": ["causes", "leads-to"], "entails": "causes"},
            {"chain": ["calls", "depends-on"], "entails": "depends-on"},
        ]

    def verify_path(
        self,
        traversed_edges: List[Tuple[str, str, str]],
        target_entity: Optional[str] = None,
        expected_relation: Optional[str] = None,
        mayon_nodes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """
        Verify the chain of edges.
        Returns: (is_valid_proof, reward, reason)
        """
        if not traversed_edges:
            return False, -0.5, "Empty reasoning path"

        # Check for cycles (including self-loops)
        visited = set()
        for src, _, tgt in traversed_edges:
            if src == tgt or tgt in visited:
                return False, -0.3, f"Cycle detected at node {tgt}"
            visited.add(src)
            visited.add(tgt)

        relations = [rel for _, rel, _ in traversed_edges]
        final_target = traversed_edges[-1][2]

        # Resolve target text if mayon_nodes dictionary is available
        final_target_text = final_target
        if mayon_nodes and final_target in mayon_nodes:
            final_target_text = f"{final_target} {mayon_nodes[final_target].source_text}"

        # Check if target entity reached (if target is specified)
        if target_entity is not None:
            t_low = target_entity.lower()
            if t_low not in final_target.lower() and t_low not in final_target_text.lower():
                return False, -0.4, f"Target {target_entity} not reached (ended at {final_target})"

        # Check multi-hop logical soundness
        # 1-hop direct connection
        if len(relations) == 1:
            rel_norm = relations[0].lower().replace("_", "-")
            exp_norm = expected_relation.lower().replace("_", "-") if expected_relation else None
            if exp_norm is None or exp_norm in rel_norm:
                return True, 1.0, f"Direct 1-hop proof: {relations[0]}"
            return True, 0.7, f"Direct connection via {relations[0]}"

        # 2-hop or multi-hop deductive inference
        for rule in self.deductive_rules:
            chain = rule["chain"]
            if len(relations) >= len(chain):
                # Check if chain appears as a sub-path
                for i in range(len(relations) - len(chain) + 1):
                    match = True
                    for j, required_rel in enumerate(chain):
                        req_norm = required_rel.lower().replace("_", "-")
                        rel_norm = relations[i + j].lower().replace("_", "-")
                        if req_norm not in rel_norm:
                            match = False
                            break
                    if match:
                        entailed = rule["entails"]
                        exp_norm = expected_relation.lower().replace("_", "-") if expected_relation else None
                        if exp_norm is None or exp_norm in entailed.lower().replace("_", "-"):
                            return True, 1.0, f"Sound deduction: {' -> '.join(chain)} entails {entailed}"

        # If it reached target entity with connected path but not matching a strict rule
        if target_entity is not None:
            t_low = target_entity.lower()
            if t_low in final_target.lower() or t_low in final_target_text.lower():
                return True, 0.8, f"Connected path reached target {target_entity}"

        return False, -0.2, f"Connected path but incomplete deductive proof ({' -> '.join(relations)})"


class GraphReasoningEnv:
    """
    Reinforcement Learning environment wrapping the Mayon Knowledge Graph.
    Allows an RL policy (Nucleus) to explore graph nodes and receive verifiable rewards.
    """

    def __init__(
        self,
        mayon: MayonGraph,
        verifier: Optional[SymbolicVerifier] = None,
        max_hops: int = 5,
        step_penalty: float = -0.05,
    ):
        self.mayon = mayon
        self.verifier = verifier or SymbolicVerifier()
        self.max_hops = max_hops
        self.step_penalty = step_penalty
        self.state: Optional[GraphState] = None
        self.target_entity: Optional[str] = None
        self.expected_relation: Optional[str] = None

    def reset(
        self,
        query: str,
        query_vector: np.ndarray,
        start_node_id: Optional[str] = None,
        target_entity: Optional[str] = None,
        expected_relation: Optional[str] = None,
    ) -> Tuple[GraphState, List[GraphAction]]:
        """
        Reset environment for a new reasoning episode.
        If start_node_id is not provided, seeds from nearest vector match in graph.
        """
        self.target_entity = target_entity
        self.expected_relation = expected_relation

        # Select initial node
        if start_node_id and start_node_id in self.mayon.nodes:
            start_node = self.mayon.nodes[start_node_id]
        else:
            # Fallback to vector ANN search via traverse
            subgraph = self.mayon.traverse(query_vector=query_vector, top_k=1, max_hops=0)
            if subgraph.nodes:
                start_node = subgraph.nodes[0]
            elif self.mayon.nodes:
                start_node = next(iter(self.mayon.nodes.values()))
            else:
                raise ValueError("MayonGraph has no nodes to initialize environment.")

        self.state = GraphState(
            query=query,
            query_vector=query_vector,
            current_node=start_node,
            visited_nodes=[start_node.id],
            traversed_edges=[],
            hop_count=0,
            done=False,
        )

        valid_actions = self.get_valid_actions(start_node.id)
        return self.state, valid_actions

    def get_valid_actions(self, node_id: str) -> List[GraphAction]:
        """
        Extract valid outgoing actions from the given node.
        Also includes a HALT action allowing the agent to declare its proof finished.
        """
        actions: List[GraphAction] = []

        if node_id in self.mayon.graph:
            for _, neighbor_id, edge_data in self.mayon.graph.out_edges(node_id, data=True):
                # Prevent immediate 1-hop back-tracking
                if self.state and neighbor_id in self.state.visited_nodes:
                    continue

                neighbor_node = self.mayon.nodes.get(neighbor_id)
                target_text = neighbor_node.source_text if neighbor_node else neighbor_id
                rel_type = edge_data.get("relation_type", "connected_to")
                conf = edge_data.get("confidence", 1.0)

                actions.append(
                    GraphAction(
                        source_id=node_id,
                        target_id=neighbor_id,
                        relation_type=rel_type,
                        target_text=target_text,
                        confidence=conf,
                        is_halt=False,
                    )
                )

        # Always include HALT action
        actions.append(
            GraphAction(
                source_id=node_id,
                target_id=node_id,
                relation_type="HALT",
                target_text="Conclude reasoning",
                is_halt=True,
            )
        )

        return actions

    def step(self, action: GraphAction) -> Tuple[GraphState, float, bool, Dict[str, Any]]:
        """
        Execute an action taken by the policy.
        Returns: (next_state, reward, done, info)
        """
        if self.state is None or self.state.done:
            raise RuntimeError("Environment step called on an uninitialized or finished episode.")

        info: Dict[str, Any] = {"action": action}

        # 1. Check for HALT action
        if action.is_halt:
            self.state.done = True
            is_valid, proof_reward, reason = self.verifier.verify_path(
                self.state.traversed_edges,
                target_entity=self.target_entity,
                expected_relation=self.expected_relation,
                mayon_nodes=self.mayon.nodes,
            )
            # Efficiency bonus for fewer hops
            efficiency = max(0.0, (self.max_hops - self.state.hop_count) / self.max_hops * 0.2)
            total_reward = proof_reward + (efficiency if is_valid else 0.0)
            info["is_valid_proof"] = is_valid
            info["proof_reason"] = reason
            info["efficiency_bonus"] = efficiency
            return self.state, float(total_reward), True, info

        # 2. Advance to neighbor
        target_node = self.mayon.nodes.get(action.target_id)
        if target_node is None:
            # Target node does not exist
            self.state.done = True
            return self.state, -0.5, True, {"error": f"Node {action.target_id} not found in graph"}

        self.state.traversed_edges.append((action.source_id, action.relation_type, action.target_id))
        self.state.visited_nodes.append(action.target_id)
        self.state.current_node = target_node
        self.state.hop_count += 1

        # 3. Check if max hops reached
        if self.state.hop_count >= self.max_hops:
            self.state.done = True
            is_valid, proof_reward, reason = self.verifier.verify_path(
                self.state.traversed_edges,
                target_entity=self.target_entity,
                expected_relation=self.expected_relation,
                mayon_nodes=self.mayon.nodes,
            )
            info["is_valid_proof"] = is_valid
            info["proof_reason"] = reason
            return self.state, float(proof_reward), True, info

        # Normal intermediate step with small step penalty to avoid wandering
        valid_next_actions = self.get_valid_actions(action.target_id)
        # If no non-halt actions available, must halt next
        if len(valid_next_actions) <= 1:
            info["dead_end"] = True

        return self.state, float(self.step_penalty), False, info

    def format_proof_trace(self) -> str:
        """Format the current reasoning trajectory into a formal proof trace."""
        if not self.state or not self.state.traversed_edges:
            return "No reasoning path traversed."

        steps = []
        for i, (src, rel, tgt) in enumerate(self.state.traversed_edges, 1):
            src_text = self.mayon.nodes[src].source_text if src in self.mayon.nodes else src
            tgt_text = self.mayon.nodes[tgt].source_text if tgt in self.mayon.nodes else tgt
            steps.append(f"Step {i}: [{src_text}] --({rel})--> [{tgt_text}]")

        return "\n".join(steps)
