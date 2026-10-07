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
Logical Chainer — Forward & Backward Inference on Graph Edges
===============================================================
Phase 1 of the Brain-Like Intelligence upgrade.

Implements 5 domain-agnostic logical inference rules that operate
directly on graph edges. No external reasoner — pure graph operations.

Rules:
  1. Transitivity:   A→B, B→C  ⟹  A→C  (for transitive relations like is-a, part-of)
  2. Contrapositive:  A causes B  ⟹  ¬B implies ¬A  (if no B, then no A caused it)
  3. Inheritance:     A is-a B, B has-property P  ⟹  A has-property P
  4. Exclusion:       A contraindicates B, B treats C  ⟹  A may prevent treatment of C
  5. Symmetry:        A similar-to B  ⟹  B similar-to A  (for symmetric relations)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class InferenceRule(Enum):
    """The five core inference rules."""
    TRANSITIVITY = "transitivity"
    CONTRAPOSITIVE = "contrapositive"
    INHERITANCE = "inheritance"
    EXCLUSION = "exclusion"
    SYMMETRY = "symmetry"


# Relations that support transitivity (chaining)
TRANSITIVE_RELATIONS = {
    "is-a", "part-of", "located-in", "derived-from", "causes",
    "enables", "precedes", "implies", "inherits",
}

# Relations that support inheritance (properties flow up/down)
INHERITABLE_RELATIONS = {
    "has-property", "treats", "contraindicates", "manifests-as",
    "uses", "requires-check", "monitored-by",
}

# Relations that are symmetric
SYMMETRIC_RELATIONS = {
    "same-as", "similar-to", "interacts-with", "adjacent-to",
    "analogous-to", "correlated-with",
}

# Opposing relation pairs
OPPOSING_PAIRS = {
    "treats": "contraindicates",
    "enables": "prevents",
    "implies": "contradicts",
    "causes": "prevents",
    "approved": "prohibits",
}


@dataclass
class InferenceStep:
    """A single inference step in a chain."""
    rule: InferenceRule
    premise_edges: List[Tuple[str, str, str]]  # (source, relation, target)
    conclusion: Tuple[str, str, str]            # (source, relation, target)
    confidence: float
    explanation: str


@dataclass
class InferenceChain:
    """A complete chain of inferences leading to a conclusion."""
    query: str
    steps: List[InferenceStep] = field(default_factory=list)
    final_conclusion: str = ""
    total_confidence: float = 0.0
    rules_applied: List[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.steps)


class LogicalChainer:

    """
    Forward and backward logical inference on the knowledge graph.
    
    Forward chaining: Start from known facts → apply rules → derive new facts
    Backward chaining: Start from goal → find what rules could prove it → verify premises
    
    All inference is domain-agnostic — rules operate on relation types, not content.
    """

    def __init__(self, max_chain_depth: int = 6, min_confidence: float = 0.1):
        self.max_chain_depth = max_chain_depth
        self.min_confidence = min_confidence

    def forward_chain(
        self,
        start_node_id: str,
        graph: Any,
        target_relation: Optional[str] = None,
        max_steps: int = 0,
    ) -> List[InferenceChain]:
        """
        Forward chaining: start from a node, apply rules to derive new facts.
        
        Args:
            start_node_id: Starting node in the graph
            graph: MayonGraph instance
            target_relation: Optional — stop when this relation type is found
            max_steps: Override max chain depth (0 = use default)
            
        Returns:
            List of inference chains with derived conclusions
        """
        max_steps = max_steps or self.max_chain_depth
        if start_node_id not in graph.nodes:
            return []

        chains: List[InferenceChain] = []
        visited: Set[str] = {start_node_id}

        self._forward_recursive(
            node_id=start_node_id,
            graph=graph,
            chain=InferenceChain(query=graph.nodes[start_node_id].source_text),
            visited=visited,
            depth=0,
            max_steps=max_steps,
            target_relation=target_relation,
            chains=chains,
        )
        return chains

    def _forward_recursive(
        self,
        node_id: str,
        graph: Any,
        chain: InferenceChain,
        visited: Set[str],
        depth: int,
        max_steps: int,
        target_relation: Optional[str],
        chains: List[InferenceChain],
    ) -> None:
        """Recursive forward chaining from a node."""
        if depth >= max_steps:
            if chain.steps:
                chain.total_confidence = self._chain_confidence(chain)
                chain.final_conclusion = chain.steps[-1].conclusion[2] if chain.steps else ""
                chains.append(chain)
            return

        node = graph.nodes.get(node_id)
        if not node:
            return

        # Get outgoing edges
        out_edges = graph._out_edges.get(node_id, [])
        produced_inference = False

        for eid in out_edges:
            if eid not in graph.edges:
                continue
            edge = graph.edges[eid]
            if not edge.is_active or edge.confidence < self.min_confidence:
                continue
            if edge.target_id in visited:
                continue

            target = graph.nodes.get(edge.target_id)
            if not target:
                continue

            # Try each inference rule
            new_steps = self._apply_rules_forward(
                node_id, edge, target, graph, visited
            )

            for step in new_steps:
                new_chain = InferenceChain(
                    query=chain.query,
                    steps=chain.steps + [step],
                    rules_applied=chain.rules_applied + [step.rule.value],
                )

                # Check if we found what we're looking for
                if target_relation and step.conclusion[1] == target_relation:
                    new_chain.total_confidence = self._chain_confidence(new_chain)
                    new_chain.final_conclusion = step.explanation
                    chains.append(new_chain)
                    produced_inference = True
                    continue

                # Continue exploring
                visited.add(edge.target_id)
                self._forward_recursive(
                    edge.target_id, graph, new_chain, visited,
                    depth + 1, max_steps, target_relation, chains,
                )
                visited.discard(edge.target_id)
                produced_inference = True

        # If no more inferences can be made, save current chain
        if not produced_inference and chain.steps:
            chain.total_confidence = self._chain_confidence(chain)
            chain.final_conclusion = chain.steps[-1].explanation
            chains.append(chain)

    def backward_chain(
        self,
        goal: str,
        goal_relation: str,
        graph: Any,
        embedder: Any,
        max_steps: int = 0,
    ) -> List[InferenceChain]:
        """
        Backward chaining: start from a goal, find what could prove it.
        
        Args:
            goal: The text of what we want to prove (e.g., "Lisinopril treats hypertension")
            goal_relation: The relation type to prove (e.g., "treats")
            graph: MayonGraph instance
            embedder: Embedding model for text→vector
            max_steps: Override max depth
            
        Returns:
            List of inference chains that support the goal
        """
        max_steps = max_steps or self.max_chain_depth
        goal_vec = embedder.encode_single(goal)

        # Find candidate nodes related to the goal
        if graph.num_nodes == 0:
            return []

        norm = np.linalg.norm(goal_vec)
        if norm > 0:
            q_normed = (goal_vec / norm).reshape(1, -1).astype(np.float32)
            scores, indices = graph.faiss_index.search(q_normed, min(10, graph.num_nodes))
        else:
            return []

        id_list = list(graph.nodes.keys())
        chains: List[InferenceChain] = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(id_list):
                continue
            nid = id_list[idx]
            # Try to build backward chain from this node to the goal
            result = self._backward_recursive(
                nid, goal_relation, graph, set(), 0, max_steps
            )
            for r_chain in result:
                r_chain.query = goal
                r_chain.total_confidence = self._chain_confidence(r_chain)
                chains.append(r_chain)

        # Sort by confidence
        chains.sort(key=lambda c: c.total_confidence, reverse=True)
        return chains[:5]

    def _backward_recursive(
        self,
        node_id: str,
        target_relation: str,
        graph: Any,
        visited: Set[str],
        depth: int,
        max_depth: int,
    ) -> List[InferenceChain]:
        """Recursive backward search for supporting evidence."""
        if depth >= max_depth or node_id in visited:
            return []

        visited.add(node_id)
        chains: List[InferenceChain] = []
        node = graph.nodes.get(node_id)
        if not node:
            return []

        # Check incoming edges — what leads to this node?
        in_edges = graph._in_edges.get(node_id, [])
        for eid in in_edges:
            if eid not in graph.edges:
                continue
            edge = graph.edges[eid]
            if not edge.is_active:
                continue

            source = graph.nodes.get(edge.source_id)
            if not source or edge.source_id in visited:
                continue

            # Direct evidence found
            if edge.relation_type.lower() == target_relation.lower():
                step = InferenceStep(
                    rule=InferenceRule.TRANSITIVITY,
                    premise_edges=[(edge.source_id, edge.relation_type, node_id)],
                    conclusion=(edge.source_id, edge.relation_type, node_id),
                    confidence=edge.confidence,
                    explanation=f"{source.source_text} {edge.relation_type} {node.source_text}",
                )
                chains.append(InferenceChain(query="", steps=[step]))

            # Try transitive inference
            if edge.relation_type.lower() in TRANSITIVE_RELATIONS:
                sub_chains = self._backward_recursive(
                    edge.source_id, target_relation, graph, visited, depth + 1, max_depth
                )
                for sc in sub_chains:
                    step = InferenceStep(
                        rule=InferenceRule.TRANSITIVITY,
                        premise_edges=[(edge.source_id, edge.relation_type, node_id)],
                        conclusion=(edge.source_id, "transitive-link", node_id),
                        confidence=edge.confidence,
                        explanation=f"{source.source_text} →{edge.relation_type}→ {node.source_text}",
                    )
                    sc.steps.append(step)
                    chains.append(sc)

        visited.discard(node_id)
        return chains

    def _apply_rules_forward(
        self,
        source_id: str,
        edge: Any,
        target: Any,
        graph: Any,
        visited: Set[str],
    ) -> List[InferenceStep]:
        """Apply all applicable inference rules at this edge."""
        steps: List[InferenceStep] = []
        source = graph.nodes.get(source_id)
        if not source:
            return steps

        rel = edge.relation_type.lower()

        # Rule 1: Transitivity
        if rel in TRANSITIVE_RELATIONS:
            steps.append(InferenceStep(
                rule=InferenceRule.TRANSITIVITY,
                premise_edges=[(source_id, edge.relation_type, edge.target_id)],
                conclusion=(source_id, edge.relation_type, edge.target_id),
                confidence=edge.confidence,
                explanation=f"{source.source_text} {edge.relation_type} {target.source_text}",
            ))

        # Rule 3: Inheritance — propagate properties
        if rel in {"is-a", "inherits"}:
            for t_eid in graph._out_edges.get(edge.target_id, []):
                if t_eid in graph.edges:
                    t_edge = graph.edges[t_eid]
                    if t_edge.relation_type.lower() in INHERITABLE_RELATIONS:
                        t_target = graph.nodes.get(t_edge.target_id)
                        if t_target and t_edge.target_id not in visited:
                            inherited_conf = edge.confidence * t_edge.confidence * 0.9
                            steps.append(InferenceStep(
                                rule=InferenceRule.INHERITANCE,
                                premise_edges=[
                                    (source_id, edge.relation_type, edge.target_id),
                                    (edge.target_id, t_edge.relation_type, t_edge.target_id),
                                ],
                                conclusion=(source_id, t_edge.relation_type, t_edge.target_id),
                                confidence=inherited_conf,
                                explanation=(
                                    f"{source.source_text} inherits '{t_edge.relation_type}' "
                                    f"from {target.source_text}: "
                                    f"{source.source_text} {t_edge.relation_type} {t_target.source_text}"
                                ),
                            ))

        # Rule 4: Exclusion — detect conflicts
        if rel in OPPOSING_PAIRS:
            opposing_rel = OPPOSING_PAIRS[rel]
            for t_eid in graph._out_edges.get(edge.target_id, []):
                if t_eid in graph.edges:
                    t_edge = graph.edges[t_eid]
                    if t_edge.relation_type.lower() == opposing_rel:
                        t_target = graph.nodes.get(t_edge.target_id)
                        if t_target:
                            steps.append(InferenceStep(
                                rule=InferenceRule.EXCLUSION,
                                premise_edges=[
                                    (source_id, edge.relation_type, edge.target_id),
                                    (edge.target_id, t_edge.relation_type, t_edge.target_id),
                                ],
                                conclusion=(source_id, "conflicts-with", t_edge.target_id),
                                confidence=edge.confidence * t_edge.confidence,
                                explanation=(
                                    f"CONFLICT: {source.source_text} {edge.relation_type} "
                                    f"{target.source_text}, but {target.source_text} "
                                    f"{t_edge.relation_type} {t_target.source_text}"
                                ),
                            ))

        # Rule 5: Symmetry
        if rel in SYMMETRIC_RELATIONS:
            steps.append(InferenceStep(
                rule=InferenceRule.SYMMETRY,
                premise_edges=[(source_id, edge.relation_type, edge.target_id)],
                conclusion=(edge.target_id, edge.relation_type, source_id),
                confidence=edge.confidence,
                explanation=f"By symmetry: {target.source_text} {edge.relation_type} {source.source_text}",
            ))

        return steps

    def _chain_confidence(self, chain: InferenceChain) -> float:
        """Compute overall confidence of an inference chain (multiplicative decay)."""
        if not chain.steps:
            return 0.0
        conf = 1.0
        for step in chain.steps:
            conf *= step.confidence
        # Apply length penalty (longer chains are less certain)
        length_penalty = 0.95 ** max(0, len(chain.steps) - 1)
        return conf * length_penalty

    def infer_all(
        self,
        node_id: str,
        graph: Any,
        max_steps: int = 0,
    ) -> List[InferenceStep]:
        """
        Apply all inference rules reachable from a node (1-hop).
        Useful for enriching reasoning without full chain search.
        """
        max_steps = max_steps or 1
        if node_id not in graph.nodes:
            return []

        all_steps: List[InferenceStep] = []
        visited: Set[str] = {node_id}

        for eid in graph._out_edges.get(node_id, []):
            if eid not in graph.edges:
                continue
            edge = graph.edges[eid]
            target = graph.nodes.get(edge.target_id)
            if target:
                steps = self._apply_rules_forward(node_id, edge, target, graph, visited)
                all_steps.extend(steps)

        return all_steps


# Aliases
ProofStep = InferenceStep
FormalProof = InferenceChain
LogicEngine = LogicalChainer

