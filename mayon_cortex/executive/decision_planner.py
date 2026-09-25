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
Decision Planner — A→Z Ordered Action Chains
==============================================
Generates ranked, ordered decision plans from cross-domain evidence.

Real intelligence doesn't give a single answer — it gives an ordered action plan:
  Step 1: Check for contraindications (SAFETY — always first)
  Step 2: Verify regulatory approval (LEGAL)
  Step 3: Evaluate clinical evidence (CLINICAL)
  Step 4: Compare costs (FINANCIAL)
  Step 5: Monitor outcomes (MONITORING)

Each step is grounded in graph evidence, ranked by priority, and linked
to its dependencies (can't do step 3 before step 1).

Usage:
    planner = DecisionPlanner(config)
    chain = planner.plan(cross_domain_evidence, graph)
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from mayon_cortex.core.config import (
    CortexConfig, DecisionPriority, RELATION_PRIORITY_MAP,
)
from mayon_cortex.reasoning.cross_domain import CrossDomainEvidence
from mayon_cortex.dynamics.salience import SalienceWeighter


@dataclass
class DecisionStep:
    """One step in an ordered decision chain."""
    rank: int                    # 1, 2, 3, ... (execution order)
    action: str                  # What to do
    reason: str                  # Why (grounded in evidence)
    sector: str                  # Which domain this step is from
    priority: int                # DecisionPriority value
    confidence: float            # How confident we are
    evidence_edges: List[Tuple]  # Graph edges backing this step
    depends_on: List[int] = field(default_factory=list)  # Step ranks this depends on


@dataclass
class DecisionChain:
    """A complete ordered decision plan."""
    steps: List[DecisionStep] = field(default_factory=list)
    query: str = ""
    primary_sector: str = "general"
    total_confidence: float = 0.0

    def add_step(self, step: DecisionStep):
        self.steps.append(step)

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass
class DecisionFactor:
    """An individual factor extracted from evidence before ordering."""
    text: str
    action: str
    sector: str
    priority: int
    confidence: float
    evidence_edges: List[Tuple]
    _depends: List = field(default_factory=list)

    def __lt__(self, other):
        # For heapq comparison (higher priority = lower number for min-heap)
        return self.priority > other.priority


class DecisionPlanner:
    """
    Generates ordered A→Z decision chains from cross-domain evidence.

    Algorithm:
    1. Extract decision factors from all evidence paths
    2. Assign priorities using relation-type → priority mapping
    3. Build dependency graph between factors
    4. Topological sort with priority weighting
    5. Output ordered DecisionChain
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.salience = SalienceWeighter(config)

    def plan(
        self,
        evidence: CrossDomainEvidence,
        graph=None,
        query: str = "",
    ) -> DecisionChain:
        """
        Generate an ordered decision chain from cross-domain evidence.

        Args:
            evidence: CrossDomainEvidence from CrossDomainResolver
            graph: MayonGraph for node text lookup
            query: Original query text

        Returns:
            DecisionChain with ordered, prioritized steps
        """
        # 1. Extract decision factors
        factors = self._extract_factors(evidence, graph)

        if not factors:
            return DecisionChain(query=query, primary_sector=evidence.primary_sector)

        # 2. Build dependency graph
        deps = self._resolve_dependencies(factors)

        # 3. Topological sort by priority + dependencies
        ordered = self._topological_priority_sort(factors, deps)

        # 4. Build decision chain
        chain = DecisionChain(
            query=query,
            primary_sector=evidence.primary_sector,
        )

        for rank, factor in enumerate(ordered, 1):
            if rank > self.config.max_decision_steps:
                break

            step = DecisionStep(
                rank=rank,
                action=factor.action,
                reason=factor.text,
                sector=factor.sector,
                priority=factor.priority,
                confidence=factor.confidence,
                evidence_edges=factor.evidence_edges,
                depends_on=[
                    ordered.index(d) + 1
                    for d in factor._depends
                    if d in ordered
                ],
            )
            chain.add_step(step)

        chain.total_confidence = (
            sum(s.confidence for s in chain.steps) / max(len(chain.steps), 1)
        )

        return chain

    def _extract_factors(
        self,
        evidence: CrossDomainEvidence,
        graph=None,
    ) -> List[DecisionFactor]:
        """
        Extract decision-relevant factors from cross-domain evidence.
        Each edge with a decision-relevant relation type becomes a factor.
        """
        factors = []
        seen_actions = set()

        for path in evidence.resolved_paths:
            for src_id, rel, tgt_id in path.edges:
                # Get priority for this relation type
                priority = RELATION_PRIORITY_MAP.get(rel, DecisionPriority.EFFICIENCY)

                # Skip low-priority or low-confidence
                if path.confidence < self.config.min_step_confidence:
                    continue

                # Get human-readable texts
                src_text = self._get_text(src_id, graph)
                tgt_text = self._get_text(tgt_id, graph)

                # Build action text
                action = self._build_action(rel, src_text, tgt_text)
                reason = f"{src_text} {rel} {tgt_text}"

                # Deduplicate
                action_key = f"{rel}:{src_text}:{tgt_text}"
                if action_key in seen_actions:
                    continue
                seen_actions.add(action_key)

                # Determine sector
                sector = "general"
                if graph:
                    if src_id in graph.nodes:
                        sector = graph.nodes[src_id].sector
                    elif tgt_id in graph.nodes:
                        sector = graph.nodes[tgt_id].sector

                factors.append(DecisionFactor(
                    text=reason,
                    action=action,
                    sector=sector,
                    priority=priority,
                    confidence=path.confidence,
                    evidence_edges=[(src_id, rel, tgt_id)],
                ))

        return factors

    def _build_action(self, relation: str, src: str, tgt: str) -> str:
        """Build a human-readable action from a relation."""
        ACTION_TEMPLATES = {
            "contraindicates": f"CHECK: {src} is contraindicated with {tgt}",
            "interacts-with":  f"CHECK: {src} interacts with {tgt}",
            "causes":          f"CHECK: {src} causes {tgt}",
            "risk-factor-for": f"CHECK: {src} is a risk factor for {tgt}",
            "prevents":        f"NOTE: {src} prevents {tgt}",
            "requires-check":  f"VERIFY: {src} requires checking {tgt}",
            "regulated-by":    f"VERIFY: {src} is regulated by {tgt}",
            "prohibits":       f"VERIFY: {tgt} prohibits {src}",
            "treats":          f"CONSIDER: {src} treats {tgt}",
            "diagnoses":       f"CONSIDER: {src} diagnoses {tgt}",
            "proves":          f"EVIDENCE: {src} proves {tgt}",
            "implies":         f"NOTE: {src} implies {tgt}",
            "costs":           f"COMPARE: {src} costs {tgt}",
            "covered-by":      f"CHECK: {src} covered by {tgt}",
            "alternative-to":  f"CONSIDER: {src} as alternative to {tgt}",
            "monitored-by":    f"SCHEDULE: {src} monitored by {tgt}",
        }
        return ACTION_TEMPLATES.get(
            relation,
            f"NOTE: {src} {relation} {tgt}",
        )

    def _resolve_dependencies(
        self, factors: List[DecisionFactor]
    ) -> Dict[int, List[int]]:
        """
        Build dependency graph between factors.

        Rules:
        - SAFETY checks must precede CLINICAL actions
        - LEGAL checks must precede CLINICAL actions
        - CLINICAL decisions must precede FINANCIAL analysis
        - FINANCIAL must precede MONITORING setup
        - Everything precedes DOCUMENTATION
        """
        DEPENDENCY_RULES = [
            (DecisionPriority.SAFETY, DecisionPriority.CLINICAL),
            (DecisionPriority.CONTRAINDICATION, DecisionPriority.CLINICAL),
            (DecisionPriority.LEGAL, DecisionPriority.CLINICAL),
            (DecisionPriority.CLINICAL, DecisionPriority.FINANCIAL),
            (DecisionPriority.CLINICAL, DecisionPriority.MONITORING),
            (DecisionPriority.FINANCIAL, DecisionPriority.MONITORING),
        ]

        deps: Dict[int, List[int]] = {i: [] for i in range(len(factors))}

        for i, factor_i in enumerate(factors):
            for j, factor_j in enumerate(factors):
                if i == j:
                    continue
                for prereq_prio, dependent_prio in DEPENDENCY_RULES:
                    if factor_j.priority == prereq_prio and factor_i.priority == dependent_prio:
                        deps[i].append(j)
                        factor_i._depends.append(factor_j)

        return deps

    def _topological_priority_sort(
        self,
        factors: List[DecisionFactor],
        deps: Dict[int, List[int]],
    ) -> List[DecisionFactor]:
        """
        Topological sort with priority weighting.
        Higher-priority factors come first, but dependencies are respected.

        Uses modified Kahn's algorithm with a priority queue.
        """
        n = len(factors)
        in_degree = {i: len(deps[i]) for i in range(n)}

        # Start with factors that have no dependencies
        # Use negative priority for max-heap behavior with heapq (min-heap)
        ready = []
        for i in range(n):
            if in_degree[i] == 0:
                heapq.heappush(ready, (-factors[i].priority, i))

        # Build reverse dependency map: who depends on factor i?
        dependents: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, dep_list in deps.items():
            for j in dep_list:
                dependents[j].append(i)

        ordered = []
        while ready:
            _, idx = heapq.heappop(ready)
            ordered.append(factors[idx])

            for dependent_idx in dependents[idx]:
                in_degree[dependent_idx] -= 1
                if in_degree[dependent_idx] == 0:
                    heapq.heappush(ready, (-factors[dependent_idx].priority, dependent_idx))

        # Handle any remaining (cycles shouldn't exist, but safety fallback)
        remaining = [factors[i] for i in range(n) if factors[i] not in ordered]
        remaining.sort(key=lambda f: -f.priority)
        ordered.extend(remaining)

        return ordered

    def _get_text(self, node_id: str, graph) -> str:
        """Get display text for a node."""
        if graph and node_id in graph.nodes:
            text = graph.nodes[node_id].source_text
            if ": " in text:
                text = text.split(": ")[0]
            return text[:80]
        return node_id
