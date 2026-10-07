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
Cross-Domain Resolver — Multi-Sector Parallel Reasoning
=========================================================
Queries ALL relevant sectors simultaneously — medical, legal, financial,
ethical — collects evidence from each, detects conflicts, and resolves
them using a priority hierarchy.

A real human making a decision considers ALL angles:
  Medical: Does the drug work?
  Legal:   Is it approved?
  Financial: Can the patient afford it?
  Ethical: Is there a cheaper equally effective alternative?

This module makes the brain do the same.

Usage:
    resolver = CrossDomainResolver(config)
    evidence = resolver.resolve(query_vec, graph, ndu, wave)
    # evidence.sector_evidence = per-sector paths
    # evidence.conflicts = detected contradictions
    # evidence.resolution = resolved cross-domain answer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig, DOMAIN_PRIORITY, OPPOSING_RELATIONS
from mayon_cortex.dynamics.activation_wave import ActivationWave, ActivationResult, ActivatedPath
from mayon_cortex.core.ndu import NodeDecisionUnit
from mayon_cortex.executive.self_correction import SelfCorrector, Contradiction


@dataclass
class DomainEvidence:
    """Evidence collected from a single sector."""
    sector: str
    paths: List[ActivatedPath]
    avg_confidence: float
    node_count: int
    top_relations: List[str]


@dataclass
class CrossDomainConflict:
    """A conflict between evidence from two different sectors."""
    sector_a: str
    sector_b: str
    relation_a: str
    relation_b: str
    entity: str
    resolution: str
    winning_sector: str


@dataclass
class CrossDomainEvidence:
    """Complete cross-domain evidence package."""
    sector_evidence: Dict[str, DomainEvidence]
    all_paths: List[ActivatedPath]
    conflicts: List[CrossDomainConflict]
    resolved_paths: List[ActivatedPath]  # after conflict resolution
    relevant_sectors: List[str]
    primary_sector: str


class CrossDomainResolver:
    """
    Multi-sector parallel reasoning engine.

    Algorithm:
    1. Detect which sectors are relevant to the query
    2. Launch parallel activation waves — one per sector
    3. Also launch an unfiltered cross-domain wave
    4. Detect conflicts between sectors
    5. Resolve conflicts using priority hierarchy
    6. Produce unified evidence package
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.corrector = SelfCorrector(config)
        self.domain_priority = dict(DOMAIN_PRIORITY)

    def resolve(
        self,
        query_vec: np.ndarray,
        graph,
        ndu: NodeDecisionUnit,
        wave: ActivationWave,
    ) -> CrossDomainEvidence:
        """
        Run cross-domain parallel reasoning.

        1. Detect relevant sectors
        2. Launch parallel sector-filtered waves
        3. Launch unfiltered wave for cross-domain links
        4. Detect and resolve conflicts
        5. Return unified evidence
        """
        # 1. Detect relevant sectors
        relevant_sectors = self._detect_relevant_sectors(query_vec, graph)

        if len(relevant_sectors) < self.config.cross_domain_min_sectors:
            # Not enough sectors — just do single-domain
            result = wave.activate(query_vec, graph, ndu)
            primary = relevant_sectors[0] if relevant_sectors else "general"
            return CrossDomainEvidence(
                sector_evidence={primary: DomainEvidence(
                    sector=primary,
                    paths=result.paths,
                    avg_confidence=self._avg_confidence(result.paths),
                    node_count=len(result.active_node_ids),
                    top_relations=self._top_relations(result.paths),
                )},
                all_paths=result.paths,
                conflicts=[],
                resolved_paths=result.paths,
                relevant_sectors=relevant_sectors,
                primary_sector=primary,
            )

        # 2. Launch parallel sector-filtered waves
        sector_evidence: Dict[str, DomainEvidence] = {}
        all_paths: List[ActivatedPath] = []

        for sector in relevant_sectors:
            result = wave.activate(query_vec, graph, ndu, sector_filter=sector)
            sector_evidence[sector] = DomainEvidence(
                sector=sector,
                paths=result.paths,
                avg_confidence=self._avg_confidence(result.paths),
                node_count=len(result.active_node_ids),
                top_relations=self._top_relations(result.paths),
            )
            all_paths.extend(result.paths)

        # 3. Also launch unfiltered wave for cross-domain connections
        cross_result = wave.activate(query_vec, graph, ndu, sector_filter=None)
        all_paths.extend(cross_result.paths)

        # 4. Detect conflicts between sectors
        conflicts = self._detect_cross_domain_conflicts(sector_evidence, graph)

        # 5. Resolve conflicts
        resolved_paths = self._resolve_conflicts(all_paths, conflicts, graph)

        # 6. Determine primary sector
        primary = max(
            relevant_sectors,
            key=lambda s: sector_evidence.get(s, DomainEvidence(s, [], 0.0, 0, [])).avg_confidence,
        )

        return CrossDomainEvidence(
            sector_evidence=sector_evidence,
            all_paths=all_paths,
            conflicts=conflicts,
            resolved_paths=resolved_paths,
            relevant_sectors=relevant_sectors,
            primary_sector=primary,
        )

    def _detect_relevant_sectors(
        self, query_vec: np.ndarray, graph
    ) -> List[str]:
        """
        Detect which sectors are relevant to the query.
        Uses FAISS to find seed nodes and checks their sectors.
        """
        if graph.num_nodes == 0:
            return ["general"]

        q_vec = query_vec.astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        k = min(20, graph.num_nodes)
        scores, indices = graph.faiss_index.search(q_vec.reshape(1, -1), k)

        sector_scores: Dict[str, float] = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            nid = graph._faiss_idx_to_id.get(int(idx))
            if nid and nid in graph.nodes:
                sector = graph.nodes[nid].sector
                if sector not in sector_scores:
                    sector_scores[sector] = 0.0
                sector_scores[sector] = max(sector_scores[sector], float(score))

        # Filter by relevance threshold
        relevant = [
            s for s, score in sector_scores.items()
            if score >= self.config.sector_relevance_threshold
        ]

        # Always include "general" if anything else is found
        if relevant and "general" not in relevant:
            if "general" in sector_scores:
                relevant.append("general")

        return relevant if relevant else ["general"]

    def _detect_cross_domain_conflicts(
        self,
        sector_evidence: Dict[str, DomainEvidence],
        graph,
    ) -> List[CrossDomainConflict]:
        """Detect conflicts between evidence from different sectors."""
        conflicts = []
        sectors = list(sector_evidence.keys())

        for i, s1 in enumerate(sectors):
            for s2 in sectors[i + 1:]:
                ev1 = sector_evidence[s1]
                ev2 = sector_evidence[s2]

                for path_a in ev1.paths:
                    for path_b in ev2.paths:
                        conflict = self._check_path_conflict(
                            path_a, path_b, s1, s2, graph
                        )
                        if conflict:
                            conflicts.append(conflict)

        return conflicts

    def _check_path_conflict(
        self, path_a, path_b, sector_a, sector_b, graph
    ) -> Optional[CrossDomainConflict]:
        """Check if two paths from different sectors conflict."""
        opposing = dict(OPPOSING_RELATIONS)
        # Build reverse mapping
        for k, v in list(opposing.items()):
            opposing[v] = k

        for src_a, rel_a, tgt_a in path_a.edges:
            for src_b, rel_b, tgt_b in path_b.edges:
                # Check same entity pair
                same_pair = (
                    (src_a == src_b and tgt_a == tgt_b) or
                    (src_a == tgt_b and tgt_a == src_b)
                )
                if not same_pair:
                    continue

                # Check opposing relations
                if opposing.get(rel_a) == rel_b or opposing.get(rel_b) == rel_a:
                    # Resolve by domain priority
                    prio_a = self.domain_priority.get(sector_a, 30)
                    prio_b = self.domain_priority.get(sector_b, 30)
                    winning = sector_a if prio_a >= prio_b else sector_b

                    entity = src_a
                    if graph and src_a in graph.nodes:
                        entity = graph.nodes[src_a].source_text[:50]

                    return CrossDomainConflict(
                        sector_a=sector_a,
                        sector_b=sector_b,
                        relation_a=rel_a,
                        relation_b=rel_b,
                        entity=entity,
                        resolution=f"{winning} sector takes priority ({self.domain_priority.get(winning, 30)} > {self.domain_priority.get(sector_a if winning == sector_b else sector_b, 30)})",
                        winning_sector=winning,
                    )

        return None

    def _resolve_conflicts(
        self,
        all_paths: List[ActivatedPath],
        conflicts: List[CrossDomainConflict],
        graph,
    ) -> List[ActivatedPath]:
        """Remove losing paths from conflicts, keep winning ones."""
        if not conflicts:
            return all_paths

        # Use self-corrector for general contradiction resolution
        clean_paths, _ = self.corrector.correct(all_paths, graph)
        return clean_paths

    def _avg_confidence(self, paths: List[ActivatedPath]) -> float:
        if not paths:
            return 0.0
        return float(np.mean([p.confidence for p in paths]))

    def _top_relations(self, paths: List[ActivatedPath]) -> List[str]:
        rel_counts: Dict[str, int] = {}
        for p in paths:
            for _, rel, _ in p.edges:
                rel_counts[rel] = rel_counts.get(rel, 0) + 1
        sorted_rels = sorted(rel_counts, key=rel_counts.get, reverse=True)
        return sorted_rels[:5]
