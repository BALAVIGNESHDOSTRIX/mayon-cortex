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
Uncertainty Gate — Epistemic Humility
======================================
Real intelligence admits when it doesn't know.

If the graph has no path, or all paths are low-confidence, or query
coverage is poor, the system must say "I don't have enough evidence"
instead of guessing. This prevents hallucination at the architectural
level — not as a filter, but as a fundamental design property.

Usage:
    gate = UncertaintyGate(config)
    assessment = gate.evaluate(paths, query_vec, graph)
    if assessment.is_uncertain:
        return f"I don't know: {assessment.reason}. But I do know: {assessment.partial}"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class UncertaintyAssessment:
    """Result of uncertainty evaluation."""
    is_uncertain: bool                # True = don't have enough evidence
    confidence: float                 # Overall confidence in the answer
    reason: str                       # Why we're uncertain (if uncertain)
    partial_knowledge: List[str]      # What we DO know (even if uncertain)
    missing_evidence: List[str]       # What evidence we'd need
    coverage: float                   # What fraction of query is covered


class UncertaintyGate:
    """
    Evaluates whether the brain has enough evidence to answer confidently.

    Checks:
    1. Path count — do we have ANY paths? (minimum 1)
    2. Path confidence — are the paths high-confidence? (above threshold)
    3. Query coverage — do the paths actually address the question? (similarity)
    4. Evidence diversity — are we seeing corroborating evidence from multiple sources?
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def evaluate(
        self,
        paths: List,  # List of ranked paths from PathRanker
        query_vec: np.ndarray,
        graph,  # MayonGraph
    ) -> UncertaintyAssessment:
        """
        Evaluate if we have sufficient evidence to answer.

        Returns UncertaintyAssessment with is_uncertain=True if:
        - No paths found at all
        - All paths below confidence threshold
        - Query coverage is too low (paths don't address the question)
        """
        # Case 1: No paths at all
        if not paths:
            return UncertaintyAssessment(
                is_uncertain=True,
                confidence=0.0,
                reason="No relevant knowledge found in the graph",
                partial_knowledge=[],
                missing_evidence=["Any information related to this query"],
                coverage=0.0,
            )

        # Compute path confidences
        path_confidences = []
        for path in paths:
            if hasattr(path, "confidence"):
                path_confidences.append(path.confidence)
            elif hasattr(path, "score"):
                path_confidences.append(path.score)
            else:
                path_confidences.append(0.5)

        max_confidence = max(path_confidences) if path_confidences else 0.0
        avg_confidence = np.mean(path_confidences) if path_confidences else 0.0

        # Case 2: All paths below confidence threshold
        if max_confidence < self.config.uncertainty_min_confidence:
            partial = self._extract_partial_knowledge(paths, graph)
            return UncertaintyAssessment(
                is_uncertain=True,
                confidence=max_confidence,
                reason=f"Evidence confidence too low ({max_confidence:.2f} < {self.config.uncertainty_min_confidence})",
                partial_knowledge=partial,
                missing_evidence=["Higher-confidence evidence for this topic"],
                coverage=0.0,
            )

        # Case 3: Check query coverage (do paths address the question?)
        coverage = self._compute_query_coverage(paths, query_vec, graph)
        if coverage < self.config.uncertainty_min_coverage:
            partial = self._extract_partial_knowledge(paths, graph)
            return UncertaintyAssessment(
                is_uncertain=True,
                confidence=avg_confidence,
                reason=f"Query coverage too low ({coverage:.2f}). Found related but not directly relevant information",
                partial_knowledge=partial,
                missing_evidence=["Direct evidence addressing the specific question"],
                coverage=coverage,
            )

        # Case 4: Sufficient evidence — confident answer
        return UncertaintyAssessment(
            is_uncertain=False,
            confidence=avg_confidence,
            reason="",
            partial_knowledge=[],
            missing_evidence=[],
            coverage=coverage,
        )

    def _compute_query_coverage(
        self, paths: List, query_vec: np.ndarray, graph
    ) -> float:
        """
        How well do the retrieved paths cover the query?
        Measured by max cosine similarity between query and endpoint nodes.
        """
        max_sim = 0.0
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return 0.0

        for path in paths:
            nodes = []
            if hasattr(path, "nodes"):
                nodes = path.nodes
            elif hasattr(path, "node_ids") and graph is not None:
                nodes = [graph.nodes[nid] for nid in path.node_ids if nid in graph.nodes]

            for node in nodes:
                if hasattr(node, "vector"):
                    n_norm = np.linalg.norm(node.vector)
                    if n_norm > 0:
                        sim = float(np.dot(query_vec, node.vector) / (q_norm * n_norm))
                        max_sim = max(max_sim, sim)

        return max_sim

    def _extract_partial_knowledge(self, paths: List, graph) -> List[str]:
        """Extract what we DO know, even if it's not enough for a full answer."""
        partial = []
        seen = set()
        for path in paths[:3]:  # Top 3 paths only
            nodes = []
            if hasattr(path, "nodes"):
                nodes = path.nodes
            elif hasattr(path, "node_ids") and graph is not None:
                nodes = [graph.nodes[nid] for nid in path.node_ids if nid in graph.nodes]

            for node in nodes:
                text = getattr(node, "source_text", str(node))
                if text not in seen and len(text) > 5:
                    partial.append(text[:150])
                    seen.add(text)
                    if len(partial) >= 5:
                        return partial

        return partial

    def format_uncertain_response(self, assessment: UncertaintyAssessment) -> str:
        """Format an uncertainty response as human-readable text."""
        parts = [f"I don't have sufficient evidence to fully answer this. {assessment.reason}."]

        if assessment.partial_knowledge:
            parts.append("\nWhat I do know:")
            for i, fact in enumerate(assessment.partial_knowledge, 1):
                parts.append(f"  {i}. {fact}")

        if assessment.missing_evidence:
            parts.append("\nTo answer fully, I would need:")
            for item in assessment.missing_evidence:
                parts.append(f"  • {item}")

        return "\n".join(parts)
