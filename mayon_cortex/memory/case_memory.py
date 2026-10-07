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
Case Memory — Precedent Storage & Retrieval
=============================================
Phase 7 of the Brain-Like Intelligence upgrade.

Stores structured cases (situation → judgment → reasoning) as subgraph
patterns. Each case becomes a connected cluster of nodes in the knowledge graph.

Like a legal system: store past cases, find similar ones, adapt precedent.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class Case:
    """A stored case with situation, judgment, and reasoning."""
    case_id: str
    situation: str
    situation_vec: Optional[np.ndarray] = None
    judgment: str = ""
    reasoning: str = ""
    outcome: Optional[str] = None
    sector: str = "general"
    factors: List[str] = field(default_factory=list)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.9
    node_ids: List[str] = field(default_factory=list)  # Graph node IDs for this case

    verdict: Optional[str] = None
    rationale: Optional[str] = None

    def __post_init__(self):
        if self.verdict and not self.judgment:
            self.judgment = self.verdict
        if self.rationale and not self.reasoning:
            self.reasoning = self.rationale


# Aliases
PrecedentCase = Case


@dataclass
class CaseMatch:
    """A matched precedent case with similarity analysis."""
    case: Case
    similarity: float
    shared_factors: List[str]
    different_factors: List[str]
    adaptation_notes: str


class CaseMemory:
    """
    Stores and retrieves cases as graph subpatterns.
    """

    def __init__(self, config: Optional[CortexConfig] = None, capacity: int = 2000):
        self.config = config or CortexConfig()
        self.capacity = capacity
        self.cases: Dict[str, Case] = {}
        self._case_counter = 0

    @property
    def size(self) -> int:
        return len(self.cases)

    def store_case(
        self,
        situation: Any = None,
        judgment: str = "",
        reasoning: str = "",
        graph: Optional[Any] = None,
        embedder: Optional[Any] = None,
        sector: str = "general",
        factors: Optional[List[str]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        outcome: Optional[str] = None,
    ) -> str:
        """
        Store a case as a connected subgraph or memory object.
        """
        if isinstance(situation, Case):
            c = situation
            self.cases[c.case_id] = c
            return c.case_id

        sit_text = situation or ""

        self._case_counter += 1
        case_id = f"case_{self._case_counter}_{int(time.time())}"
        factors = factors or []
        factor_weights = factor_weights or {}

        if graph is None or embedder is None:
            c = Case(
                case_id=case_id,
                situation=situation,
                judgment=judgment,
                reasoning=reasoning,
                outcome=outcome,
                sector=sector,
                factors=factors,
                factor_weights=factor_weights,
            )
            self.cases[case_id] = c
            return case_id

        """
        Store a case as a connected subgraph.
        
        Creates:
          situation_node --led-to--> judgment_node
          judgment_node --because--> reasoning_node
          situation_node --has-factor--> factor_nodes
          
        Returns case_id.
        """
        self._case_counter += 1
        case_id = f"case_{self._case_counter}_{int(time.time())}"
        factors = factors or []
        factor_weights = factor_weights or {}

        # Embed
        sit_vec = embedder.encode_single(situation)
        jdg_vec = embedder.encode_single(judgment)
        rsn_vec = embedder.encode_single(reasoning)

        node_ids = []

        # Create situation node
        sit_node = graph.add_node(
            vector=sit_vec,
            source_text=f"[CASE {case_id}] Situation: {situation[:300]}",
            level=0,
            sector=sector,
            node_type="case_situation",
            provenance=f"case_memory:{case_id}",
            confidence=0.95,
            metadata={"case_id": case_id, "type": "situation"},
        )
        node_ids.append(sit_node.id)

        # Create judgment node
        jdg_node = graph.add_node(
            vector=jdg_vec,
            source_text=f"[CASE {case_id}] Judgment: {judgment[:300]}",
            level=0,
            sector=sector,
            node_type="case_judgment",
            provenance=f"case_memory:{case_id}",
            confidence=0.95,
            metadata={"case_id": case_id, "type": "judgment"},
        )
        node_ids.append(jdg_node.id)

        # Create reasoning node
        rsn_node = graph.add_node(
            vector=rsn_vec,
            source_text=f"[CASE {case_id}] Reasoning: {reasoning[:300]}",
            level=0,
            sector=sector,
            node_type="case_reasoning",
            provenance=f"case_memory:{case_id}",
            confidence=0.95,
            metadata={"case_id": case_id, "type": "reasoning"},
        )
        node_ids.append(rsn_node.id)

        # Create edges: situation → judgment → reasoning
        try:
            graph.add_edge(
                source_id=sit_node.id,
                target_id=jdg_node.id,
                relation_type="led-to",
                confidence=0.95,
                sector=sector,
                provenance=f"case_memory:{case_id}",
                metadata={"case_id": case_id},
            )
            graph.add_edge(
                source_id=jdg_node.id,
                target_id=rsn_node.id,
                relation_type="because",
                confidence=0.95,
                sector=sector,
                provenance=f"case_memory:{case_id}",
                metadata={"case_id": case_id},
            )
        except Exception:
            pass

        # Create factor nodes and edges
        for factor in factors:
            try:
                fac_vec = embedder.encode_single(factor)
                fac_node = graph.add_node(
                    vector=fac_vec,
                    source_text=f"[CASE {case_id}] Factor: {factor}",
                    level=0,
                    sector=sector,
                    node_type="case_factor",
                    provenance=f"case_memory:{case_id}",
                    confidence=0.9,
                    metadata={
                        "case_id": case_id,
                        "type": "factor",
                        "weight": factor_weights.get(factor, 0.5),
                    },
                )
                node_ids.append(fac_node.id)

                graph.add_edge(
                    source_id=sit_node.id,
                    target_id=fac_node.id,
                    relation_type="has-factor",
                    confidence=0.9,
                    sector=sector,
                    provenance=f"case_memory:{case_id}",
                )

                weight = factor_weights.get(factor, 0.5)
                graph.add_edge(
                    source_id=fac_node.id,
                    target_id=jdg_node.id,
                    relation_type="influenced",
                    confidence=weight,
                    sector=sector,
                    provenance=f"case_memory:{case_id}",
                    metadata={"weight": weight},
                )
            except Exception:
                pass

        # Store case record
        case = Case(
            case_id=case_id,
            situation=situation,
            situation_vec=sit_vec,
            judgment=judgment,
            reasoning=reasoning,
            outcome=outcome,
            sector=sector,
            factors=factors,
            factor_weights=factor_weights,
            confidence=0.95,
            node_ids=node_ids,
        )
        self.cases[case_id] = case

        return case_id

    def find_precedents(
        self,
        new_situation: str,
        graph: Any = None,
        embedder: Any = None,
        top_k: int = 5,
    ) -> List[CaseMatch]:
        """
        Find the most similar past cases to a new situation.
        """
        if not self.cases:
            return []

        new_vec = None
        new_norm = 0.0
        if embedder is not None:
            try:
                new_vec = embedder.encode_single(new_situation)
                new_norm = float(np.linalg.norm(new_vec))
            except Exception:
                new_vec = None

        new_words = set(new_situation.lower().split())

        # Compare against all stored cases
        matches: List[CaseMatch] = []

        for case_id, case in self.cases.items():
            sim = 0.0
            if new_vec is not None and case.situation_vec is not None and new_norm > 1e-8:
                c_norm = float(np.linalg.norm(case.situation_vec))
                if c_norm > 1e-8:
                    sim = float(np.dot(new_vec, case.situation_vec) / (new_norm * c_norm))
            else:
                # Fallback word-overlap Jaccard similarity
                case_words = set(case.situation.lower().split())
                union = len(new_words | case_words)
                if union > 0:
                    sim = len(new_words & case_words) / union

            if sim > 0.05 or len(self.cases) <= 5:
                # Extract factors from new situation (basic keyword matching)
                new_factors = self._extract_factors_basic(new_situation)
                shared, different = self._compare_factors(case.factors, new_factors)

                adaptation = self._generate_adaptation_notes(
                    case, shared, different, sim
                )

                matches.append(CaseMatch(
                    case=case,
                    similarity=sim,
                    shared_factors=shared,
                    different_factors=different,
                    adaptation_notes=adaptation,
                ))

        # Sort by similarity
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    def _compare_factors(
        self,
        case_factors: List[str],
        new_factors: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Identify shared and different factors between cases."""
        case_set = set(f.lower() for f in case_factors)
        new_set = set(f.lower() for f in new_factors)

        shared = list(case_set & new_set)
        different = list((case_set | new_set) - (case_set & new_set))

        return shared, different

    def _extract_factors_basic(self, text: str) -> List[str]:
        """Basic factor extraction from text using keyword matching."""
        import re
        words = re.findall(r'\b[a-z_]+\b', text.lower())
        # Filter for significant words (length > 4, not common)
        common = {"about", "after", "being", "could", "every", "first", "their",
                  "there", "these", "think", "those", "under", "which", "would"}
        return [w for w in words if len(w) > 4 and w not in common][:10]

    def _generate_adaptation_notes(
        self,
        case: Case,
        shared_factors: List[str],
        different_factors: List[str],
        similarity: float,
    ) -> str:
        """Generate notes on how to adapt the precedent to the new case."""
        parts = []

        if similarity > 0.8:
            parts.append("Strong precedent match — judgment likely directly applicable.")
        elif similarity > 0.6:
            parts.append("Moderate precedent match — judgment may need adaptation.")
        else:
            parts.append("Weak precedent match — use with caution, significant differences exist.")

        if shared_factors:
            parts.append(f"Shared factors: {', '.join(shared_factors[:5])}")
        if different_factors:
            parts.append(f"Different factors: {', '.join(different_factors[:5])}")
            parts.append("Consider how these differences affect the judgment.")

        return " ".join(parts)

    @property
    def case_count(self) -> int:
        return len(self.cases)
