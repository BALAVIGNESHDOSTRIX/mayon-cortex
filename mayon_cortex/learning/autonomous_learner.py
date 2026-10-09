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
Pillar 2: Autonomous Curiosity-Driven Self-Expansion
===================================================
Enables Mayon-Cortex to autonomously identify its own epistemic voids,
formulate targeted inquiry questions, acquire resolving facts from connected
data feeds or LLMs, integrate the knowledge via Hebbian plasticity, and verify
that its reasoning confidence increased.

Features:
- Epistemic gap detection across graph topology and uncertain queries.
- Persistent storage of discovered gaps, resolution cycles, and growth metrics.
- Self-expansion loop: `self_expand(cycles=5)`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mayon_cortex.engine import CortexGraph, CortexResponse


@dataclass
class KnowledgeGap:
    """Represents a discovered epistemic void in the graph."""
    gap_id: str
    concept: str
    target_concept: Optional[str] = None
    missing_relation: str = "relates_to"
    sector: str = "general"
    gap_score: float = 0.5  # Higher means greater epistemic uncertainty
    inquiry_question: str = ""
    discovered_from: str = "graph_topology"  # "query_failure" or "graph_topology"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpansionResult:
    """Outcome of an autonomous expansion cycle."""
    cycle_index: int
    gaps_detected: int
    gaps_resolved: int
    new_nodes_added: int
    new_edges_added: int
    initial_confidence: float
    final_confidence: float
    confidence_gain: float
    elapsed_ms: float
    resolved_gaps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutonomousLearner:
    """
    Autonomous knowledge expansion engine for Mayon-Cortex.
    Persistent storage in cortex_storage/curiosity/.
    """

    def __init__(
        self,
        cortex: CortexGraph,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.cortex = cortex
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/curiosity")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.gaps_file = self.storage_dir / "discovered_gaps.jsonl"
        self.cycles_file = self.storage_dir / "expansion_cycles.jsonl"
        self._cycle_counter = 0

    def detect_knowledge_gaps(
        self,
        failed_query: Optional[str] = None,
        max_gaps: int = 5,
        sector: Optional[str] = None,
    ) -> List[KnowledgeGap]:
        """
        Detect gaps via query failure or topological analysis of the graph.
        """
        gaps: List[KnowledgeGap] = []

        # 1. Gaps from query failure / uncertainty
        if failed_query:
            resp = self.cortex.query(failed_query)
            if resp.is_uncertain or resp.confidence < 0.6:
                # Extract subject / target concepts from query words
                words = [w for w in failed_query.split() if len(w) > 3]
                primary = words[0] if words else failed_query
                target = words[-1] if len(words) > 1 else None
                gap = KnowledgeGap(
                    gap_id=f"gap_query_{int(time.time()*1000)}",
                    concept=primary,
                    target_concept=target,
                    missing_relation="causes_or_treats",
                    sector=sector or "general",
                    gap_score=1.0 - resp.confidence,
                    inquiry_question=f"What is the exact relation between {primary} and {target or 'surrounding mechanisms'}?",
                    discovered_from="query_failure",
                )
                gaps.append(gap)

        # 2. Gaps from topological sparsity (isolated or low-degree nodes)
        nodes = list(self.cortex.graph.nodes.values())
        if nodes:
            # Sort by degree (fewest connections = highest uncertainty)
            def _get_node_degree(nid: str) -> int:
                out_e = len(self.cortex.graph._out_edges.get(nid, []))
                in_e = len(self.cortex.graph._in_edges.get(nid, []))
                return out_e + in_e

            low_degree_nodes = sorted(
                nodes,
                key=lambda n: _get_node_degree(n.id),
            )[:max_gaps]

            for idx, node in enumerate(low_degree_nodes):
                degree = _get_node_degree(node.id)
                if degree < 2:
                    concept_label = node.source_text[:40] if node.source_text else f"Concept_{node.id}"
                    gap = KnowledgeGap(
                        gap_id=f"gap_topo_{node.id}_{int(time.time())}_{idx}",
                        concept=concept_label,
                        sector=node.sector or sector or "general",
                        gap_score=max(0.1, 0.8 - (degree * 0.2)),
                        inquiry_question=f"What are the primary mechanisms, causes, and consequences of {concept_label}?",
                        discovered_from="graph_topology",
                    )
                    gaps.append(gap)

        # Persist detected gaps to disk
        self._record_gaps(gaps)
        return gaps[:max_gaps]

    def resolve_gap(
        self,
        gap: KnowledgeGap,
        resolution_facts: Optional[List[str]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> bool:
        """
        Resolve a specific knowledge gap using provided facts or an LLM inquiry.
        """
        facts_to_ingest = resolution_facts or []

        # If an LLM or oracle is provided, query it to acquire resolving facts
        if not facts_to_ingest and llm_fn:
            inquiry_prompt = (
                f"Knowledge Expansion Request:\n"
                f"Question: {gap.inquiry_question}\n"
                f"Provide 2-3 precise factual statements describing the mechanism, relationship, or rule."
            )
            answer = llm_fn(inquiry_prompt)
            if answer:
                # Split by sentence or line
                facts_to_ingest = [line.strip("- ") for line in answer.split("\n") if len(line.strip()) > 10]

        if not facts_to_ingest:
            return False

        # Ingest new facts into MayonGraph
        nodes_before = self.cortex.graph.num_nodes
        edges_before = self.cortex.graph.num_edges

        for fact in facts_to_ingest:
            self.cortex.ingest(fact, sector=gap.sector)

        # Trigger Hebbian reinforcement on graph
        self.cortex.learner.learn(
            scored_paths=[],
            reward=0.8,
            graph=self.cortex.graph,
            ndu=self.cortex.ndu,
            ranker=self.cortex.ranker,
            answer_text=" ".join(facts_to_ingest),
            embedder=self.cortex.embedder,
            word_graph=self.cortex.word_graph,
        )

        nodes_added = self.cortex.graph.num_nodes - nodes_before
        edges_added = self.cortex.graph.num_edges - edges_before
        return (nodes_added + edges_added) > 0

    def run_curiosity_cycle(
        self,
        trigger_query: Optional[str] = None,
        external_facts: Optional[Dict[str, List[str]]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        sector: Optional[str] = None,
    ) -> ExpansionResult:
        """
        Execute one complete curiosity cycle:
        Detect gaps -> Inquire -> Ingest -> Verify confidence gain.
        """
        t0 = time.time()
        self._cycle_counter += 1

        # Baseline confidence
        init_conf = 0.0
        if trigger_query:
            init_conf = self.cortex.query(trigger_query).confidence

        # 1. Detect
        gaps = self.detect_knowledge_gaps(failed_query=trigger_query, max_gaps=3, sector=sector)
        resolved_count = 0
        resolved_details = []

        n_before = self.cortex.graph.num_nodes
        e_before = self.cortex.graph.num_edges

        # 2. Resolve each gap
        for gap in gaps:
            facts = None
            if external_facts and gap.concept in external_facts:
                facts = external_facts[gap.concept]
            elif external_facts and "default" in external_facts:
                facts = external_facts["default"]

            success = self.resolve_gap(gap, resolution_facts=facts, llm_fn=llm_fn)
            if success:
                resolved_count += 1
                resolved_details.append(gap.to_dict())

        # 3. Post-cycle verification
        final_conf = init_conf
        if trigger_query:
            final_conf = self.cortex.query(trigger_query).confidence

        elapsed = (time.time() - t0) * 1000.0
        result = ExpansionResult(
            cycle_index=self._cycle_counter,
            gaps_detected=len(gaps),
            gaps_resolved=resolved_count,
            new_nodes_added=self.cortex.graph.num_nodes - n_before,
            new_edges_added=self.cortex.graph.num_edges - e_before,
            initial_confidence=init_conf,
            final_confidence=final_conf,
            confidence_gain=max(0.0, final_conf - init_conf),
            elapsed_ms=elapsed,
            resolved_gaps=resolved_details,
        )

        self._record_cycle(result)
        return result

    def self_expand(
        self,
        cycles: int = 3,
        knowledge_feed: Optional[List[str]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        sector: Optional[str] = None,
    ) -> List[ExpansionResult]:
        """
        Run multiple consecutive autonomous self-expansion cycles.
        """
        feed = knowledge_feed or []
        results = []
        for i in range(cycles):
            # Pick batch of feed if available
            facts_map = {}
            if feed:
                batch = feed[i * 2 : (i + 1) * 2]
                facts_map["default"] = batch

            res = self.run_curiosity_cycle(
                external_facts=facts_map if facts_map else None,
                llm_fn=llm_fn,
                sector=sector,
            )
            results.append(res)
        return results

    def _record_gaps(self, gaps: List[KnowledgeGap]):
        """Append gaps to persistent disk store."""
        try:
            with open(self.gaps_file, "a", encoding="utf-8") as f:
                for gap in gaps:
                    f.write(json.dumps(gap.to_dict()) + "\n")
        except Exception:
            pass

    def _record_cycle(self, result: ExpansionResult):
        """Append cycle result to persistent disk store."""
        try:
            with open(self.cycles_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception:
            pass

    def get_summary(self) -> Dict[str, Any]:
        """Retrieve historical curiosity expansion summary."""
        total_cycles = 0
        total_resolved = 0
        total_nodes = 0
        total_edges = 0

        if self.cycles_file.exists():
            with open(self.cycles_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            total_cycles += 1
                            total_resolved += d.get("gaps_resolved", 0)
                            total_nodes += d.get("new_nodes_added", 0)
                            total_edges += d.get("new_edges_added", 0)
                        except Exception:
                            pass

        return {
            "total_cycles_run": total_cycles,
            "total_gaps_resolved": total_resolved,
            "total_nodes_added": total_nodes,
            "total_edges_added": total_edges,
            "current_graph_size": self.cortex.graph.num_nodes,
        }
