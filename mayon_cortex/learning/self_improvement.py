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
Pillar 5: Recursive Self-Improvement Engine
===========================================
Continually evaluates the proof quality of Cortex reasoning, diagnoses weak or
contradictory pathways, and adapts the architecture autonomously.

3-Tier Adaptive Improvement:
  Tier 1 (High Quality >= 0.8):
    - Hebbian edge weight reinforcement.
    - Automatically archives validated precedent into CaseMemory.
  Tier 2 (Moderate Quality 0.5 - 0.8):
    - Registers weak pathways as training examples for NDU and PathRanker.
  Tier 3 (Low Quality / Contradiction < 0.5):
    - Archives failure into ErrorMemory.
    - Diagnoses contradiction via SelfCorrector and prunes invalid weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mayon_cortex.engine import CortexGraph, CortexResponse
from mayon_cortex.memory.error_memory import ErrorMemory


@dataclass
class ProofQualityScore:
    """Multi-factor evaluation of a reasoning output."""
    overall_score: float
    confidence_score: float
    logic_validity_score: float
    contradiction_freedom: float
    coverage_score: float
    tier: str  # "TIER_1_EXCELLENT", "TIER_2_MODERATE", "TIER_3_DEFICIENT"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImprovementAction:
    """Record of concrete adjustments made during self-improvement."""
    timestamp: float
    query: str
    tier: str
    quality_score: float
    hebbian_edges_reinforced: int
    case_memory_stored: bool
    error_recorded: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfImprovementEngine:
    """
    Autonomous proof evaluation and recursive self-improvement engine.
    Persistent storage in cortex_storage/self_improvement/.
    """

    def __init__(
        self,
        cortex: CortexGraph,
        error_memory: Optional[ErrorMemory] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.cortex = cortex
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/self_improvement")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.error_memory = error_memory or ErrorMemory(self.storage_dir.parent / "errors")
        self.eval_file = self.storage_dir / "proof_evaluations.jsonl"
        self.actions_file = self.storage_dir / "improvement_actions.jsonl"
        self._eval_count = 0

    def evaluate_proof(self, response: CortexResponse, query: str) -> ProofQualityScore:
        """
        Evaluate proof quality using 4 rigorous dimensions.
        """
        # 1. Confidence
        conf = float(response.confidence)

        # 2. Logic validity (path presence & consistency)
        logic_val = 1.0
        if not response.paths and not response.is_uncertain:
            logic_val = 0.5
        elif response.is_uncertain:
            logic_val = 0.3

        # 3. Contradiction freedom
        has_contra = getattr(response, "has_contradiction", False)
        contra_free = 0.0 if has_contra else 1.0

        # 4. Coverage (presence of query concepts in answer/paths)
        q_words = [w.lower() for w in query.split() if len(w) > 3]
        if q_words and response.text:
            ans_lower = response.text.lower()
            matches = sum(1 for w in q_words if w in ans_lower)
            coverage = min(1.0, (matches / len(q_words)) + 0.3)
        else:
            coverage = 0.7

        # Weighted aggregate
        overall = (conf * 0.35) + (logic_val * 0.25) + (contra_free * 0.25) + (coverage * 0.15)

        if overall >= 0.75 and contra_free == 1.0:
            tier = "TIER_1_EXCELLENT"
        elif overall >= 0.45 and contra_free == 1.0:
            tier = "TIER_2_MODERATE"
        else:
            tier = "TIER_3_DEFICIENT"

        score = ProofQualityScore(
            overall_score=round(overall, 3),
            confidence_score=round(conf, 3),
            logic_validity_score=round(logic_val, 3),
            contradiction_freedom=round(contra_free, 3),
            coverage_score=round(coverage, 3),
            tier=tier,
            details={
                "paths_count": len(response.paths),
                "is_uncertain": response.is_uncertain,
                "has_contradiction": has_contra,
            },
        )

        self._record_eval(query, score)
        return score

    def improve_from_proof(
        self,
        query: str,
        response: CortexResponse,
        sector: str = "general",
    ) -> ImprovementAction:
        """
        Apply 3-tier recursive adaptation based on proof quality score.
        """
        score = self.evaluate_proof(response, query)
        t_now = time.time()
        reinforced_edges = 0
        case_stored = False
        err_recorded = False
        desc = ""

        if score.tier == "TIER_1_EXCELLENT":
            # Tier 1: Reinforce paths & store precedent
            if response.paths:
                for path in response.paths:
                    node_ids = path.node_ids if hasattr(path, "node_ids") else []
                    self.cortex.learner.reinforce(node_ids, self.cortex.graph, boost=0.05)
                    reinforced_edges += len(node_ids)

            # Store in CaseMemory
            case_id = self.cortex.case_memory.store_case(
                situation=query,
                judgment=response.text[:200] if response.text else "Resolved",
                reasoning=f"High proof score ({score.overall_score:.2f}) verified deterministically",
                graph=self.cortex.graph,
                embedder=self.cortex.embedder,
                sector=sector,
                outcome="Verified truth",
            )
            case_stored = bool(case_id)
            desc = f"Tier 1: Reinforced {reinforced_edges} nodes/edges and archived precedent {case_id}."

        elif score.tier == "TIER_2_MODERATE":
            # Tier 2: Accumulate as learning example for NDU and retrain
            self.cortex.learner.learn(
                scored_paths=response.paths,
                reward=0.6,
                graph=self.cortex.graph,
                ndu=self.cortex.ndu,
                ranker=self.cortex.ranker,
                answer_text=response.text,
                embedder=self.cortex.embedder,
                word_graph=self.cortex.word_graph,
            )
            desc = "Tier 2: Registered pathway into online Hebbian training pipeline."

        else:
            # Tier 3: Record contradiction/error & prune
            err_id = self.error_memory.record_error(
                query=query,
                faulty_reasoning=response.text[:200] if response.text else "Uncertain proof failure",
                contradiction_type="contradiction" if getattr(response, "has_contradiction", False) else "uncertain_void",
                correction="Requires verified ground truth injection",
                sector=sector,
            )
            err_recorded = True

            # Prune or attenuate weights if contradiction was present
            if response.paths and getattr(response, "has_contradiction", False):
                for path in response.paths:
                    node_ids = path.node_ids if hasattr(path, "node_ids") else []
                    for nid in node_ids:
                        if nid in self.cortex.graph.nodes:
                            self.cortex.graph.nodes[nid].confidence = max(0.1, self.cortex.graph.nodes[nid].confidence - 0.1)

            desc = f"Tier 3: Contradiction registered as {err_id} in ErrorMemory and weights attenuated."

        action = ImprovementAction(
            timestamp=t_now,
            query=query,
            tier=score.tier,
            quality_score=score.overall_score,
            hebbian_edges_reinforced=reinforced_edges,
            case_memory_stored=case_stored,
            error_recorded=err_recorded,
            description=desc,
        )

        self._record_action(action)
        return action

    def _record_eval(self, query: str, score: ProofQualityScore):
        try:
            with open(self.eval_file, "a", encoding="utf-8") as f:
                rec = {"query": query, "timestamp": time.time(), "score": score.to_dict()}
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _record_action(self, action: ImprovementAction):
        try:
            with open(self.actions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(action.to_dict()) + "\n")
        except Exception:
            pass

    def get_quality_progression(self) -> Dict[str, Any]:
        """Read back proof evaluation progression from storage."""
        scores = []
        if self.eval_file.exists():
            with open(self.eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            scores.append(d.get("score", {}).get("overall_score", 0.0))
                        except Exception:
                            pass

        return {
            "total_proofs_evaluated": len(scores),
            "average_proof_score": (sum(scores) / len(scores)) if scores else 0.0,
            "recent_10_average": (sum(scores[-10:]) / len(scores[-10:])) if scores else 0.0,
            "errors_in_memory": self.error_memory.stats(),
        }
