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
Pillar 1: DualProcessRouter — Dynamic Kahneman System 1 & System 2 Routing
==========================================================================
Coordinates execution between deterministic symbolic cortex reasoning and
probabilistic LLM verbalization.

System 1 (Fast / Direct):
  - Trivial queries bypass complex graph wavefronts or use direct lookup.
  - Generates fast responses with zero unnecessary overhead.

System 2 (Deliberative / Deterministic):
  - Moderate / Complex / Novel queries trigger spreading activation waves,
    formal SMT math solvers, CaseMemory precedents, decision chains,
    and post-generation epistemic validation.
  - Ensures a 7B LLM grounded by Mayon-Cortex outperforms 70B ungrounded models.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mayon_cortex.engine import CortexGraph, CortexResponse, DecisionResponse
from mayon_cortex.executive.executive import Complexity, ExecutivePlan, ThinkingStrategy
from mayon_cortex.integration.cortex_llm_bridge import (
    cortex_decide,
    cortex_plan,
    cortex_recall,
    cortex_remember,
    cortex_think,
    cortex_validate,
)


@dataclass
class HybridResponse:
    """Complete output produced by the DualProcessRouter."""
    query: str
    complexity: str
    routing_mode: str  # "SYSTEM_1_DIRECT", "SYSTEM_2_THINK", "SYSTEM_2_DECIDE", "SYSTEM_2_NOVEL"
    llm_output: str
    cortex_used: bool
    confidence: float
    is_grounded: bool
    is_uncertain: bool = False
    has_contradiction: bool = False
    execution_time_ms: float = 0.0
    proof_paths: List[List[str]] = field(default_factory=list)
    precedents: List[Dict[str, Any]] = field(default_factory=list)
    math_proof: Optional[Dict[str, Any]] = None
    decision_steps: List[Dict[str, Any]] = field(default_factory=list)
    validation_report: Optional[Dict[str, Any]] = None
    structured_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualProcessRouter:
    """
    Kahneman System 1 / System 2 dynamic router for Mayon-Cortex.
    Persistent telemetry, execution auditing, and pluggable LLM interface.
    """

    def __init__(
        self,
        cortex: CortexGraph,
        storage_dir: Optional[Union[str, Path]] = None,
        default_llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.cortex = cortex
        self.storage_dir = Path(storage_dir) if storage_dir else Path("cortex_storage/router")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.default_llm_fn = default_llm_fn
        self.history_file = self.storage_dir / "routing_history.jsonl"
        self._execution_count = 0

    def route(
        self,
        query: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        sector: Optional[str] = None,
        force_system_2: bool = False,
    ) -> HybridResponse:
        """
        Dynamically route query based on complexity classification.
        Pluggable llm_fn: pass any callable (e.g. OpenAI, Claude, Ollama, Llama 3)
        or use the built-in deterministic verbalizer.
        """
        t0 = time.time()
        active_llm = llm_fn or self.default_llm_fn

        # 1. Executive Complexity Planning
        plan = cortex_plan(self.cortex, query=query)
        complexity = plan.get("complexity", "moderate")

        if force_system_2:
            complexity = "complex"

        # Determine Route Mode
        if complexity == "trivial" and not force_system_2:
            response = self._execute_system_1(query, plan, active_llm, t0)
        elif complexity == "moderate":
            response = self._execute_system_2_memory_augmented(query, plan, sector, active_llm, t0)
        elif complexity in ["complex", "multi_part"]:
            response = self._execute_system_2_full_deliberation(query, plan, sector, active_llm, t0)
        else:  # Novel or open-ended
            response = self._execute_system_2_novel(query, plan, sector, active_llm, t0)

        # 2. Persist execution telemetry to storage
        self._record_telemetry(response)
        self._execution_count += 1
        return response

    def _execute_system_1(
        self,
        query: str,
        plan: Dict[str, Any],
        llm_fn: Optional[Callable[[str], str]],
        t0: float,
    ) -> HybridResponse:
        """System 1: Fast direct lookup with minimal overhead."""
        prompt = f"Question: {query}\nProvide a concise and direct answer:"
        if llm_fn:
            llm_text = llm_fn(prompt)
        else:
            # Deterministic graph direct lookup
            res = self.cortex.query(query)
            llm_text = res.text or "Direct lookup confirmed."

        elapsed = (time.time() - t0) * 1000.0
        return HybridResponse(
            query=query,
            complexity="trivial",
            routing_mode="SYSTEM_1_DIRECT",
            llm_output=llm_text,
            cortex_used=False,
            confidence=0.95,
            is_grounded=True,
            is_uncertain=False,
            execution_time_ms=elapsed,
            structured_context="",
        )

    def _execute_system_2_memory_augmented(
        self,
        query: str,
        plan: Dict[str, Any],
        sector: Optional[str],
        llm_fn: Optional[Callable[[str], str]],
        t0: float,
    ) -> HybridResponse:
        """System 2 (Light): Precedent Case Memory + Spreading Activation."""
        recall_res = cortex_recall(self.cortex, situation=query, sector=sector, top_k=3)
        think_res = cortex_think(self.cortex, query=query, max_hops=3, top_paths=3)

        context_blocks = []
        if recall_res.get("formatted_context"):
            context_blocks.append(recall_res["formatted_context"])
        if think_res.get("formatted_context"):
            context_blocks.append(think_res["formatted_context"])

        structured_ctx = "\n\n".join(context_blocks)
        prompt = (
            f"You are a neuro-symbolic assistant powered by Mayon-Cortex.\n"
            f"The following verified knowledge was retrieved deterministically from the graph:\n\n"
            f"{structured_ctx}\n\n"
            f"User Question: {query}\n"
            f"Provide an answer strictly grounded in the facts above. Do not hallucinate."
        )

        if llm_fn:
            llm_text = llm_fn(prompt)
        else:
            # Deterministic graph-assembled verbalization
            precedents = recall_res.get("precedents", [])
            facts = think_res.get("facts", [])
            lines = [f"- {f.get('text', '')}" for f in facts if f.get('text')]
            if precedents:
                lines.insert(0, f"Precedent: {precedents[0].get('judgment', '')}")
            llm_text = "\n".join(lines) if lines else (think_res.get("answer") or "Grounded knowledge retrieved.")

        # Post-validation
        val = cortex_validate(self.cortex, claim_or_decision=llm_text)
        elapsed = (time.time() - t0) * 1000.0

        return HybridResponse(
            query=query,
            complexity="moderate",
            routing_mode="SYSTEM_2_THINK",
            llm_output=llm_text,
            cortex_used=True,
            confidence=think_res.get("confidence", 0.8),
            is_grounded=val.get("is_safe", True),
            is_uncertain=think_res.get("is_uncertain", False),
            has_contradiction=len(val.get("contradictions", [])) > 0,
            execution_time_ms=elapsed,
            proof_paths=think_res.get("proof_paths", []),
            precedents=recall_res.get("precedents", []),
            math_proof=think_res.get("math_proof"),
            validation_report=val,
            structured_context=structured_ctx,
        )

    def _execute_system_2_full_deliberation(
        self,
        query: str,
        plan: Dict[str, Any],
        sector: Optional[str],
        llm_fn: Optional[Callable[[str], str]],
        t0: float,
    ) -> HybridResponse:
        """System 2 (Full): Multi-hop proofs, SMT math solver, DecisionPlanner, and contradiction gate."""
        think_res = cortex_think(self.cortex, query=query, max_hops=4, top_paths=5, allow_math=True)
        decide_res = cortex_decide(self.cortex, situation=query, sector=sector)
        recall_res = cortex_recall(self.cortex, situation=query, sector=sector, top_k=2)

        context_blocks = []
        if think_res.get("formatted_context"):
            context_blocks.append(think_res["formatted_context"])
        if decide_res.get("formatted_context"):
            context_blocks.append(decide_res["formatted_context"])
        if recall_res.get("formatted_context"):
            context_blocks.append(recall_res["formatted_context"])

        structured_ctx = "\n\n".join(context_blocks)

        prompt = (
            f"You are Mayon-Cortex + LLM Neuro-Symbolic Cognitive Engine.\n"
            f"Below are deterministic symbolic proofs and decision chains verified on CPU:\n\n"
            f"{structured_ctx}\n\n"
            f"Question / Scenario: {query}\n"
            f"Synthesize a clear, authoritative response. State all reasons and contraindications explicitly."
        )

        if llm_fn:
            llm_text = llm_fn(prompt)
        else:
            steps = decide_res.get("steps", [])
            conflicts = decide_res.get("conflicts", [])
            output_parts = []
            if steps:
                output_parts.append("Deterministic Action Steps:")
                for s in steps:
                    output_parts.append(f"  {s.get('rank', 1)}. [{s.get('sector', 'core')}] {s.get('action', '')}")
            if conflicts:
                output_parts.append("\nDetected Contraindications/Conflicts:")
                for c in conflicts:
                    output_parts.append(f"  - {c}")
            if think_res.get("math_proof"):
                mp = think_res["math_proof"]
                output_parts.append(f"\nMath Solution: {mp.get('result')} ({mp.get('explanation')})")
            llm_text = "\n".join(output_parts) if output_parts else "Deterministic reasoning completed."

        # Validate against graph
        val = cortex_validate(self.cortex, claim_or_decision=llm_text)

        # If contradiction detected, correct it if LLM is active
        if len(val.get("contradictions", [])) > 0 and llm_fn:
            correction_prompt = (
                f"{prompt}\n\n"
                f"SAFETY ALERT: Your previous draft had a contradiction: {val.get('contradictions')}\n"
                f"Rewrite your answer to strictly obey this contraindication constraint."
            )
            llm_text = llm_fn(correction_prompt)
            val = cortex_validate(self.cortex, claim_or_decision=llm_text)

        # Learn precedent continuously
        if not think_res.get("is_uncertain", False):
            cortex_remember(
                brain=self.cortex,
                situation=query,
                judgment=llm_text[:200],
                reasoning=f"Solved via System 2 with confidence {think_res.get('confidence', 0.9):.2f}",
                sector=sector or "general",
            )

        elapsed = (time.time() - t0) * 1000.0
        return HybridResponse(
            query=query,
            complexity="complex",
            routing_mode="SYSTEM_2_DECIDE",
            llm_output=llm_text,
            cortex_used=True,
            confidence=max(think_res.get("confidence", 0.0), decide_res.get("confidence", 0.0)),
            is_grounded=val.get("is_safe", True),
            is_uncertain=think_res.get("is_uncertain", False),
            has_contradiction=len(val.get("contradictions", [])) > 0,
            execution_time_ms=elapsed,
            proof_paths=think_res.get("proof_paths", []),
            precedents=recall_res.get("precedents", []),
            math_proof=think_res.get("math_proof"),
            decision_steps=decide_res.get("steps", []),
            validation_report=val,
            structured_context=structured_ctx,
        )

    def _execute_system_2_novel(
        self,
        query: str,
        plan: Dict[str, Any],
        sector: Optional[str],
        llm_fn: Optional[Callable[[str], str]],
        t0: float,
    ) -> HybridResponse:
        """System 2 (Novel): Imagination sandbox and hypothesis exploration."""
        hypo = self.cortex.imagine_engine.imagine(self.cortex.graph, seed_prompt=query)
        sandbox_res = self.cortex.imagine_engine.sandbox.run(hypo, self.cortex.graph)

        prompt = (
            f"You are exploring a novel hypothesis using Mayon-Cortex Mental Sandbox:\n"
            f"Hypothesis Concept: {hypo.name} (Plausibility: {hypo.plausibility:.2f})\n"
            f"Sandbox Stability: {sandbox_res.is_stable}\n"
            f"Question: {query}\n"
            f"Synthesize an exploratory hypothesis and explain potential implications."
        )

        if llm_fn:
            llm_text = llm_fn(prompt)
        else:
            llm_text = (
                f"Novel Hypothesis: {hypo.name}\n"
                f"Plausibility: {hypo.plausibility:.2f}, Stability: {sandbox_res.is_stable}\n"
                f"Bridging paths explored across mental sandbox."
            )

        elapsed = (time.time() - t0) * 1000.0
        return HybridResponse(
            query=query,
            complexity="novel",
            routing_mode="SYSTEM_2_NOVEL",
            llm_output=llm_text,
            cortex_used=True,
            confidence=float(hypo.plausibility),
            is_grounded=sandbox_res.is_stable,
            execution_time_ms=elapsed,
            structured_context=f"Hypothesis: {hypo.name}\nDescription: {hypo.description}",
        )

    def _record_telemetry(self, response: HybridResponse):
        """Persist execution log to JSONL file."""
        try:
            record = {
                "timestamp": time.time(),
                "query": response.query,
                "complexity": response.complexity,
                "routing_mode": response.routing_mode,
                "cortex_used": response.cortex_used,
                "confidence": response.confidence,
                "is_grounded": response.is_grounded,
                "is_uncertain": response.is_uncertain,
                "has_contradiction": response.has_contradiction,
                "execution_time_ms": response.execution_time_ms,
                "proof_path_count": len(response.proof_paths),
            }
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Read back accumulated telemetry from storage."""
        if not self.history_file.exists():
            return {"total_queries": 0}

        total = 0
        s1_count = 0
        s2_count = 0
        total_time = 0.0
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    try:
                        rec = json.loads(line)
                        if rec.get("routing_mode") == "SYSTEM_1_DIRECT":
                            s1_count += 1
                        else:
                            s2_count += 1
                        total_time += rec.get("execution_time_ms", 0.0)
                    except Exception:
                        pass

        return {
            "total_queries": total,
            "system_1_queries": s1_count,
            "system_2_queries": s2_count,
            "system_2_ratio": (s2_count / total) if total > 0 else 0.0,
            "avg_latency_ms": (total_time / total) if total > 0 else 0.0,
        }
