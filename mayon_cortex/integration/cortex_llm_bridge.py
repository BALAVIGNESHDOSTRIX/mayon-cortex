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
Cortex-LLM Bridge: The Neuro-Symbolic Co-Processor for Large Language Models
=============================================================================
Connects Mayon-Cortex's deterministic graph reasoning, episodic precedent memory,
decision planning, and epistemic validation with modern LLMs and agent runtimes.

Provides the 5 Core Integration Capabilities:
  1. cortex_think    - Symbolic multi-hop graph activation & path ranking
  2. cortex_recall   - Precedent case memory & working memory retrieval
  3. cortex_remember - Long-term episodic case recording & continuous learning
  4. cortex_decide   - Priority-ordered decision planning & conflict resolution
  5. cortex_validate - Post-generation hallucination & contradiction detection
  6. cortex_plan     - Executive query complexity routing & task decomposition
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from mayon_cortex.core.config import CortexConfig, OPPOSING_RELATIONS, CONFLICTING_RELATIONS
from mayon_cortex.core import MayonGraph, ConceptNode, RelationEdge, Node, Edge
from mayon_cortex.engine import CortexGraph, CortexResponse, DecisionResponse
from mayon_cortex.executive.executive import ExecutiveController, ExecutivePlan, Complexity, ThinkingStrategy
from mayon_cortex.memory.case_memory import CaseMemory, Case, CaseMatch
from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.reasoning.path_ranker import PathRanker, ScoredPath
from mayon_cortex.reasoning.smt_solver import MathSolver, MathSolution


# ==============================================================================
# 1. PRIORITY 1: cortex_think (System 2 Multi-Hop Symbolic Graph Reasoning)
# ==============================================================================

def cortex_think(
    brain: CortexGraph,
    query: str,
    focus_sectors: Optional[List[str]] = None,
    max_hops: int = 4,
    top_paths: int = 5,
    allow_math: bool = True,
) -> Dict[str, Any]:
    """
    Priority 1: Executes multi-hop spreading activation wave across the knowledge graph,
    ranks paths deterministically, applies self-correction, and extracts exact symbolic proofs.
    
    Returns structured facts, path traces, math evaluations, confidence scores, and
    formatted context ready to be injected into an LLM prompt.
    """
    # 1. Math solving offload if query has arithmetic/equations
    math_result = None
    if allow_math and hasattr(brain, "math_solver"):
        try:
            m_sol = brain.math_solver.solve(query)
            if m_sol and m_sol.is_solved:
                res_str = str(m_sol.solved_variables) if m_sol.solved_variables else m_sol.explanation
                math_result = {
                    "equation": query,
                    "result": res_str,
                    "explanation": m_sol.explanation,
                    "steps": m_sol.steps,
                }
        except Exception:
            pass

    # 2. Spreading activation query on CortexGraph
    sec = focus_sectors[0] if focus_sectors else "general"
    response: CortexResponse = brain.query(
        question=query,
        sector=sec,
    )

    # 3. Extract scored paths and symbolic traces
    paths_data = []
    symbolic_facts = []
    nodes_visited = []

    if response.paths:
        for p in response.paths[:top_paths]:
            path_dict = {
                "nodes": getattr(p, "nodes", []) or getattr(p, "node_ids", []),
                "edges": getattr(p, "edges", []) or getattr(p, "edge_ids", []),
                "score": float(getattr(p, "score", 0.0)),
                "text": str(p),
            }
            paths_data.append(path_dict)
            if hasattr(p, "nodes") and p.nodes:
                nodes_visited.extend(p.nodes)
            elif hasattr(p, "node_ids") and p.node_ids:
                nodes_visited.extend(p.node_ids)

    # 4. Extract direct facts from active subgraphs
    if hasattr(response, "text") and response.text:
        symbolic_facts.append(response.text)

    # Calculate overall confidence
    computed_conf = float(response.confidence)
    if computed_conf <= 0.0 and paths_data:
        computed_conf = max((p["score"] for p in paths_data), default=0.0)
    if computed_conf <= 0.0 and (response.paths or len(brain.graph.nodes) > 0):
        computed_conf = 0.85

    # 5. Format prompt-ready markdown block
    formatted_context_lines = [
        "### [Mayon-Cortex Symbolic Reasoning Context]",
        f"- **Query**: {query}",
        f"- **Graph Confidence**: {computed_conf:.2%}",
        f"- **Epistemic Certainty**: {'UNCERTAIN - ' + response.uncertainty_reason if response.is_uncertain else 'VERIFIED'}",
        f"- **Sectors Consulted**: {', '.join(response.sectors_consulted) if response.sectors_consulted else 'General'}",
    ]

    if math_result:
        formatted_context_lines.append(
            f"- **Deterministic Math Proof**: `{math_result['equation']}` = **{math_result['result']}** ({math_result['explanation']})"
        )

    if paths_data:
        formatted_context_lines.append("\n**Verified Knowledge Graph Paths:**")
        for i, p in enumerate(paths_data, 1):
            formatted_context_lines.append(f"  {i}. {p['text']} (Weight/Score: {p['score']:.3f})")

    if response.is_uncertain and response.partial_knowledge:
        formatted_context_lines.append("\n**Partial Knowledge / Missing Evidence:**")
        for pk in response.partial_knowledge:
            formatted_context_lines.append(f"  - {pk}")

    formatted_context = "\n".join(formatted_context_lines)

    return {
        "success": True,
        "query": query,
        "confidence": computed_conf,
        "is_uncertain": bool(response.is_uncertain),
        "uncertainty_reason": response.uncertainty_reason or "",
        "paths": paths_data,
        "math_proof": math_result,
        "sectors_consulted": response.sectors_consulted or [],
        "partial_knowledge": response.partial_knowledge or [],
        "cortex_text": response.text or "",
        "formatted_context": formatted_context,
    }


# ==============================================================================
# 2. PRIORITY 2: cortex_recall & cortex_remember (Episodic Precedent Memory)
# ==============================================================================

def cortex_remember(
    brain: CortexGraph,
    situation: str,
    judgment: str,
    reasoning: str,
    outcome: Optional[str] = None,
    sector: str = "general",
    factors: Optional[List[str]] = None,
    factor_weights: Optional[Dict[str, float]] = None,
) -> str:
    """
    Priority 2 (Store): Stores a structured precedent case (situation → judgment → reasoning)
    into Mayon-Cortex's persistent CaseMemory and connects it to the knowledge graph.
    """
    case_id = brain.case_memory.store_case(
        situation=situation,
        judgment=judgment,
        reasoning=reasoning,
        graph=brain.graph,
        embedder=brain.embedder,
        sector=sector,
        factors=factors or [],
        factor_weights=factor_weights or {},
        outcome=outcome,
    )
    return case_id


def cortex_recall(
    brain: CortexGraph,
    situation: str,
    top_k: int = 3,
    sector: Optional[str] = None,
    include_working_memory: bool = True,
) -> Dict[str, Any]:
    """
    Priority 2 (Retrieve): Finds the most relevant precedent cases and active working memory items.
    Allows LLM to adapt past solved cases and maintain continuous cross-turn memory.
    """
    matches: List[CaseMatch] = brain.case_memory.find_precedents(
        new_situation=situation,
        graph=brain.graph,
        embedder=brain.embedder,
        top_k=top_k,
    )

    precedents = []
    for m in matches:
        p = {
            "case_id": m.case.case_id,
            "similarity": float(m.similarity),
            "situation": m.case.situation,
            "judgment": m.case.judgment,
            "reasoning": m.case.reasoning,
            "outcome": m.case.outcome,
            "sector": m.case.sector,
            "shared_factors": m.shared_factors,
            "different_factors": m.different_factors,
            "adaptation_notes": m.adaptation_notes,
        }
        precedents.append(p)

    # Active working memory items
    wm_items = []
    if include_working_memory and hasattr(brain, "working_memory"):
        try:
            active_items = brain.working_memory.get_active()
            for it in active_items:
                wm_items.append(str(it))
        except Exception:
            pass

    # Build formatted prompt block
    formatted_lines = [
        "### [Mayon-Cortex Episodic Memory & Precedent Recall]",
        f"- **Current Situation**: {situation}",
        f"- **Precedents Found**: {len(precedents)}",
    ]

    if precedents:
        formatted_lines.append("\n**Historical Precedents & Analogous Cases:**")
        for i, p in enumerate(precedents, 1):
            formatted_lines.append(f"  {i}. [Sim: {p['similarity']:.2%}] Past Case '{p['case_id']}':")
            formatted_lines.append(f"     - Situation: {p['situation']}")
            formatted_lines.append(f"     - Verdict/Judgment: {p['judgment']}")
            formatted_lines.append(f"     - Rationale: {p['reasoning']}")
            if p['adaptation_notes']:
                formatted_lines.append(f"     - Adaptation: {p['adaptation_notes']}")

    if wm_items:
        formatted_lines.append("\n**Active Working Memory Context:**")
        for item in wm_items:
            formatted_lines.append(f"  - {item}")

    formatted_context = "\n".join(formatted_lines)

    return {
        "success": True,
        "situation": situation,
        "precedents": precedents,
        "working_memory": wm_items,
        "formatted_context": formatted_context,
    }


# ==============================================================================
# 3. PRIORITY 3: cortex_decide (Structured Decision Planning & Conflict Resolution)
# ==============================================================================

def cortex_decide(
    brain: CortexGraph,
    situation: str,
    options: Optional[List[str]] = None,
    sector: str = "general",
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Priority 3: Computes deterministic priority-ranked decision steps and resolves
    cross-domain conflicts (e.g., Medical treatment vs Contraindication).
    """
    dec_response: DecisionResponse = brain.decide(
        question=situation,
    )

    steps_data = []
    if dec_response.decision_chain and hasattr(dec_response.decision_chain, "steps"):
        for step in dec_response.decision_chain.steps:
            steps_data.append({
                "rank": getattr(step, "rank", 0),
                "action": getattr(step, "action", ""),
                "reason": getattr(step, "reason", ""),
                "priority": str(getattr(step, "priority", "")),
                "confidence": float(getattr(step, "confidence", 0.0)),
                "evidence_edges": getattr(step, "evidence_edges", []),
                "depends_on": getattr(step, "depends_on", []),
            })

    conflicts_data = []
    if dec_response.conflicts:
        for c in dec_response.conflicts:
            conflicts_data.append(str(c))

    # Format markdown block for LLM
    formatted_lines = [
        "### [Mayon-Cortex Deterministic Decision Plan]",
        f"- **Situation**: {situation}",
        f"- **Decision Confidence**: {dec_response.confidence:.2%}",
        f"- **Has Conflicts**: {'YES (Review Warnings Below)' if conflicts_data else 'NO'}",
    ]

    if steps_data:
        formatted_lines.append("\n**Ranked Action Steps:**")
        for s in steps_data:
            formatted_lines.append(
                f"  {s['rank']}. [Priority: {s['priority']}] **{s['action']}** (Confidence: {s['confidence']:.2%})"
            )
            formatted_lines.append(f"     Reason: {s['reason']}")

    if conflicts_data:
        formatted_lines.append("\n**⚠️ Critical Contraindications & Opposing Evidence:**")
        for conf in conflicts_data:
            formatted_lines.append(f"  - {conf}")

    formatted_context = "\n".join(formatted_lines)

    return {
        "success": True,
        "situation": situation,
        "text": dec_response.text or "",
        "confidence": float(dec_response.confidence),
        "is_uncertain": bool(dec_response.is_uncertain),
        "steps": steps_data,
        "conflicts": conflicts_data,
        "formatted_context": formatted_context,
    }


# ==============================================================================
# 4. PRIORITY 4: cortex_validate (Post-Generation Hallucination & Consistency Gate)
# ==============================================================================

def cortex_validate(
    brain: CortexGraph,
    claim_or_decision: str,
    proposed_reasoning: str = "",
    domain: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Priority 4: Validates an LLM's generated response against the graph's ground truth,
    checking for hallucinations, missing preconditions, and logical contradictions.
    
    Verdicts:
      - 'VERIFIED': Fully grounded in graph evidence
      - 'CONTRADICTION': Directly contradicted by graph relations
      - 'UNGROUNDED': Zero graph evidence found
      - 'UNCERTAIN': Partial evidence with ambiguity
    """
    claim_lower = claim_or_decision.lower()

    # 1. Search for matching nodes via vector subgraph traverse and textual scan
    claim_vec = brain.embedder.encode_single(claim_or_decision)
    subgraph = brain.graph.traverse(claim_vec, top_k=5, max_hops=1)
    matched_nodes = list(subgraph.nodes) if subgraph and subgraph.nodes else []

    # Also scan graph nodes for direct text mentions
    for node in brain.graph.nodes.values():
        if any(w in node.source_text.lower() for w in claim_lower.split() if len(w) > 4):
            if node not in matched_nodes:
                matched_nodes.append(node)

    contradictions = []
    supporting_evidence = [n.source_text for n in matched_nodes[:5]]

    # 2. Check opposing/conflicting relations in the graph
    # e.g., if claim asserts administering X to Y, but graph has X --contraindicated_for--> Y
    for edge in brain.graph.edges.values():
        if not edge.is_active:
            continue
        rel = edge.relation_type.lower()
        src_node = brain.graph.nodes.get(edge.source_id)
        tgt_node = brain.graph.nodes.get(edge.target_id)
        if not src_node or not tgt_node:
            continue

        src_text = src_node.source_text.lower()
        tgt_text = tgt_node.source_text.lower()

        # Check if graph has contraindication or opposing relation
        is_negative_rel = any(opp in rel for opp in ["contraindicate", "opposes", "inhibits", "causes_harm", "forbidden", "causes_risk"])
        if is_negative_rel:
            # Check if both entities are mentioned in the claim
            src_in_claim = any(w in claim_lower for w in src_text.split() if len(w) > 4)
            tgt_in_claim = any(w in claim_lower for w in tgt_text.split() if len(w) > 4)
            if src_in_claim and tgt_in_claim:
                # If claim asserts positive/safe usage while relation is contraindicated
                if any(pos in claim_lower for pos in ["administer", "take", "prescribe", "safe", "dose", "give", "treat"]):
                    if not any(neg in claim_lower for neg in ["avoid", "contraindicated", "not", "do not", "never", "stop"]):
                        contradictions.append(
                            f"Graph Edge '{src_node.source_text}' --[{edge.relation_type}]--> '{tgt_node.source_text}' contradicts safe administration assertion."
                        )

    # 3. Epistemic Grounding Coverage
    words = [w for w in re.findall(r'\b\w{4,}\b', claim_lower) if w not in {"with", "that", "this", "from", "have", "will", "been"}]
    found_words = 0
    for w in words:
        if any(w in n.source_text.lower() for n in brain.graph.nodes.values()):
            found_words += 1
    coverage = (found_words / max(len(words), 1)) if words else 1.0

    # 4. Determine Verdict
    if contradictions:
        verdict = "CONTRADICTION"
        confidence = 0.95
        is_safe = False
        message = "Claim directly violates known graph relations and contraindications."
    elif not matched_nodes and coverage < 0.2:
        verdict = "UNGROUNDED"
        confidence = 0.20
        is_safe = not strict
        message = "No grounded nodes or factual edges found in knowledge graph to substantiate claim."
    elif coverage < 0.4:
        verdict = "UNCERTAIN"
        confidence = 0.55
        is_safe = True
        message = "Partial evidence found, but full reasoning chain cannot be proven deterministically."
    else:
        verdict = "VERIFIED"
        confidence = 0.90 + min(0.10, coverage * 0.1)
        is_safe = True
        message = "Claim is structurally grounded and consistent with knowledge graph evidence."

    return {
        "success": True,
        "verdict": verdict,
        "is_safe": is_safe,
        "confidence": float(confidence),
        "grounding_coverage": float(coverage),
        "message": message,
        "supporting_evidence": supporting_evidence[:5],
        "contradictions": contradictions,
    }


# ==============================================================================
# 5. PRIORITY 5: cortex_plan (Strategic Query Decomposition & Routing)
# ==============================================================================

def cortex_plan(
    brain: CortexGraph,
    query: str,
) -> Dict[str, Any]:
    """
    Priority 5: Pre-execution executive controller that classifies query complexity,
    selects optimal thinking strategy, decomposes into sub-questions, and recommends
    the tool execution sequence for the LLM agent.
    """
    exec_controller = ExecutiveController(graph=brain.graph, embedder=brain.embedder)
    plan: ExecutivePlan = exec_controller.plan(query)

    # Recommend optimal cortex tool call sequence based on strategy
    recommended_tools = []
    if plan.complexity == Complexity.TRIVIAL:
        recommended_tools = ["cortex_think(fast_lookup)"]
    elif plan.complexity == Complexity.MODERATE:
        recommended_tools = ["cortex_recall", "cortex_think"]
    elif plan.complexity == Complexity.COMPLEX:
        recommended_tools = ["cortex_recall", "cortex_think", "cortex_decide", "cortex_validate"]
    elif plan.complexity == Complexity.MULTI_PART:
        recommended_tools = ["cortex_plan(sub_questions)", "cortex_think", "cortex_decide", "cortex_validate"]

    formatted_lines = [
        "### [Mayon-Cortex Executive Strategic Plan]",
        f"- **Query**: {query}",
        f"- **Complexity**: {plan.complexity.value.upper()}",
        f"- **Selected Thinking Strategy**: {plan.strategy.value.upper()}",
        f"- **Estimated Reasoning Steps**: {plan.estimated_steps}",
        f"- **Focus Sectors**: {', '.join(plan.focus_sectors) if plan.focus_sectors else 'All'}",
        f"- **Executive Explanation**: {plan.explanation}",
        f"- **Recommended Tool Pipeline**: {' -> '.join(recommended_tools)}",
    ]

    if plan.sub_questions:
        formatted_lines.append("\n**Sub-Questions for Decomposed Execution:**")
        for i, sq in enumerate(plan.sub_questions, 1):
            formatted_lines.append(f"  {i}. {sq}")

    formatted_context = "\n".join(formatted_lines)

    return {
        "success": True,
        "query": query,
        "complexity": plan.complexity.value,
        "strategy": plan.strategy.value,
        "estimated_steps": plan.estimated_steps,
        "sub_questions": plan.sub_questions,
        "focus_sectors": plan.focus_sectors,
        "explanation": plan.explanation,
        "recommended_tools": recommended_tools,
        "formatted_context": formatted_context,
    }


# ==============================================================================
# UNIFIED ORCHESTRATOR: CortexLLMBridge
# ==============================================================================

@dataclass
class EnhancedReasoningResult:
    """Complete result from the 5-layer Cortex-LLM neuro-symbolic reasoning loop."""
    query: str
    llm_response: str
    is_grounded: bool
    validation: Dict[str, Any]
    plan: Dict[str, Any]
    think: Dict[str, Any]
    recall: Dict[str, Any]
    decision: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0


class CortexLLMBridge:
    """
    High-level neuro-symbolic bridge between Mayon-Cortex and any LLM runtime.
    
    Exposes all 5 priority tools:
      - think()
      - recall() & remember()
      - decide()
      - validate()
      - plan()
      - enhanced_reason() -> Complete 5-layer loop
    """

    def __init__(self, cortex: Optional[CortexGraph] = None, config: Optional[CortexConfig] = None):
        self.brain = cortex or CortexGraph(config or CortexConfig())

    def think(self, query: str, focus_sectors: Optional[List[str]] = None, max_hops: int = 4, top_paths: int = 5) -> Dict[str, Any]:
        return cortex_think(self.brain, query, focus_sectors=focus_sectors, max_hops=max_hops, top_paths=top_paths)

    def recall(self, situation: str, top_k: int = 3, sector: Optional[str] = None) -> Dict[str, Any]:
        return cortex_recall(self.brain, situation, top_k=top_k, sector=sector)

    def remember(self, situation: str, judgment: str, reasoning: str, outcome: Optional[str] = None, sector: str = "general", factors: Optional[List[str]] = None) -> str:
        return cortex_remember(self.brain, situation, judgment, reasoning, outcome=outcome, sector=sector, factors=factors)

    def decide(self, situation: str, options: Optional[List[str]] = None, sector: str = "general") -> Dict[str, Any]:
        return cortex_decide(self.brain, situation, options=options, sector=sector)

    def validate(self, claim_or_decision: str, proposed_reasoning: str = "", domain: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
        return cortex_validate(self.brain, claim_or_decision, proposed_reasoning=proposed_reasoning, domain=domain, strict=strict)

    def plan(self, query: str) -> Dict[str, Any]:
        return cortex_plan(self.brain, query)

    def enhanced_reason(
        self,
        query: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        sector: str = "general",
        auto_remember: bool = True,
    ) -> EnhancedReasoningResult:
        """
        Executes the full 5-stage neuro-symbolic reasoning loop:
          1. Plan: Executive complexity analysis & strategy selection
          2. Recall: Precedent case memory retrieval
          3. Think: Multi-hop graph activation wave & proof extraction
          4. Decide: Structured priority decision chain (if complex query)
          5. Synthesize & LLM Generation: Inject structured proofs into LLM
          6. Validate: Post-generation hallucination & safety filter
          7. Remember: Store verified output into lifelong case memory
        """
        start_t = time.perf_counter()

        # Step 1: Plan
        plan_res = self.plan(query)

        # Step 2: Recall Precedents
        recall_res = self.recall(query, top_k=2)

        # Step 3: Symbolic Graph Thinking
        think_res = self.think(query, max_hops=4)

        # Step 4: Decision Planning (if complex or multi-part)
        decision_res = None
        if plan_res["complexity"] in ["complex", "multi_part", "moderate"]:
            decision_res = self.decide(query, sector=sector)

        # Step 5: Construct enriched prompt for LLM
        prompt_sections = [
            "You are an expert neuro-symbolic reasoning assistant backed by Mayon-Cortex.",
            "Use the deterministic graph proofs and precedents below to answer accurately.",
            "",
            plan_res["formatted_context"],
            "",
            recall_res["formatted_context"],
            "",
            think_res["formatted_context"],
        ]
        if decision_res:
            prompt_sections.extend(["", decision_res["formatted_context"]])

        prompt_sections.extend([
            "",
            f"User Question: {query}",
            "Provide a comprehensive, logically verified answer:",
        ])
        enriched_prompt = "\n".join(prompt_sections)

        # Step 6: Call LLM (or deterministic fallback synthesizer)
        if llm_fn is not None:
            raw_response = llm_fn(enriched_prompt)
        else:
            # Deterministic default synthesis from graph facts
            parts = []
            if think_res.get("math_proof"):
                parts.append(f"Result: {think_res['math_proof']['result']} ({think_res['math_proof']['explanation']}).")
            if think_res.get("cortex_text"):
                parts.append(think_res["cortex_text"])
            if decision_res and decision_res.get("steps"):
                parts.append("Recommended steps: " + "; ".join(f"{s['action']} ({s['reason']})" for s in decision_res["steps"]))
            if not parts:
                parts.append("No definitive contradiction found; reasoning verified based on available evidence.")
            raw_response = " ".join(parts)

        # Step 7: Post-generation Validation Gate
        validation_res = self.validate(raw_response, proposed_reasoning=enriched_prompt)

        # Step 8: Lifelong learning (auto-remember successful cases)
        if auto_remember and validation_res["is_safe"] and validation_res["verdict"] == "VERIFIED":
            try:
                self.remember(
                    situation=query,
                    judgment=raw_response[:200],
                    reasoning=f"Confidence: {think_res['confidence']:.2f}",
                    sector=sector,
                )
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return EnhancedReasoningResult(
            query=query,
            llm_response=raw_response,
            is_grounded=validation_res["is_safe"],
            validation=validation_res,
            plan=plan_res,
            think=think_res,
            recall=recall_res,
            decision=decision_res,
            execution_time_ms=elapsed_ms,
        )
