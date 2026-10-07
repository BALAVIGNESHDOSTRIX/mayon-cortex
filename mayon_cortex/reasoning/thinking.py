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
Thinking Loop — System 2 Deliberative Reasoning
==================================================
Phase 1 of the Brain-Like Intelligence upgrade.

The "inner monologue" that chains multiple reasoning steps:
  1. Executive classifies complexity → selects strategy
  2. Loop iterates: observe → reason → check → decide
  3. MetaCognition monitors progress, advises strategy switches
  4. LogicalChainer derives new facts via inference rules
  5. ThoughtScratchpad records everything for explainability

This is the bridge from System 1 (reactive) to System 2 (deliberative).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from mayon_cortex.memory.thought_scratchpad import (
    ThoughtScratchpad,
    ThoughtType,
)
from mayon_cortex.reasoning.logic import LogicalChainer, InferenceChain
from mayon_cortex.executive.metacognition import (
    MetaCognition,
    MetaCognitionReport,
    ReasoningHealth,
    StrategyAdvice,
)
from mayon_cortex.executive.executive import (
    ExecutiveController,
    ExecutivePlan,
    Complexity,
    ThinkingStrategy,
)


@dataclass
class ThinkingResult:
    """Complete result of a System 2 thinking session."""
    question: str
    answer: str
    confidence: float
    reasoning_trace: str                    # Human-readable thought trace
    steps_taken: int
    strategy_used: str
    complexity: str
    sub_answers: List[Dict[str, str]] = field(default_factory=list)
    inference_chains: List[InferenceChain] = field(default_factory=list)
    meta_reports: List[MetaCognitionReport] = field(default_factory=list)
    scratchpad_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def conclusion(self) -> str:
        return self.answer

    @property
    def subgoals(self) -> List[Any]:
        from mayon_cortex.executive.executive import SubGoal
        return [
            SubGoal(goal=sa.get("question", sa.get("goal", str(sa))), status="done", result=sa.get("answer", ""))
            for sa in self.sub_answers
        ] if self.sub_answers else [SubGoal(goal=self.question, status="done", result=self.answer)]

    @property
    def metacognition(self) -> Dict[str, Any]:
        last_report = self.meta_reports[-1] if self.meta_reports else None
        state_str = last_report.health.value if last_report else "confident"
        return {
            "epistemic_state": state_str,
            "reports": self.meta_reports,
        }



class ThinkingLoop:
    """
    Multi-step deliberative reasoning loop.
    
    The core System 2 engine that transforms cortex-graph from reactive
    (query → answer) to deliberative (query → think → reason → verify → answer).
    
    Algorithm:
      1. Executive classifies question → produces ExecutivePlan
      2. For each step (up to budget):
         a. Query the graph (System 1 call) for relevant evidence
         b. Apply logical inference rules on the evidence
         c. Record observations, deductions, conclusions in scratchpad
         d. MetaCognition evaluates: continue, switch, or conclude?
      3. Synthesize final answer from all conclusions
      4. Return full reasoning trace for explainability
    """

    def __init__(
        self,
        max_steps: int = 15,
        min_confidence: float = 0.6,
    ):
        self.max_steps = max_steps
        self.min_confidence = min_confidence

        # Sub-systems
        self.executive = ExecutiveController()
        self.logic = LogicalChainer(max_chain_depth=6, min_confidence=0.1)
        self.meta = MetaCognition(
            max_thinking_steps=max_steps,
            min_confidence_to_conclude=min_confidence,
        )
        self.scratchpad = ThoughtScratchpad(max_thoughts=200)

    def think(
        self,
        question: str,
        graph: Any,
        embedder: Any,
        wave: Any,
        ndu: Any,
        ranker: Any,
        assembler: Any,
        uncertainty: Any,
        corrector: Any,
    ) -> ThinkingResult:
        """
        Main entry point for System 2 deliberative reasoning.
        
        Args:
            question: The question to reason about
            graph: MayonGraph knowledge graph
            embedder: Text→vector embedding model
            wave: ActivationWave for graph traversal
            ndu: NodeDecisionUnit for firing decisions
            ranker: PathRanker for proof chain ranking
            assembler: SentenceAssembler for text generation
            uncertainty: UncertaintyGate for confidence assessment
            corrector: SelfCorrector for contradiction removal
            
        Returns:
            ThinkingResult with answer, reasoning trace, and metadata
        """
        # Reset for new session
        self.scratchpad.reset()
        self.meta.reset()

        # Step 1: Executive classification
        plan = self.executive.plan(question, graph, embedder)

        self.scratchpad.push_goal(f"Answer: {question}")
        self.scratchpad.record(
            ThoughtType.OBSERVATION,
            f"Executive assessment: {plan.complexity.value} complexity, "
            f"strategy: {plan.strategy.value}",
            confidence=0.9,
            source="executive",
        )

        # Route based on strategy
        if plan.strategy == ThinkingStrategy.DIRECT_LOOKUP:
            return self._direct_lookup(question, graph, embedder, wave, ndu, ranker,
                                       assembler, uncertainty, corrector, plan)

        if plan.strategy == ThinkingStrategy.DECOMPOSE_AND_SOLVE:
            return self._decompose_and_solve(question, plan, graph, embedder, wave,
                                             ndu, ranker, assembler, uncertainty, corrector)

        if plan.strategy == ThinkingStrategy.CONTRAST_AND_COMPARE:
            return self._contrast_compare(question, plan, graph, embedder, wave,
                                          ndu, ranker, assembler, uncertainty, corrector)

        # Default: iterative deliberation (SHORT_CHAIN or FULL_DELIBERATION)
        return self._deliberate(question, plan, graph, embedder, wave, ndu,
                                ranker, assembler, uncertainty, corrector)

    def _direct_lookup(
        self, question, graph, embedder, wave, ndu, ranker,
        assembler, uncertainty, corrector, plan,
    ) -> ThinkingResult:
        """System 1: single activation wave lookup."""
        query_vec = embedder.encode_single(question)
        activation = wave.activate(query_vec, graph, ndu)
        scored = ranker.rank_paths(activation.paths, query_vec, graph)

        if scored:
            clean, _ = corrector.correct(scored, graph)
            scored = clean if clean else scored

        assessment = uncertainty.evaluate(scored, query_vec, graph)
        text = assembler.assemble(scored, graph=graph, thought_vector=query_vec)

        self.scratchpad.record(
            ThoughtType.CONCLUSION,
            text[:200],
            confidence=assessment.confidence,
            source="direct_lookup",
        )
        self.scratchpad.resolve_goal(f"Answer: {question}", text[:200], assessment.confidence)

        return ThinkingResult(
            question=question,
            answer=text,
            confidence=assessment.confidence,
            reasoning_trace=self.scratchpad.reasoning_trace(),
            steps_taken=1,
            strategy_used=plan.strategy.value,
            complexity=plan.complexity.value,
            scratchpad_summary=self.scratchpad.summary(),
        )

    def _deliberate(
        self, question, plan, graph, embedder, wave, ndu,
        ranker, assembler, uncertainty, corrector,
    ) -> ThinkingResult:
        """
        Iterative deliberation loop — the core System 2 algorithm.
        
        For each step:
          1. Query graph for evidence related to current goal
          2. Apply logical inference rules
          3. Record findings in scratchpad
          4. Check meta-cognition: continue or conclude?
        """
        query_vec = embedder.encode_single(question)
        all_inference_chains: List[InferenceChain] = []
        meta_reports: List[MetaCognitionReport] = []
        all_evidence_texts: List[str] = []

        for step in range(plan.estimated_steps):
            # Check for loops
            if self.scratchpad.check_loop():
                self.scratchpad.record(
                    ThoughtType.BACKTRACK,
                    "Circular reasoning detected, breaking loop",
                    source="thinking_loop",
                )
                break

            # Step A: Query the graph
            current_goal = self.scratchpad.current_goal()
            search_text = current_goal.description if current_goal else question
            search_vec = embedder.encode_single(search_text)

            activation = wave.activate(search_vec, graph, ndu)
            scored = ranker.rank_paths(activation.paths, search_vec, graph)

            if scored:
                clean, _ = corrector.correct(scored, graph)
                scored = clean if clean else scored

            # Step B: Record observations
            if scored:
                top_path = scored[0]
                path_obj = top_path.path if hasattr(top_path, "path") else top_path
                node_ids = path_obj.node_ids if hasattr(path_obj, "node_ids") else []

                # Build evidence text from path
                evidence_parts = []
                for nid in node_ids[:5]:
                    if nid in graph.nodes:
                        evidence_parts.append(graph.nodes[nid].source_text)

                evidence_text = " → ".join(evidence_parts)
                if evidence_text:
                    all_evidence_texts.append(evidence_text)
                    self.scratchpad.record(
                        ThoughtType.OBSERVATION,
                        f"Evidence found: {evidence_text[:150]}",
                        confidence=top_path.score if hasattr(top_path, "score") else 0.5,
                        source="activation_wave",
                    )

                # Step C: Apply logical inference
                if node_ids:
                    for nid in node_ids[:3]:
                        chains = self.logic.forward_chain(nid, graph, max_steps=2)
                        for chain in chains[:2]:
                            all_inference_chains.append(chain)
                            if chain.steps:
                                self.scratchpad.record(
                                    ThoughtType.DEDUCTION,
                                    f"Inferred: {chain.steps[-1].explanation[:150]}",
                                    confidence=chain.total_confidence,
                                    source="logical_chainer",
                                )
            else:
                self.scratchpad.record(
                    ThoughtType.OBSERVATION,
                    f"No strong evidence found for: {search_text[:80]}",
                    confidence=0.1,
                    source="activation_wave",
                )

            # Step D: Meta-cognitive check
            report = self.meta.evaluate(self.scratchpad)
            meta_reports.append(report)

            if report.advice == StrategyAdvice.CONCLUDE:
                self.scratchpad.record(
                    ThoughtType.CONCLUSION,
                    f"MetaCognition advises conclusion: {report.explanation}",
                    confidence=report.confidence,
                    source="metacognition",
                )
                break

            if report.advice == StrategyAdvice.SWITCH_STRATEGY:
                self.scratchpad.record(
                    ThoughtType.BACKTRACK,
                    "Switching strategy: trying broader search",
                    source="metacognition",
                )
                # Try backward chaining
                back_chains = self.logic.backward_chain(
                    question, "treats", graph, embedder, max_steps=3
                )
                for bc in back_chains[:2]:
                    all_inference_chains.append(bc)
                    if bc.steps:
                        self.scratchpad.record(
                            ThoughtType.DEDUCTION,
                            f"Backward inference: {bc.steps[-1].explanation[:150]}",
                            confidence=bc.total_confidence,
                            source="backward_chainer",
                        )

        # Synthesize final answer
        answer = self._synthesize_answer(
            question, all_evidence_texts, all_inference_chains,
            assembler, graph, query_vec, scored if scored else [],
        )

        # Compute final confidence
        conclusions = self.scratchpad.conclusions
        final_confidence = (
            sum(c.confidence for c in conclusions) / len(conclusions)
            if conclusions else 0.3
        )

        self.scratchpad.resolve_goal(
            f"Answer: {question}", answer[:200], final_confidence
        )

        return ThinkingResult(
            question=question,
            answer=answer,
            confidence=final_confidence,
            reasoning_trace=self.scratchpad.reasoning_trace(),
            steps_taken=self.scratchpad.step_count,
            strategy_used=plan.strategy.value,
            complexity=plan.complexity.value,
            inference_chains=all_inference_chains,
            meta_reports=meta_reports,
            scratchpad_summary=self.scratchpad.summary(),
        )

    def _decompose_and_solve(
        self, question, plan, graph, embedder, wave, ndu,
        ranker, assembler, uncertainty, corrector,
    ) -> ThinkingResult:
        """Decompose into sub-questions and solve each."""
        sub_answers: List[Dict[str, str]] = []
        all_evidence: List[str] = []
        total_confidence = 0.0

        if not plan.sub_questions:
            # Fallback to deliberation if decomposition failed
            return self._deliberate(question, plan, graph, embedder, wave, ndu,
                                    ranker, assembler, uncertainty, corrector)

        for i, sub_q in enumerate(plan.sub_questions):
            self.scratchpad.push_goal(f"Sub-question {i+1}: {sub_q}", parent=question)

            # Solve each sub-question with direct lookup
            sub_vec = embedder.encode_single(sub_q)
            activation = wave.activate(sub_vec, graph, ndu)
            scored = ranker.rank_paths(activation.paths, sub_vec, graph)

            if scored:
                clean, _ = corrector.correct(scored, graph)
                scored = clean if clean else scored
                sub_text = assembler.assemble(scored, graph=graph, thought_vector=sub_vec)
                sub_conf = scored[0].score if hasattr(scored[0], "score") else 0.5
            else:
                sub_text = f"No information found for: {sub_q}"
                sub_conf = 0.1

            sub_answers.append({"question": sub_q, "answer": sub_text})
            all_evidence.append(f"Q{i+1}: {sub_q} → {sub_text[:100]}")
            total_confidence += sub_conf

            self.scratchpad.record(
                ThoughtType.CONCLUSION,
                f"Sub-answer {i+1}: {sub_text[:150]}",
                confidence=sub_conf,
                source="decompose",
            )
            self.scratchpad.resolve_goal(f"Sub-question {i+1}: {sub_q}", sub_text[:100], sub_conf)

        # Synthesize combined answer
        combined_parts = [f"Regarding '{sq['question']}': {sq['answer']}" for sq in sub_answers]
        answer = " ".join(combined_parts)

        avg_confidence = total_confidence / max(len(plan.sub_questions), 1)
        self.scratchpad.resolve_goal(f"Answer: {question}", answer[:200], avg_confidence)

        return ThinkingResult(
            question=question,
            answer=answer,
            confidence=avg_confidence,
            reasoning_trace=self.scratchpad.reasoning_trace(),
            steps_taken=self.scratchpad.step_count,
            strategy_used=plan.strategy.value,
            complexity=plan.complexity.value,
            sub_answers=sub_answers,
            scratchpad_summary=self.scratchpad.summary(),
        )

    def _contrast_compare(
        self, question, plan, graph, embedder, wave, ndu,
        ranker, assembler, uncertainty, corrector,
    ) -> ThinkingResult:
        """Compare two concepts by finding their paths and contrasting."""
        self.scratchpad.record(
            ThoughtType.OBSERVATION,
            "Using contrast-and-compare strategy",
            source="executive",
        )

        # Extract comparison subjects from the question
        subjects = self._extract_comparison_subjects(question)
        sub_answers = []

        for subj in subjects[:2]:
            self.scratchpad.push_goal(f"Analyze: {subj}", parent=question)
            vec = embedder.encode_single(subj)
            activation = wave.activate(vec, graph, ndu)
            scored = ranker.rank_paths(activation.paths, vec, graph)

            if scored:
                clean, _ = corrector.correct(scored, graph)
                scored = clean if clean else scored
                text = assembler.assemble(scored, graph=graph, thought_vector=vec)
                conf = scored[0].score if hasattr(scored[0], "score") else 0.5
            else:
                text = f"Limited information available about {subj}"
                conf = 0.1

            sub_answers.append({"question": f"About {subj}", "answer": text})
            self.scratchpad.record(
                ThoughtType.OBSERVATION,
                f"About {subj}: {text[:150]}",
                confidence=conf,
                source="contrast",
            )
            self.scratchpad.resolve_goal(f"Analyze: {subj}", text[:100], conf)

        # Generate comparison conclusion
        if len(sub_answers) >= 2:
            comparison = (
                f"Comparing: {sub_answers[0]['answer'][:200]} "
                f"vs. {sub_answers[1]['answer'][:200]}"
            )
        else:
            comparison = sub_answers[0]["answer"] if sub_answers else "Unable to compare"

        self.scratchpad.record(
            ThoughtType.CONCLUSION,
            comparison[:200],
            confidence=0.6,
            source="contrast",
        )
        self.scratchpad.resolve_goal(f"Answer: {question}", comparison[:200], 0.6)

        return ThinkingResult(
            question=question,
            answer=comparison,
            confidence=0.6,
            reasoning_trace=self.scratchpad.reasoning_trace(),
            steps_taken=self.scratchpad.step_count,
            strategy_used=plan.strategy.value,
            complexity=plan.complexity.value,
            sub_answers=sub_answers,
            scratchpad_summary=self.scratchpad.summary(),
        )

    def _synthesize_answer(
        self, question, evidence_texts, inference_chains,
        assembler, graph, query_vec, scored_paths,
    ) -> str:
        """Synthesize a final answer from all collected evidence."""
        parts = []

        # Use assembler for the primary answer if paths available
        if scored_paths:
            primary = assembler.assemble(scored_paths, graph=graph, thought_vector=query_vec)
            parts.append(primary)

        # Add inference conclusions
        for chain in inference_chains[:3]:
            if chain.steps and chain.total_confidence > 0.3:
                last_step = chain.steps[-1]
                parts.append(f"Furthermore, {last_step.explanation}")

        if parts:
            return " ".join(parts)

        # Fallback: concatenate evidence
        if evidence_texts:
            return " ".join(evidence_texts[:3])

        return f"After {self.scratchpad.step_count} reasoning steps, insufficient evidence was found to answer: {question}"

    def _extract_comparison_subjects(self, question: str) -> List[str]:
        """Extract subjects being compared from a comparison question."""
        import re
        q = question.lower()

        # Try "X vs Y" or "X versus Y"
        vs_match = re.search(r'(.+?)\s+(?:vs\.?|versus|compared to|or)\s+(.+?)(?:\?|$)', q)
        if vs_match:
            return [vs_match.group(1).strip(), vs_match.group(2).strip()]

        # Try "compare X and Y"
        compare_match = re.search(r'compare\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\?|$)', q)
        if compare_match:
            return [compare_match.group(1).strip(), compare_match.group(2).strip()]

        # Try "difference between X and Y"
        diff_match = re.search(r'(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)', q)
        if diff_match:
            return [diff_match.group(1).strip(), diff_match.group(2).strip()]

        # Fallback: extract capitalized words as subjects
        words = question.split()
        subjects = [w for w in words if len(w) > 2 and w[0].isupper() and w.lower() not in {
            "what", "how", "why", "the", "and", "compare", "between"
        }]
        return subjects[:2] if subjects else [question]


# Aliases for backward/forward compatibility
ThoughtResult = ThinkingResult


class ThinkingEngine:
    """
    Convenience wrapper around ThinkingLoop bound to a CortexGraph instance.
    """

    def __init__(self, cortex: Optional[Any] = None, max_steps: int = 15, min_confidence: float = 0.6):
        self.cortex = cortex
        self.loop = ThinkingLoop(max_steps=max_steps, min_confidence=min_confidence)

    def think(
        self,
        prompt: str,
        mode: str = "balanced",
        timeout_ms: int = 5000,
        cortex: Optional[Any] = None,
    ) -> ThinkingResult:
        ctx = cortex or self.cortex
        if ctx is None:
            raise ValueError("CortexGraph instance required to run ThinkingEngine.")

        return self.loop.think(
            question=prompt,
            graph=ctx.graph,
            embedder=ctx.embedder,
            wave=ctx.wave,
            ndu=ctx.ndu,
            ranker=ctx.ranker,
            assembler=ctx.assembler,
            uncertainty=ctx.uncertainty,
            corrector=ctx.corrector,
        )

