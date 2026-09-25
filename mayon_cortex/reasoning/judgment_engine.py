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
Judgment Engine — Precedent-Based Decision Making
===================================================
Phase 7 of the Brain-Like Intelligence upgrade.

Like a judge: find precedent → analyze similarities/differences →
render judgment → EXPLAIN the reasoning.

This is NOT hallucination — it's applying LEARNED PATTERNS from actual examples.
The system can only judge based on cases it has been taught.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mayon_cortex.memory.case_memory import Case, CaseMatch, CaseMemory
from mayon_cortex.core.config import CortexConfig


@dataclass
class Judgment:
    """A rendered judgment with full explanation."""
    situation: str
    verdict: str
    confidence: float
    reasoning_chain: List[str]
    precedents_used: List[CaseMatch]
    key_factors: Dict[str, str]
    dissent: Optional[str] = None
    uncertainty_notes: List[str] = field(default_factory=list)

    @property
    def rationale(self) -> str:
        """Convenience property for text summary of reasoning."""
        return "\n".join(self.reasoning_chain) if self.reasoning_chain else self.verdict


# Aliases
JudgmentVerdict = Judgment


class JudgmentEngine:
    """
    Renders judgments by analyzing precedent cases and explaining reasoning.
    """

    def __init__(
        self,
        case_memory: CaseMemory,
        config: Optional[CortexConfig] = None,
        cortex: Optional[Any] = None,
    ):
        self.case_memory = case_memory
        self.config = config or CortexConfig()
        self.cortex = cortex

    def judge(
        self,
        situation: str,
        graph: Optional[Any] = None,
        embedder: Optional[Any] = None,
        relevant_facts: Optional[List[str]] = None,
    ) -> Judgment:
        """
        Render a judgment on a new situation based on learned precedents.
        """
        g = graph or (self.cortex.graph if self.cortex else None)
        emb = embedder or (self.cortex.embedder if self.cortex else None)

        """
        Render a judgment on a new situation based on learned precedents.
        
        Returns a Judgment with verdict, reasoning chain, precedents used,
        and optional dissent.
        """
        # Step 1: Find precedents
        precedents = self.case_memory.find_precedents(
            situation, g, emb, top_k=5
        )


        if not precedents:
            return Judgment(
                situation=situation,
                verdict="INSUFFICIENT PRECEDENT — Cannot render judgment",
                confidence=0.0,
                reasoning_chain=[
                    "Step 1: Searched for similar past cases",
                    "Step 2: No relevant precedents found in case memory",
                    "Conclusion: Cannot render a judgment without precedent. "
                    "Teach more cases to improve judgment capability.",
                ],
                precedents_used=[],
                key_factors={},
                uncertainty_notes=["No precedent cases available"],
            )

        # Step 2: Analyze factors
        new_factors = self.case_memory._extract_factors_basic(situation)
        key_factors: Dict[str, str] = {}

        for factor in new_factors[:5]:
            # Determine how this factor influenced precedent judgments
            influence = self._analyze_factor_influence(factor, precedents)
            key_factors[factor] = influence

        # Step 3: Weight precedent judgments
        verdict, confidence = self._synthesize_verdict(precedents)

        # Step 4: Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(
            situation, precedents, new_factors, key_factors
        )

        # Step 5: Check for dissent (counter-precedents)
        dissent = self._find_dissent(precedents)

        # Step 6: Note uncertainties
        uncertainty_notes = self._assess_uncertainty(precedents, new_factors)

        return Judgment(
            situation=situation,
            verdict=verdict,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            precedents_used=precedents,
            key_factors=key_factors,
            dissent=dissent,
            uncertainty_notes=uncertainty_notes,
        )

    def learn_from_feedback(
        self,
        judgment: Judgment,
        correct_verdict: str,
        correct_reasoning: str,
        graph: Any,
        embedder: Any,
    ) -> str:
        """
        Learn from a corrected judgment.
        
        When a user corrects a judgment:
        1. Store the corrected case as a new precedent
        2. Future similar cases will use this corrected precedent
        
        Returns the new case_id.
        """
        # Extract factors from the situation
        factors = list(judgment.key_factors.keys())

        # Store as new case with high confidence
        case_id = self.case_memory.store_case(
            situation=judgment.situation,
            judgment=correct_verdict,
            reasoning=correct_reasoning,
            graph=graph,
            embedder=embedder,
            sector="general",
            factors=factors,
        )

        return case_id

    def _synthesize_verdict(
        self,
        precedents: List[CaseMatch],
    ) -> tuple:
        """Synthesize a verdict from multiple precedents."""
        if not precedents:
            return "No judgment possible", 0.0

        # Weighted voting by similarity
        verdict_votes: Dict[str, float] = {}
        total_weight = 0.0

        for p in precedents:
            verdict = p.case.judgment
            weight = p.similarity * p.case.confidence
            verdict_votes[verdict] = verdict_votes.get(verdict, 0.0) + weight
            total_weight += weight

        if not verdict_votes or total_weight == 0:
            return "Uncertain", 0.0

        # Pick the verdict with most weighted votes
        best_verdict = max(verdict_votes, key=verdict_votes.get)
        confidence = verdict_votes[best_verdict] / total_weight

        # Adjust confidence based on precedent agreement
        unique_verdicts = len(verdict_votes)
        if unique_verdicts == 1:
            confidence = min(1.0, confidence * 1.1)  # Unanimous
        elif unique_verdicts > 2:
            confidence *= 0.8  # Disagreement reduces confidence

        return best_verdict, min(1.0, confidence)

    def _build_reasoning_chain(
        self,
        situation: str,
        precedents: List[CaseMatch],
        new_factors: List[str],
        key_factors: Dict[str, str],
    ) -> List[str]:
        """Build a step-by-step reasoning chain."""
        chain = []

        # Step 1: Identify the situation
        chain.append(f"Step 1: Analyzing situation: '{situation[:100]}...'")

        # Step 2: Factors identified
        if new_factors:
            chain.append(
                f"Step 2: Key factors identified: {', '.join(new_factors[:5])}"
            )
        else:
            chain.append("Step 2: No specific decision factors extracted")

        # Step 3: Precedent analysis
        for i, p in enumerate(precedents[:3], start=3):
            shared_str = ", ".join(p.shared_factors[:3]) if p.shared_factors else "none specific"
            diff_str = ", ".join(p.different_factors[:3]) if p.different_factors else "none"
            chain.append(
                f"Step {i}: Precedent '{p.case.case_id}' (similarity={p.similarity:.2f}): "
                f"Judgment was '{p.case.judgment[:80]}'. "
                f"Shared factors: [{shared_str}]. "
                f"Different factors: [{diff_str}]."
            )

        # Step 4: Factor influence
        step_num = 3 + min(len(precedents), 3)
        for factor, influence in list(key_factors.items())[:3]:
            step_num += 1
            chain.append(f"Step {step_num}: Factor '{factor}': {influence}")

        # Step 5: Reasoning from precedent
        if precedents:
            top = precedents[0]
            step_num += 1
            chain.append(
                f"Step {step_num}: Primary reasoning from {top.case.case_id}: "
                f"{top.case.reasoning[:200]}"
            )

        # Final conclusion
        step_num += 1
        chain.append(
            f"Step {step_num}: Based on {len(precedents)} precedent(s), "
            f"rendering judgment."
        )

        return chain

    def _find_dissent(self, precedents: List[CaseMatch]) -> Optional[str]:
        """Look for counter-precedents that disagree with the majority."""
        if len(precedents) < 2:
            return None

        # Get the majority verdict
        verdicts = [p.case.judgment for p in precedents]
        from collections import Counter
        verdict_counts = Counter(verdicts)
        majority_verdict = verdict_counts.most_common(1)[0][0]

        # Find dissenting precedents
        dissenters = [
            p for p in precedents
            if p.case.judgment != majority_verdict and p.similarity > 0.4
        ]

        if dissenters:
            d = dissenters[0]
            return (
                f"Counter-argument from {d.case.case_id} "
                f"(similarity={d.similarity:.2f}): "
                f"Judgment was '{d.case.judgment}' because: {d.case.reasoning[:150]}"
            )

        return None

    def _assess_uncertainty(
        self,
        precedents: List[CaseMatch],
        new_factors: List[str],
    ) -> List[str]:
        """Assess sources of uncertainty in the judgment."""
        notes = []

        if not precedents:
            notes.append("No precedent cases found")
            return notes

        # Low similarity
        max_sim = max(p.similarity for p in precedents)
        if max_sim < 0.6:
            notes.append(
                f"Best precedent match is only {max_sim:.1%} similar — judgment may be unreliable"
            )

        # Conflicting precedents
        verdicts = set(p.case.judgment for p in precedents)
        if len(verdicts) > 1:
            notes.append(
                f"Precedents disagree: {len(verdicts)} different judgments found"
            )

        # New factors not in precedents
        all_precedent_factors = set()
        for p in precedents:
            all_precedent_factors.update(f.lower() for f in p.case.factors)

        novel_factors = [f for f in new_factors if f.lower() not in all_precedent_factors]
        if novel_factors:
            notes.append(
                f"Novel factors not seen in precedents: {', '.join(novel_factors[:3])}"
            )

        return notes

    def _analyze_factor_influence(
        self,
        factor: str,
        precedents: List[CaseMatch],
    ) -> str:
        """Analyze how a factor influenced precedent judgments."""
        factor_lower = factor.lower()

        influence_notes = []
        for p in precedents[:3]:
            if factor_lower in [f.lower() for f in p.case.factors]:
                weight = p.case.factor_weights.get(factor, 0.5)
                influence_notes.append(
                    f"In {p.case.case_id}: weight={weight:.1f}, judgment='{p.case.judgment[:50]}'"
                )

        if influence_notes:
            return "; ".join(influence_notes)
        return "No direct influence observed in precedents"
