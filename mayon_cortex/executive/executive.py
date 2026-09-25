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
Executive Controller — Complexity Classification & Strategy Selection
=======================================================================
Phase 1 of the Brain-Like Intelligence upgrade.

The "prefrontal cortex" that decides HOW to think:
  - Trivial questions → System 1 (fast, reactive query)
  - Complex questions → System 2 (slow, deliberative thinking)
  - Multi-part questions → Decompose into sub-questions
  - Contradictory evidence → Contrast and resolve
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Complexity(Enum):
    """How complex the question is."""
    TRIVIAL = "trivial"            # "What is X?" → direct lookup
    MODERATE = "moderate"          # "How does X relate to Y?" → short chain
    COMPLEX = "complex"            # "Should we do X given Y and Z?" → multi-step
    MULTI_PART = "multi_part"      # "What is X, and how does it affect Y?" → decompose


class ThinkingStrategy(Enum):
    """Which reasoning strategy to use."""
    DIRECT_LOOKUP = "direct_lookup"          # System 1: single activation wave
    SHORT_CHAIN = "short_chain"              # System 2 lite: 2-3 reasoning steps
    FULL_DELIBERATION = "full_deliberation"  # System 2: full thinking loop
    DECOMPOSE_AND_SOLVE = "decompose"        # Split into sub-problems
    CONTRAST_AND_COMPARE = "contrast"        # Compare multiple paths
    HYPOTHESIZE_AND_TEST = "hypothesize"     # Generate hypothesis, then verify


@dataclass
class ExecutivePlan:
    """The executive's plan for how to answer a question."""
    complexity: Complexity
    strategy: ThinkingStrategy
    estimated_steps: int
    sub_questions: List[str] = field(default_factory=list)
    focus_sectors: List[str] = field(default_factory=list)
    explanation: str = ""


# Question complexity indicators
_COMPLEX_MARKERS = {
    "should", "would", "could", "recommend", "compare",
    "evaluate", "analyze", "decide", "tradeoff", "risk",
    "implications", "consequences", "impact", "versus", "vs",
    "pros and cons", "advantages", "disadvantages",
    "why does", "how does", "what if",
}

_MULTI_PART_MARKERS = {
    " and ", " also ", " additionally ", " furthermore ",
    " plus ", " as well as ", " both ", " along with ",
}

_TRIVIAL_PATTERNS = [
    r"^what is \w+",
    r"^who is \w+",
    r"^define \w+",
    r"^what does \w+ mean",
    r"^where is \w+",
]


class ExecutiveController:
    """
    Classifies query complexity and selects the optimal thinking strategy.
    
    Acts as the "prefrontal cortex" — the decision-maker that routes
    queries to the appropriate reasoning system.
    """

    def __init__(self, graph: Optional[Any] = None, embedder: Optional[Any] = None):
        self.graph = graph
        self.embedder = embedder

    def plan(
        self,
        question: str,
        graph: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ) -> ExecutivePlan:
        """
        Analyze a question and produce an execution plan.
        
        Returns:
            ExecutivePlan with complexity, strategy, and sub-questions
        """
        g = graph or self.graph
        e = embedder or self.embedder

        complexity = self._classify_complexity(question, g, e)
        strategy = self._select_strategy(complexity, question, g)
        sub_questions = self._decompose_if_needed(question, complexity)
        focus_sectors = self._detect_sectors(question, g)
        estimated_steps = self._estimate_steps(complexity, strategy)
        explanation = self._explain_plan(complexity, strategy, sub_questions)

        return ExecutivePlan(
            complexity=complexity,
            strategy=strategy,
            estimated_steps=estimated_steps,
            sub_questions=sub_questions,
            focus_sectors=focus_sectors,
            explanation=explanation,
        )

    def _classify_complexity(
        self,
        question: str,
        graph: Optional[Any],
        embedder: Optional[Any],
    ) -> Complexity:
        """Classify the complexity of a question."""
        q_lower = question.lower().strip()
        word_count = len(q_lower.split())

        # Check trivial patterns
        for pattern in _TRIVIAL_PATTERNS:
            if re.match(pattern, q_lower):
                return Complexity.TRIVIAL

        # Check multi-part markers
        for marker in _MULTI_PART_MARKERS:
            if marker in q_lower:
                # Count question marks or conjunctions
                parts = [p.strip() for p in re.split(r'[?;]|(?:\band\b)', q_lower) if len(p.strip()) > 5]
                if len(parts) >= 2:
                    return Complexity.MULTI_PART

        # Check complex markers
        complex_count = sum(1 for m in _COMPLEX_MARKERS if m in q_lower)
        if complex_count >= 2 or word_count > 20:
            return Complexity.COMPLEX

        if complex_count >= 1 or word_count > 10:
            return Complexity.MODERATE

        # Check graph density (if available) — more edges = potentially more complex reasoning
        if graph and embedder:
            query_vec = embedder.encode_single(question)
            if graph.num_nodes > 0:
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    q_normed = (query_vec / norm).reshape(1, -1).astype(np.float32)
                    scores, _ = graph.faiss_index.search(q_normed, min(5, graph.num_nodes))
                    avg_sim = float(np.mean(scores[0][scores[0] > 0])) if np.any(scores[0] > 0) else 0
                    # Very high similarity → direct lookup available
                    if avg_sim > 0.85:
                        return Complexity.TRIVIAL

        return Complexity.MODERATE

    def _select_strategy(
        self,
        complexity: Complexity,
        question: str,
        graph: Optional[Any],
    ) -> ThinkingStrategy:
        """Select the best thinking strategy for the complexity level."""
        q_lower = question.lower()

        if complexity == Complexity.TRIVIAL:
            return ThinkingStrategy.DIRECT_LOOKUP

        if complexity == Complexity.MULTI_PART:
            return ThinkingStrategy.DECOMPOSE_AND_SOLVE

        if complexity == Complexity.COMPLEX:
            # Check if it's a comparison question
            compare_words = {"compare", "versus", "vs", "difference", "differ", "contrast"}
            if any(w in q_lower for w in compare_words):
                return ThinkingStrategy.CONTRAST_AND_COMPARE

            # Check if it's a hypothetical
            hypo_words = {"what if", "suppose", "hypothetically", "assuming", "imagine if"}
            if any(w in q_lower for w in hypo_words):
                return ThinkingStrategy.HYPOTHESIZE_AND_TEST

            return ThinkingStrategy.FULL_DELIBERATION

        # MODERATE
        return ThinkingStrategy.SHORT_CHAIN

    def _decompose_if_needed(
        self,
        question: str,
        complexity: Complexity,
    ) -> List[str]:
        """Decompose multi-part or complex questions into sub-questions."""
        if complexity not in (Complexity.MULTI_PART, Complexity.COMPLEX):
            return []

        sub_questions: List[str] = []
        q_lower = question.lower()

        # Split on conjunctions and question marks
        parts = re.split(r'[?]|\band\b|\balso\b', question)
        parts = [p.strip().rstrip('?.! ') for p in parts if len(p.strip()) > 5]

        if len(parts) >= 2:
            for part in parts:
                # Add question mark if not present
                if not part.endswith("?"):
                    # Convert statement fragments to questions
                    if not any(part.lower().startswith(w) for w in ["what", "who", "how", "why", "where", "when", "is", "are", "does", "do", "can", "should"]):
                        part = f"What about {part}?"
                    else:
                        part = f"{part}?"
                sub_questions.append(part)
        elif complexity == Complexity.COMPLEX:
            # For complex single questions, generate analytical sub-questions
            # Extract key entities
            words = question.split()
            nouns = [w for w in words if len(w) > 3 and w[0].isupper()]
            if len(nouns) >= 2:
                sub_questions.append(f"What is {nouns[0]}?")
                sub_questions.append(f"What is {nouns[1]}?")
                sub_questions.append(f"How does {nouns[0]} relate to {nouns[1]}?")

        return sub_questions

    def _detect_sectors(
        self,
        question: str,
        graph: Optional[Any],
    ) -> List[str]:
        """Detect which knowledge sectors are relevant to this question."""
        q_lower = question.lower()
        sectors = []

        # Keyword-based sector detection
        sector_keywords = {
            "medical": {"drug", "medicine", "patient", "disease", "treatment", "symptom", "diagnosis"},
            "legal": {"law", "legal", "court", "regulation", "statute", "compliance", "policy"},
            "finance": {"cost", "price", "budget", "financial", "revenue", "profit", "investment"},
            "engineering": {"design", "specification", "material", "structure", "mechanical", "electrical"},
            "science": {"theory", "experiment", "hypothesis", "research", "study", "evidence"},
            "code": {"function", "class", "algorithm", "code", "program", "software", "api"},
        }

        for sector, keywords in sector_keywords.items():
            if any(kw in q_lower for kw in keywords):
                sectors.append(sector)

        # If no specific sector detected, use general
        if not sectors:
            sectors.append("general")

        return sectors

    def _estimate_steps(
        self,
        complexity: Complexity,
        strategy: ThinkingStrategy,
    ) -> int:
        """Estimate how many reasoning steps are needed."""
        step_estimates = {
            ThinkingStrategy.DIRECT_LOOKUP: 1,
            ThinkingStrategy.SHORT_CHAIN: 3,
            ThinkingStrategy.FULL_DELIBERATION: 8,
            ThinkingStrategy.DECOMPOSE_AND_SOLVE: 10,
            ThinkingStrategy.CONTRAST_AND_COMPARE: 6,
            ThinkingStrategy.HYPOTHESIZE_AND_TEST: 7,
        }
        return step_estimates.get(strategy, 5)

    def _explain_plan(
        self,
        complexity: Complexity,
        strategy: ThinkingStrategy,
        sub_questions: List[str],
    ) -> str:
        """Generate human-readable explanation of the plan."""
        parts = [f"Complexity: {complexity.value}, Strategy: {strategy.value}"]
        if sub_questions:
            parts.append(f"Decomposed into {len(sub_questions)} sub-questions:")
            for i, sq in enumerate(sub_questions, 1):
                parts.append(f"  {i}. {sq}")
        return "\n".join(parts)


@dataclass
class SubGoal:
    """A specific sub-goal in an execution plan."""
    goal: str
    status: str = "pending"
    result: str = ""
    confidence: float = 0.0


# Aliases
ExecutionPlan = ExecutivePlan

