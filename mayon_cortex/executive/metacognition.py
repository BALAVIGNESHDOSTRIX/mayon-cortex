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
MetaCognition — Self-Monitoring for System 2 Reasoning
========================================================
Phase 1 of the Brain-Like Intelligence upgrade.

Monitors the thinking process itself:
  - Progress tracking: are we making progress toward the goal?
  - Circularity detection: are we going in circles?
  - Strategy evaluation: is the current strategy working?
  - Confidence calibration: how confident should we be overall?
  - Effort budget: should we keep thinking or conclude?
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mayon_cortex.memory.thought_scratchpad import ThoughtScratchpad, ThoughtType


class ReasoningHealth(Enum):
    """Overall health of the current reasoning process."""
    PROGRESSING = "progressing"        # Making good progress
    SLOWING = "slowing"                # Progress is slowing down
    STALLED = "stalled"                # No progress being made
    CIRCULAR = "circular"              # Going in circles
    CONFIDENT = "confident"            # Ready to conclude
    EXHAUSTED = "exhausted"            # Budget spent, must conclude


class StrategyAdvice(Enum):
    """What the meta-cognition system recommends."""
    CONTINUE = "continue"              # Keep going with current strategy
    SWITCH_STRATEGY = "switch"         # Try a different approach
    DECOMPOSE = "decompose"            # Break the problem into sub-problems
    BROADEN = "broaden"                # Search more widely
    NARROW = "narrow"                  # Focus on most promising path
    CONCLUDE = "conclude"              # Stop and present findings
    BACKTRACK = "backtrack"            # Go back to a previous state


@dataclass
class MetaCognitionReport:
    """Report from the meta-cognitive monitoring system."""
    health: ReasoningHealth
    advice: StrategyAdvice
    progress_rate: float               # 0-1, how fast we're progressing
    confidence: float                  # 0-1, overall confidence in current reasoning
    steps_taken: int
    steps_remaining: int               # Estimated steps to conclusion
    circular_risk: float               # 0-1, probability of circular reasoning
    explanation: str                   # Why this advice was given


class MetaCognition:
    """
    Self-monitoring system for deliberative reasoning.
    
    Watches the ThoughtScratchpad and evaluates:
    1. Are we making progress? (new conclusions vs repeated observations)
    2. Are we going in circles? (revisiting states)
    3. Should we switch strategy? (current approach not yielding results)
    4. Should we stop? (confident enough or budget exhausted)
    """

    def __init__(
        self,
        max_thinking_steps: int = 15,
        min_confidence_to_conclude: float = 0.6,
        stall_window: int = 4,
    ):
        self.max_thinking_steps = max_thinking_steps
        self.min_confidence_to_conclude = min_confidence_to_conclude
        self.stall_window = stall_window

        # Tracking
        self._progress_history: List[float] = []
        self._strategy_switches: int = 0

    def evaluate(self, scratchpad: ThoughtScratchpad) -> MetaCognitionReport:
        """
        Evaluate the current state of reasoning and provide guidance.
        """
        steps = scratchpad.step_count
        thoughts = scratchpad.thoughts
        conclusions = scratchpad.conclusions
        contradictions = scratchpad.contradictions
        unresolved = scratchpad.unresolved_goals

        # Calculate progress rate
        progress_rate = self._compute_progress_rate(thoughts, conclusions)
        self._progress_history.append(progress_rate)

        # Calculate confidence
        confidence = self._compute_confidence(conclusions, contradictions, steps)

        # Detect circularity
        circular_risk = self._compute_circular_risk(scratchpad)

        # Determine health
        health = self._assess_health(
            progress_rate, circular_risk, confidence, steps
        )

        # Generate advice
        advice = self._generate_advice(
            health, progress_rate, confidence, steps, unresolved
        )

        # Estimate remaining steps
        steps_remaining = self._estimate_remaining(
            steps, progress_rate, len(unresolved)
        )

        # Generate explanation
        explanation = self._explain(health, advice, progress_rate, confidence, steps)

        return MetaCognitionReport(
            health=health,
            advice=advice,
            progress_rate=progress_rate,
            confidence=confidence,
            steps_taken=steps,
            steps_remaining=steps_remaining,
            circular_risk=circular_risk,
            explanation=explanation,
        )

    def _compute_progress_rate(
        self,
        thoughts: List,
        conclusions: List,
    ) -> float:
        """How much progress we're making (ratio of useful thoughts)."""
        if not thoughts:
            return 0.0

        recent = thoughts[-self.stall_window:]
        useful_types = {
            ThoughtType.DEDUCTION,
            ThoughtType.CONCLUSION,
            ThoughtType.OBSERVATION,
        }
        useful = sum(1 for t in recent if t.thought_type in useful_types)
        return useful / len(recent)

    def _compute_confidence(
        self,
        conclusions: List,
        contradictions: List,
        steps: int,
    ) -> float:
        """Overall confidence in the reasoning so far."""
        if not conclusions:
            return 0.0

        # Average conclusion confidence
        avg_conf = sum(c.confidence for c in conclusions) / len(conclusions)

        # Penalty for contradictions
        contradiction_penalty = min(0.3, len(contradictions) * 0.1)

        # Bonus for multiple confirming conclusions
        confirmation_bonus = min(0.2, (len(conclusions) - 1) * 0.05)

        return max(0.0, min(1.0, avg_conf - contradiction_penalty + confirmation_bonus))

    def _compute_circular_risk(self, scratchpad: ThoughtScratchpad) -> float:
        """Probability that reasoning is circular."""
        if scratchpad.step_count < 3:
            return 0.0

        # Check for explicit loop detection
        backtrack_count = sum(
            1 for t in scratchpad.thoughts
            if t.thought_type == ThoughtType.BACKTRACK
        )
        loop_risk = min(1.0, backtrack_count * 0.3)

        # Check for stalling
        if scratchpad.is_stalled(self.stall_window):
            loop_risk = max(loop_risk, 0.7)

        return loop_risk

    def _assess_health(
        self,
        progress_rate: float,
        circular_risk: float,
        confidence: float,
        steps: int,
    ) -> ReasoningHealth:
        """Assess overall reasoning health."""
        if confidence >= self.min_confidence_to_conclude:
            return ReasoningHealth.CONFIDENT

        if steps >= self.max_thinking_steps:
            return ReasoningHealth.EXHAUSTED

        if circular_risk > 0.6:
            return ReasoningHealth.CIRCULAR

        if progress_rate < 0.2:
            return ReasoningHealth.STALLED

        if progress_rate < 0.4:
            return ReasoningHealth.SLOWING

        return ReasoningHealth.PROGRESSING

    def _generate_advice(
        self,
        health: ReasoningHealth,
        progress_rate: float,
        confidence: float,
        steps: int,
        unresolved: List,
    ) -> StrategyAdvice:
        """Generate strategic advice based on health assessment."""
        if health == ReasoningHealth.CONFIDENT:
            return StrategyAdvice.CONCLUDE

        if health == ReasoningHealth.EXHAUSTED:
            return StrategyAdvice.CONCLUDE

        if health == ReasoningHealth.CIRCULAR:
            if self._strategy_switches < 2:
                self._strategy_switches += 1
                return StrategyAdvice.SWITCH_STRATEGY
            return StrategyAdvice.BACKTRACK

        if health == ReasoningHealth.STALLED:
            if len(unresolved) > 1:
                return StrategyAdvice.DECOMPOSE
            if self._strategy_switches < 2:
                self._strategy_switches += 1
                return StrategyAdvice.BROADEN
            return StrategyAdvice.CONCLUDE

        if health == ReasoningHealth.SLOWING:
            if progress_rate < 0.3 and steps > 5:
                return StrategyAdvice.NARROW
            return StrategyAdvice.CONTINUE

        return StrategyAdvice.CONTINUE

    def _estimate_remaining(
        self,
        steps: int,
        progress_rate: float,
        unresolved_count: int,
    ) -> int:
        """Estimate how many more steps are needed."""
        if progress_rate <= 0:
            return self.max_thinking_steps - steps

        remaining_by_progress = int(unresolved_count / max(progress_rate, 0.1))
        remaining_by_budget = self.max_thinking_steps - steps
        return min(remaining_by_progress, remaining_by_budget)

    def _explain(
        self,
        health: ReasoningHealth,
        advice: StrategyAdvice,
        progress_rate: float,
        confidence: float,
        steps: int,
    ) -> str:
        """Generate human-readable explanation."""
        explanations = {
            ReasoningHealth.PROGRESSING: f"Making good progress (rate={progress_rate:.1%}). Continue reasoning.",
            ReasoningHealth.SLOWING: f"Progress slowing (rate={progress_rate:.1%}). Consider narrowing focus.",
            ReasoningHealth.STALLED: f"No progress in recent steps. Consider decomposing or switching strategy.",
            ReasoningHealth.CIRCULAR: "Circular reasoning detected. Backtrack or try different approach.",
            ReasoningHealth.CONFIDENT: f"Sufficient confidence reached ({confidence:.1%}). Ready to conclude.",
            ReasoningHealth.EXHAUSTED: f"Thinking budget exhausted after {steps} steps. Present best findings.",
        }
        return explanations.get(health, f"Health: {health.value}, Advice: {advice.value}")

    def reset(self) -> None:
        """Reset for a new reasoning session."""
        self._progress_history.clear()
        self._strategy_switches = 0


# Aliases
EpistemicState = ReasoningHealth
MetacognitionMonitor = MetaCognition

