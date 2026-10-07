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
Thought Scratchpad — Active Working Memory for System 2 Reasoning
===================================================================
Phase 1 of the Brain-Like Intelligence upgrade.

The scratchpad is the "inner whiteboard" where the thinking loop records:
  - Bound variables (e.g., subject="Lisinopril", property="treats")
  - Goal stack (what we're trying to answer, sub-goals pushed as needed)
  - Visited states (to detect circular reasoning)
  - Intermediate conclusions (partial answers)
  - Confidence trail (how confident each step is)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ThoughtType(Enum):
    """Types of thoughts recorded in the scratchpad."""
    HYPOTHESIS = "hypothesis"
    OBSERVATION = "observation"
    DEDUCTION = "deduction"
    QUESTION = "question"
    CONCLUSION = "conclusion"
    CONTRADICTION = "contradiction"
    BACKTRACK = "backtrack"


@dataclass
class ThoughtEntry:
    """A single thought recorded during deliberation."""
    step: int
    thought_type: ThoughtType
    content: str
    confidence: float = 0.0
    source: str = ""  # Which module produced this thought
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[Step {self.step} | {self.thought_type.value}] {self.content} (conf={self.confidence:.2f})"


# Aliases
ScratchpadStep = ThoughtEntry



@dataclass
class Goal:
    """A reasoning goal on the goal stack."""
    description: str
    parent_goal: Optional[str] = None
    is_resolved: bool = False
    resolution: str = ""
    confidence: float = 0.0
    sub_goals: List[str] = field(default_factory=list)


class ThoughtScratchpad:
    """
    Active scratchpad for System 2 deliberative reasoning.

    Provides:
      - Variable binding (track named values across steps)
      - Goal stack (push/pop reasoning goals)
      - Thought journal (ordered record of all reasoning steps)
      - Loop detection (identify circular reasoning)
      - State fingerprinting (detect revisited reasoning states)
    """

    def __init__(self, max_thoughts: int = 200, max_goals: int = 20):
        self.max_thoughts = max_thoughts
        self.max_goals = max_goals

        # State
        self._thoughts: List[ThoughtEntry] = []
        self._variables: Dict[str, Any] = {}
        self._goals: List[Goal] = []
        self._visited_states: Set[str] = set()
        self._step_counter: int = 0

    # ── Variable Binding ──

    def bind(self, name: str, value: Any) -> None:
        """Bind a variable to a value (e.g., subject='Lisinopril')."""
        self._variables[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a bound variable."""
        return self._variables.get(name, default)

    def unbind(self, name: str) -> None:
        """Remove a variable binding."""
        self._variables.pop(name, None)

    @property
    def variables(self) -> Dict[str, Any]:
        """All currently bound variables."""
        return dict(self._variables)

    # ── Goal Stack ──

    def push_goal(self, description: str, parent: Optional[str] = None) -> Goal:
        """Push a new reasoning goal onto the stack."""
        goal = Goal(description=description, parent_goal=parent)
        if len(self._goals) < self.max_goals:
            self._goals.append(goal)
        return goal

    def resolve_goal(self, description: str, resolution: str, confidence: float = 1.0) -> bool:
        """Mark a goal as resolved with a result."""
        for goal in reversed(self._goals):
            if goal.description == description and not goal.is_resolved:
                goal.is_resolved = True
                goal.resolution = resolution
                goal.confidence = confidence
                return True
        return False

    def current_goal(self) -> Optional[Goal]:
        """Get the current active (unresolved) goal."""
        for goal in reversed(self._goals):
            if not goal.is_resolved:
                return goal
        return None

    @property
    def unresolved_goals(self) -> List[Goal]:
        """All unresolved goals."""
        return [g for g in self._goals if not g.is_resolved]

    @property
    def all_goals(self) -> List[Goal]:
        return list(self._goals)

    # ── Thought Journal ──

    def record(
        self,
        thought_type: ThoughtType,
        content: str,
        confidence: float = 0.0,
        source: str = "",
        variables: Optional[Dict[str, Any]] = None,
    ) -> ThoughtEntry:
        """Record a thought in the journal."""
        self._step_counter += 1
        entry = ThoughtEntry(
            step=self._step_counter,
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            source=source,
            variables=variables or {},
        )
        if len(self._thoughts) < self.max_thoughts:
            self._thoughts.append(entry)
        return entry

    @property
    def thoughts(self) -> List[ThoughtEntry]:
        """All recorded thoughts."""
        return list(self._thoughts)

    @property
    def conclusions(self) -> List[ThoughtEntry]:
        """All recorded conclusions."""
        return [t for t in self._thoughts if t.thought_type == ThoughtType.CONCLUSION]

    @property
    def contradictions(self) -> List[ThoughtEntry]:
        """All detected contradictions."""
        return [t for t in self._thoughts if t.thought_type == ThoughtType.CONTRADICTION]

    @property
    def step_count(self) -> int:
        return self._step_counter

    # ── Loop Detection ──

    def _state_fingerprint(self) -> str:
        """Create a hashable fingerprint of current reasoning state."""
        goal_str = "|".join(
            g.description for g in self._goals if not g.is_resolved
        )
        var_str = "|".join(f"{k}={v}" for k, v in sorted(self._variables.items()))
        last_thoughts = "|".join(
            t.content[:50] for t in self._thoughts[-3:]
        )
        return f"{goal_str}::{var_str}::{last_thoughts}"

    def check_loop(self) -> bool:
        """
        Check if the current reasoning state has been visited before.
        Returns True if a loop is detected.
        """
        fp = self._state_fingerprint()
        if fp in self._visited_states:
            self.record(
                ThoughtType.BACKTRACK,
                f"Loop detected: revisiting state '{fp[:80]}...'",
                confidence=0.0,
                source="loop_detector",
            )
            return True
        self._visited_states.add(fp)
        return False

    def is_stalled(self, window: int = 5) -> bool:
        """
        Check if reasoning is stalled (no progress in recent steps).
        Stalled = last N thoughts have the same type and content prefix.
        """
        if len(self._thoughts) < window:
            return False
        recent = self._thoughts[-window:]
        types_same = len(set(t.thought_type for t in recent)) == 1
        content_prefixes = set(t.content[:30] for t in recent)
        return types_same and len(content_prefixes) <= 2

    # ── Summary ──

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the current scratchpad state."""
        return {
            "steps": self._step_counter,
            "total_thoughts": len(self._thoughts),
            "conclusions": len(self.conclusions),
            "contradictions": len(self.contradictions),
            "unresolved_goals": len(self.unresolved_goals),
            "bound_variables": len(self._variables),
            "loops_detected": sum(
                1 for t in self._thoughts if t.thought_type == ThoughtType.BACKTRACK
            ),
        }

    def reasoning_trace(self) -> str:
        """Human-readable trace of all reasoning steps."""
        lines = []
        for t in self._thoughts:
            lines.append(str(t))
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear the scratchpad for a new reasoning session."""
        self._thoughts.clear()
        self._variables.clear()
        self._goals.clear()
        self._visited_states.clear()
        self._step_counter = 0
