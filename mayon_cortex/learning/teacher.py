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
Teacher — Curriculum Learning System & Real Assessment
======================================================
Teaches the brain progressively through 4 developmental phases:
  Phase 1 (Infant):  Basic categories & properties (nature, animals, living things)
  Phase 2 (Toddler): Real-world domains (stories, science, technology, society)
  Phase 3 (Child):   Cross-domain & relational links (science ↔ society, cause ↔ effect)
  Phase 4 (Adult):   Decision making & logical multi-step reasoning

Assessment Fixes (v4):
- Live evaluation: `assess(live_eval=True)` queries the graph live on all test sets.
- Real Proof Reinforcement: extracts activated proof node IDs and strengthens those pathways.
- Learning Velocity: tracks progress rates over iterations.
- Real-World General Data: grounded in everyday reality, stories, and science.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Lesson:
    """A single teaching unit."""
    id: str
    phase: int
    sector: str
    facts: List[str]              # sentences to ingest
    tests: List[Dict[str, str]]   # [{"question": ..., "expected_contains": ...}]
    metadata: Dict = field(default_factory=dict)


@dataclass
class LessonResult:
    """Result of a lesson attempt."""
    lesson_id: str
    score: float           # 0.0 - 1.0
    passed: bool
    test_results: List[bool]
    attempt: int


@dataclass
class LearningProgress:
    """Tracks what the brain has learned."""
    learned: Dict[str, float] = field(default_factory=dict)    # lesson_id → score
    attempts: Dict[str, int] = field(default_factory=dict)     # lesson_id → attempt count
    phase_scores: Dict[int, float] = field(default_factory=dict)  # phase → avg score
    history: List[Dict[str, Any]] = field(default_factory=list)   # timestamped evaluation history

    def mark_learned(self, lesson_id: str, score: float):
        self.learned[lesson_id] = score

    def is_learned(self, lesson_id: str, threshold: float = 0.75) -> bool:
        return self.learned.get(lesson_id, 0.0) >= threshold

    def increment_attempt(self, lesson_id: str):
        self.attempts[lesson_id] = self.attempts.get(lesson_id, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learned": self.learned,
            "attempts": self.attempts,
            "phase_scores": self.phase_scores,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningProgress":
        lp = cls()
        lp.learned = d.get("learned", {})
        lp.attempts = d.get("attempts", {})
        lp.phase_scores = d.get("phase_scores", {})
        return lp


class Teacher:
    """
    Curriculum learning & active assessment system.
    """

    def __init__(self, brain):
        self.brain = brain
        self.progress = LearningProgress()
        self._curriculum = self._build_curriculum()

    def teach_lesson(self, lesson: Lesson) -> LessonResult:
        """
        One teaching cycle: Teach → Test → Reinforce/Reteach.
        """
        self.progress.increment_attempt(lesson.id)
        attempt = self.progress.attempts[lesson.id]

        # 1. Teach — ingest facts
        for fact_text in lesson.facts:
            self.brain.ingest(fact_text, sector=lesson.sector)

        # 2. Test — query and verify answers
        test_results = []
        reinforced_node_ids = set()

        for test in lesson.tests:
            question = test["question"]
            expected = test.get("expected_contains", "").lower()

            response = self.brain.query(question)
            answer_text = response.text.lower() if hasattr(response, "text") else str(response).lower()

            passed = (expected in answer_text) if expected else (len(answer_text) > 8)
            test_results.append(passed)

            # Collect real proof node IDs for reinforcement
            if passed and hasattr(response, "proof_paths"):
                for sp in response.proof_paths:
                    path = sp.path if hasattr(sp, "path") else sp
                    if hasattr(path, "node_ids"):
                        reinforced_node_ids.update(path.node_ids)

        # 3. Compute score
        score = sum(test_results) / max(len(test_results), 1)

        # 4. Reinforce or record failure
        threshold = getattr(self.brain.config, "teacher_pass_threshold", 0.75)
        if score >= threshold:
            self.progress.mark_learned(lesson.id, score)
            # Direct Hebbian reinforcement of exact proof nodes
            if hasattr(self.brain, "learner") and reinforced_node_ids:
                self.brain.learner.reinforce(
                    list(reinforced_node_ids),
                    self.brain.graph,
                    boost=getattr(self.brain.config, "teacher_reinforcement_boost", 0.2),
                )

        return LessonResult(
            lesson_id=lesson.id,
            score=score,
            passed=score >= threshold,
            test_results=test_results,
            attempt=attempt,
        )

    def run_curriculum(self, phase: Optional[int] = None):
        """Run full or phase-specific curriculum with automatic reteaching."""
        max_reteach = getattr(self.brain.config, "teacher_max_reteach", 3)
        lessons = self._curriculum
        if phase is not None:
            lessons = [l for l in lessons if l.phase == phase]

        for lesson in lessons:
            if self.progress.is_learned(lesson.id):
                continue

            result = self.teach_lesson(lesson)
            attempts = 0
            while not result.passed and attempts < max_reteach:
                result = self.teach_lesson(lesson)
                attempts += 1

    def assess(self, live_eval: bool = True) -> Dict[str, Any]:
        """
        Comprehensive brain assessment with live test execution.
        """
        total_lessons = len(self._curriculum)
        total_tests = sum(len(l.tests) for l in self._curriculum)
        passed_tests = 0
        live_scores_per_lesson: Dict[str, float] = {}

        if live_eval:
            # Execute all tests live against the active graph
            for lesson in self._curriculum:
                lesson_passed = 0
                for test in lesson.tests:
                    q = test["question"]
                    exp = test.get("expected_contains", "").lower()
                    resp = self.brain.query(q)
                    ans = resp.text.lower() if hasattr(resp, "text") else str(resp).lower()
                    if exp in ans or (not exp and len(ans) > 8):
                        lesson_passed += 1
                        passed_tests += 1

                l_score = lesson_passed / max(len(lesson.tests), 1)
                live_scores_per_lesson[lesson.id] = l_score
                self.progress.mark_learned(lesson.id, l_score)

        learned_count = sum(1 for l in self._curriculum if self.progress.is_learned(l.id))
        overall_acc = (passed_tests / max(total_tests, 1)) if live_eval else (learned_count / max(total_lessons, 1))

        report = {
            "total_lessons": total_lessons,
            "total_tests": total_tests,
            "learned_lessons": learned_count,
            "overall_accuracy": overall_acc,
            "sector_accuracy": {},
            "phases": {},
            "weaknesses": [],
            "strengths": [],
            "recommendations": [],
        }

        # Sector metrics
        sectors = set(l.sector for l in self._curriculum)
        for sec in sectors:
            sec_lessons = [l for l in self._curriculum if l.sector == sec]
            sec_learned = sum(1 for l in sec_lessons if self.progress.is_learned(l.id))
            report["sector_accuracy"][sec] = sec_learned / max(len(sec_lessons), 1)

        # Phase metrics
        for phase in [1, 2, 3, 4]:
            phase_lessons = [l for l in self._curriculum if l.phase == phase]
            learned = sum(1 for l in phase_lessons if self.progress.is_learned(l.id))
            total = len(phase_lessons)
            score = learned / max(total, 1)

            report["phases"][phase] = {
                "name": {1: "Infant", 2: "Toddler", 3: "Child", 4: "Adult"}[phase],
                "learned": learned,
                "total": total,
                "score": score,
            }

            phase_name = report["phases"][phase]["name"]
            if score < 0.6:
                report["weaknesses"].append(f"Phase {phase} ({phase_name}): {score:.0%}")
                report["recommendations"].append(f"Reinforce Phase {phase} ({phase_name}) curriculum")
            elif score >= 0.8:
                report["strengths"].append(f"Phase {phase} ({phase_name}): {score:.0%}")

        return report

    def _build_curriculum(self) -> List[Lesson]:
        """Real-world curriculum spanning nature, science, stories, and reasoning."""
        lessons = []

        # ── Phase 1: Infant — Nature & Basic Categories ──
        lessons.extend([
            Lesson(
                id="p1_nature_animals",
                phase=1, sector="general",
                facts=[
                    "A dog is a type of animal.",
                    "A cat is a type of animal.",
                    "An animal is a living thing.",
                    "A bird is a type of animal.",
                    "A tree is a type of plant.",
                ],
                tests=[
                    {"question": "What is a dog?", "expected_contains": "animal"},
                    {"question": "Is an animal a living thing?", "expected_contains": "living"},
                ],
            ),
            Lesson(
                id="p1_physical_world",
                phase=1, sector="general",
                facts=[
                    "The sun is a star.",
                    "Earth is a planet.",
                    "Earth orbits the sun.",
                    "Water boils at 100 degrees.",
                    "Ice melts at 0 degrees.",
                ],
                tests=[
                    {"question": "What is the sun?", "expected_contains": "star"},
                    {"question": "What does Earth orbit around?", "expected_contains": "sun"},
                ],
            ),
            Lesson(
                id="p1_objects_parts",
                phase=1, sector="general",
                facts=[
                    "A wheel is part of a car.",
                    "An engine is part of a car.",
                    "A leaf is part of a tree.",
                    "A root is part of a tree.",
                ],
                tests=[
                    {"question": "What is part of a car?", "expected_contains": "wheel"},
                    {"question": "What is part of a tree?", "expected_contains": "leaf"},
                ],
            ),
        ])

        # ── Phase 2: Toddler — Stories, Science & Real Worlds ──
        lessons.extend([
            Lesson(
                id="p2_story_adventure",
                phase=2, sector="story",
                facts=[
                    "A brave hero named Arthur embarked on a quest to defeat the dragon.",
                    "The dragon guarded a legendary sword in the ancient kingdom.",
                    "Arthur protected the kingdom from danger.",
                ],
                tests=[
                    {"question": "Who is Arthur?", "expected_contains": "hero"},
                    {"question": "What did Arthur protect?", "expected_contains": "kingdom"},
                ],
            ),
            Lesson(
                id="p2_science_foundations",
                phase=2, sector="science",
                facts=[
                    "Photosynthesis produces oxygen and energy for plants.",
                    "Gravity attracts mass towards the center of Earth.",
                    "The cell is the basic unit of life.",
                    "DNA contains genetic code for living organisms.",
                ],
                tests=[
                    {"question": "What does photosynthesis produce?", "expected_contains": "oxygen"},
                    {"question": "What does gravity attract?", "expected_contains": "mass"},
                ],
            ),
            Lesson(
                id="p2_technology",
                phase=2, sector="code",
                facts=[
                    "A function in Python is defined with the def keyword.",
                    "A dictionary stores key value pairs.",
                    "An algorithm is a step by step procedure for solving a problem.",
                ],
                tests=[
                    {"question": "How is a function defined in Python?", "expected_contains": "def"},
                    {"question": "What is an algorithm?", "expected_contains": "procedure"},
                ],
            ),
            Lesson(
                id="p2_society_rules",
                phase=2, sector="legal",
                facts=[
                    "A law regulates societal behavior for public safety.",
                    "A contract is a legally binding agreement between parties.",
                    "Compliance requires following established regulations.",
                ],
                tests=[
                    {"question": "What does a law regulate?", "expected_contains": "safety"},
                ],
            ),
        ])

        # ── Phase 3: Child — Cross-Domain Causality & Analogies ──
        lessons.extend([
            Lesson(
                id="p3_science_environment",
                phase=3, sector="science",
                facts=[
                    "Excess carbon emissions cause global temperature increases.",
                    "Renewable energy reduces greenhouse emissions.",
                    "Forests absorb carbon dioxide through photosynthesis.",
                ],
                tests=[
                    {"question": "What reduces greenhouse emissions?", "expected_contains": "renewable"},
                    {"question": "What absorbs carbon dioxide?", "expected_contains": "forest"},
                ],
            ),
            Lesson(
                id="p3_emotion_society",
                phase=3, sector="general",
                facts=[
                    "Empathy builds trust in human relationships.",
                    "Kindness fosters community harmony.",
                    "Grief requires compassion and patience.",
                ],
                tests=[
                    {"question": "What does empathy build?", "expected_contains": "trust"},
                ],
            ),
        ])

        # ── Phase 4: Adult — High-Order Decision Making ──
        lessons.extend([
            Lesson(
                id="p4_safety_first",
                phase=4, sector="general",
                facts=[
                    "When solving problems, ensure safety first before optimizing performance.",
                    "Ethical considerations take priority over rapid execution.",
                    "Verification and testing must precede final release.",
                ],
                tests=[
                    {"question": "What should be ensured first in problem solving?", "expected_contains": "safety"},
                ],
            ),
            Lesson(
                id="p4_structured_action",
                phase=4, sector="general",
                facts=[
                    "Define goals clearly before allocating resources.",
                    "Evaluate potential risks and failure modes in advance.",
                    "Monitor ongoing progress and adjust strategy accordingly.",
                ],
                tests=[
                    {"question": "What must be defined before allocating resources?", "expected_contains": "goals"},
                ],
            ),
        ])

        return lessons
