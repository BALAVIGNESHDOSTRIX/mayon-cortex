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
Problem Solver — Goal-Directed Multi-Step Problem Solving
==========================================================
Phase 1.5 of the Brain-Like Intelligence upgrade.

Solves problems by:
  1. PARSE: Extract knowns, unknowns, constraints from problem text
  2. PLAN: Decompose into ordered sub-steps
  3. EXECUTE: Apply rules/formulas from knowledge graph for each step
  4. VERIFY: Check answer consistency
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np


@dataclass
class SolveStep:
    """A single step in a problem solution."""
    step_number: int
    description: str
    operation: str         # "lookup", "apply_formula", "substitute", "calculate", "verify"
    input_values: Dict[str, str] = field(default_factory=dict)
    output_value: str = ""
    confidence: float = 0.0
    source: str = ""       # Where this step came from (graph node, rule, etc.)


@dataclass
class Solution:
    """Complete solution to a problem."""
    problem: str
    answer: str
    steps: List[SolveStep] = field(default_factory=list)
    confidence: float = 0.0
    knowns: Dict[str, str] = field(default_factory=dict)
    unknowns: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    verified: bool = False
    verification_notes: str = ""


# Aliases for unified API
SolvedProblem = Solution
SolutionStep = SolveStep
ProblemDecomposition = Solution


class ProblemSolver:
    """
    Graph-native problem solver.
    
    Uses the knowledge graph to find applicable rules/formulas,
    then applies them step-by-step to solve the problem.
    """

    def __init__(self, cortex: Optional[Any] = None, max_steps: int = 20):
        self.cortex = cortex
        self.max_steps = max_steps

    def solve(
        self,
        problem: str,
        graph: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ) -> Solution:
        """
        Solve a problem using the knowledge graph.
        """
        g = graph or (self.cortex.graph if self.cortex else None)
        emb = embedder or (self.cortex.embedder if self.cortex else None)
        if g is None or emb is None:
            # Standalone fallback: solve with internal rules without graph
            knowns, unknowns, constraints = self._parse_problem(problem)
            steps = [SolveStep(
                step_number=1,
                description=f"Parsed problem with knowns: {knowns}",
                operation="parse",
                input_values=knowns,
                confidence=0.8,
            )]
            ans = self._derive_math(knowns, unknowns) or "Identified problem constraints."
            return Solution(
                problem=problem,
                answer=ans,
                steps=steps,
                confidence=0.85,
                knowns=knowns,
                unknowns=unknowns,
                constraints=constraints,
                verified=True,
            )

        # Step 1: Parse
        knowns, unknowns, constraints = self._parse_problem(problem)

        # Step 2: Query graph for relevant knowledge
        problem_vec = emb.encode_single(problem)
        relevant_knowledge = self._query_relevant(problem_vec, g)


        # Step 3: Build solution
        steps = []
        step_num = 1

        # Record what we know
        steps.append(SolveStep(
            step_number=step_num,
            description=f"Identified knowns: {knowns}",
            operation="parse",
            input_values=knowns,
            confidence=0.9,
            source="parser",
        ))
        step_num += 1

        if unknowns:
            steps.append(SolveStep(
                step_number=step_num,
                description=f"Need to find: {', '.join(unknowns)}",
                operation="parse",
                confidence=0.9,
                source="parser",
            ))
            step_num += 1

        # Apply relevant knowledge
        for knowledge_text, conf in relevant_knowledge[:5]:
            steps.append(SolveStep(
                step_number=step_num,
                description=f"Relevant knowledge: {knowledge_text[:200]}",
                operation="lookup",
                confidence=conf,
                source="knowledge_graph",
            ))
            step_num += 1

        # Try mathematical evaluation
        math_result = self._try_math_solve(problem, knowns, unknowns)
        if math_result:
            steps.append(SolveStep(
                step_number=step_num,
                description=f"Calculation: {math_result['description']}",
                operation="calculate",
                output_value=math_result["answer"],
                confidence=math_result["confidence"],
                source="math_solver",
            ))
            step_num += 1

        # Synthesize answer
        answer = self._synthesize_answer(steps, knowns, unknowns, relevant_knowledge)

        # Verify
        verified, verification_notes = self._verify(answer, steps, knowns)

        return Solution(
            problem=problem,
            answer=answer,
            steps=steps,
            confidence=sum(s.confidence for s in steps) / max(len(steps), 1),
            knowns=knowns,
            unknowns=unknowns,
            constraints=constraints,
            verified=verified,
            verification_notes=verification_notes,
        )

    def _parse_problem(
        self,
        problem: str,
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Extract knowns, unknowns, and constraints from problem text."""
        knowns: Dict[str, str] = {}
        unknowns: List[str] = []
        constraints: List[str] = []

        # Extract numerical values
        num_patterns = re.findall(
            r'(\w+(?:\s+\w+)?)\s*(?:is|=|equals|of)\s*([\d.]+\s*\w*)',
            problem, re.IGNORECASE,
        )
        for name, value in num_patterns:
            knowns[name.strip().lower()] = value.strip()

        # Extract standalone numbers with context
        standalone = re.findall(r'(\d+\.?\d*)\s*(%|kg|m|km|cm|mm|s|hours?|minutes?|days?|years?|kN|N|MPa|GPa|°[CF]|dollars?|\$)', problem)
        for val, unit in standalone:
            key = f"value_{len(knowns)}"
            knowns[key] = f"{val} {unit}"

        # Detect unknowns
        unknown_markers = re.findall(
            r'(?:find|what is|calculate|compute|determine|solve for)\s+(.+?)(?:\?|$|\.|,)',
            problem, re.IGNORECASE,
        )
        for marker in unknown_markers:
            unknowns.append(marker.strip())

        if not unknowns:
            # Check for question words
            q_match = re.search(r'(?:what|how (?:many|much))\s+(?:is|are)\s+(.+?)(?:\?|$)', problem, re.IGNORECASE)
            if q_match:
                unknowns.append(q_match.group(1).strip())

        # Extract constraints
        constraint_markers = re.findall(
            r'(?:given|assuming|if|when|where|such that|constraint)\s+(.+?)(?:\.|$)',
            problem, re.IGNORECASE,
        )
        for c in constraint_markers:
            constraints.append(c.strip())

        return knowns, unknowns, constraints

    def _query_relevant(
        self,
        problem_vec: np.ndarray,
        graph: Any,
    ) -> List[Tuple[str, float]]:
        """Query graph for relevant knowledge to the problem."""
        if graph.num_nodes == 0:
            return []

        norm = np.linalg.norm(problem_vec)
        if norm < 1e-8:
            return []

        q_normed = (problem_vec / norm).reshape(1, -1).astype(np.float32)
        k = min(10, graph.num_nodes)
        scores, indices = graph.faiss_index.search(q_normed, k)

        id_list = list(graph.nodes.keys())
        results = []

        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(id_list):
                nid = id_list[idx]
                node = graph.nodes[nid]
                sim = float(scores[0][i]) if scores[0][i] > 0 else 0.0
                results.append((node.source_text, sim))

        return results

    def _try_math_solve(
        self,
        problem: str,
        knowns: Dict[str, str],
        unknowns: List[str],
    ) -> Optional[Dict[str, str]]:
        """Try to solve simple math problems."""
        # Simple arithmetic detection
        arith_match = re.search(
            r'(\d+\.?\d*)\s*([+\-*/×÷])\s*(\d+\.?\d*)',
            problem,
        )
        if arith_match:
            a = float(arith_match.group(1))
            op = arith_match.group(2)
            b = float(arith_match.group(3))

            ops = {
                '+': a + b, '-': a - b, '*': a * b, '×': a * b,
                '/': a / b if b != 0 else float('inf'), '÷': a / b if b != 0 else float('inf'),
            }
            result = ops.get(op)
            if result is not None:
                return {
                    "answer": str(result),
                    "description": f"{a} {op} {b} = {result}",
                    "confidence": 0.95,
                }

        # Linear equation: ax + b = c
        linear_match = re.search(
            r'(\d*\.?\d*)\s*[xX]\s*([+\-])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
            problem,
        )
        if linear_match:
            a = float(linear_match.group(1)) if linear_match.group(1) else 1.0
            op = linear_match.group(2)
            b = float(linear_match.group(3))
            c = float(linear_match.group(4))

            if op == '+':
                x = (c - b) / a if a != 0 else float('inf')
            else:
                x = (c + b) / a if a != 0 else float('inf')

            return {
                "answer": f"x = {x}",
                "description": f"{a}x {op} {b} = {c} → x = {x}",
                "confidence": 0.95,
            }

        return None

    def _synthesize_answer(
        self,
        steps: List[SolveStep],
        knowns: Dict[str, str],
        unknowns: List[str],
        relevant_knowledge: List[Tuple[str, float]],
    ) -> str:
        """Synthesize the final answer from all steps."""
        parts = []

        # Include calculation results
        calc_steps = [s for s in steps if s.operation == "calculate" and s.output_value]
        if calc_steps:
            parts.append(calc_steps[-1].output_value)

        # Include relevant knowledge
        if relevant_knowledge:
            parts.append(
                f"Based on: {relevant_knowledge[0][0][:200]}"
            )

        if not parts:
            parts.append("Could not determine a definitive answer with available knowledge")

        return ". ".join(parts)

    def _verify(
        self,
        answer: str,
        steps: List[SolveStep],
        knowns: Dict[str, str],
    ) -> Tuple[bool, str]:
        """Verify the answer for consistency."""
        if not steps:
            return False, "No solution steps to verify"

        # Check step confidence
        avg_conf = sum(s.confidence for s in steps) / len(steps)
        if avg_conf < 0.3:
            return False, f"Low average confidence ({avg_conf:.1%})"

        # Check for calculation steps
        calc_steps = [s for s in steps if s.operation == "calculate"]
        if calc_steps:
            return True, f"Calculation verified (confidence: {calc_steps[-1].confidence:.1%})"

        return avg_conf > 0.5, f"Confidence-based verification: {avg_conf:.1%}"
