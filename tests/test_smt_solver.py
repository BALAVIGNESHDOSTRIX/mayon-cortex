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
Unit Tests for Formal SMT, CSP and Math Constraint Solver
"""

import pytest
from mayon_cortex.reasoning.smt_solver import (
    MathSolver,
    CSPSolver,
    ResolutionProver,
    Literal,
    Clause,
)


class TestMathSolver:
    def setup_method(self):
        self.solver = MathSolver()

    def test_solve_linear_equation(self):
        # 3x + 5 = 20  =>  x = 5.0
        res = self.solver.solve("If 3x + 5 = 20, what is x?")
        assert res.is_solved
        assert "x" in res.solved_variables
        assert pytest.approx(res.solved_variables["x"], 1e-4) == 5.0

    def test_solve_physics_speed_formula(self):
        # speed = distance / time => 100 / 2 = 50
        res = self.solver.solve("Given distance = 100 km and time = 2 hours, calculate speed.")
        assert res.is_solved
        assert "speed" in res.solved_variables
        assert pytest.approx(res.solved_variables["speed"], 1e-4) == 50.0

    def test_solve_financial_discount(self):
        # price = $80, discount = 20% => final_price = 64
        res = self.solver.solve("If price = $80 and discount = 20%, what is the final_price?")
        assert res.is_solved
        assert "final_price" in res.solved_variables
        assert pytest.approx(res.solved_variables["final_price"], 1e-4) == 64.0


class TestCSPSolver:
    def setup_method(self):
        self.solver = CSPSolver()

    def test_solve_map_coloring_csp(self):
        # 3 regions (A, B, C), 2 colors (red, green)
        # A != B, B != C, A != C (Impossible with 2 colors)
        variables = ["A", "B", "C"]
        domains = {
            "A": ["red", "green"],
            "B": ["red", "green"],
            "C": ["red", "green"],
        }
        constraints = [
            ("A", "B", lambda x, y: x != y),
            ("B", "C", lambda x, y: x != y),
            ("A", "C", lambda x, y: x != y),
        ]
        res = self.solver.solve(variables, domains, constraints)
        assert not res.is_satisfiable

        # With 3 colors (red, green, blue) -> should be satisfiable
        domains_3 = {v: ["red", "green", "blue"] for v in variables}
        res_3 = self.solver.solve(variables, domains_3, constraints)
        assert res_3.is_satisfiable
        assert res_3.assignment["A"] != res_3.assignment["B"]
        assert res_3.assignment["B"] != res_3.assignment["C"]
        assert res_3.assignment["A"] != res_3.assignment["C"]


class TestResolutionProver:
    def setup_method(self):
        self.prover = ResolutionProver()

    def test_sound_resolution_proof(self):
        # KB:
        # 1. TakesDrug(John, Lisinopril)
        # 2. NOT TakesDrug(x, Lisinopril) OR TreatsHypertension(x)
        # Goal: Prove TreatsHypertension(John)
        lit1 = Literal(predicate="TakesDrug", args=("John", "Lisinopril"))
        lit2_a = Literal(predicate="TakesDrug", args=("John", "Lisinopril"), is_negated=True)
        lit2_b = Literal(predicate="TreatsHypertension", args=("John",))

        clause1 = Clause({lit1})
        clause2 = Clause({lit2_a, lit2_b})

        goal = Literal(predicate="TreatsHypertension", args=("John",))
        result = self.prover.prove(kb_clauses=[clause1, clause2], query_literal=goal)

        assert result.is_proven
        assert result.contradiction_derived
        assert len(result.proof_steps) >= 1
