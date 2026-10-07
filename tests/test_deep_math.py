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

import pytest
import numpy as np
from mayon_cortex.engine import MayonCortex
from mayon_cortex.reasoning.smt_solver import MathSolver, MathSolution


class TestDeepMath:
    @pytest.fixture
    def solver(self):
        return MathSolver()

    def test_quadratic_real_roots(self, solver):
        # 2x^2 - 4x - 6 = 0 -> roots: 3.0 and -1.0
        res = solver.solve("Solve quadratic 2x^2 - 4x - 6 = 0")
        assert res.is_solved is True
        assert len(res.solved_variables) == 2
        roots = list(res.solved_variables.values())
        assert any(np.isclose(r, 3.0) for r in roots)
        assert any(np.isclose(r, -1.0) for r in roots)
        assert "D = b^2 - 4ac" in res.format_proof()

    def test_quadratic_repeated_root(self, solver):
        # x^2 - 6x + 9 = 0 -> root: 3.0
        res = solver.solve("Solve x^2 - 6x + 9 = 0")
        assert res.is_solved is True
        assert np.isclose(res.solved_variables["x"], 3.0)

    def test_system_of_equations(self, solver):
        # 2x + 3y = 13 and x - y = -1 -> x = 2, y = 3
        res = solver.solve("Solve 2x + 3y = 13 and 1x - 1y = -1")
        assert res.is_solved is True
        assert np.isclose(res.solved_variables["x"], 2.0)
        assert np.isclose(res.solved_variables["y"], 3.0)
        assert "Cramer's Rule" in res.format_proof()

    def test_polynomial_derivative(self, solver):
        # derivative of 3x^3 + 4x^2 - 5x + 7 -> 9x^2 + 8x - 5
        res = solver.solve("Find the derivative of 3x^3 + 4x^2 - 5x + 7 with respect to x")
        assert res.is_solved is True
        assert "9x^2" in res.explanation
        assert "8x" in res.explanation
        assert "- 5" in res.explanation

    def test_polynomial_integral(self, solver):
        # integral of 6x^2 + 4x - 3 dx -> 2x^3 + 2x^2 - 3x + C
        res = solver.solve("Find the integral of 6x^2 + 4x - 3 dx")
        assert res.is_solved is True
        assert "2x^3" in res.explanation
        assert "2x^2" in res.explanation
        assert "3x" in res.explanation
        assert "C" in res.explanation

    def test_array_statistics(self, solver):
        res = solver.solve("Given data = [10, 20, 30, 40, 50], calculate mean and variance")
        assert res.is_solved is True
        assert np.isclose(res.solved_variables["mean"], 30.0)
        assert np.isclose(res.solved_variables["variance"], 200.0)

    def test_combinatorics_and_factorials(self, solver):
        # C(10, 3) = 120
        res = solver.solve("Given n = 10 and r = 3, calculate combinations")
        assert res.is_solved is True
        assert np.isclose(res.solved_variables["combinations"], 120.0)

        # 5! = 120
        res_fact = solver.solve("Given n = 5, calculate factorial")
        assert res_fact.is_solved is True
        assert np.isclose(res_fact.solved_variables["factorial"], 120.0)

    def test_epistemic_uncertainty_missing_variable(self, solver):
        # speed requires distance AND time, but time is missing
        res = solver.solve("Given distance = 100 km, calculate speed.")
        assert res.is_solved is False
        assert "time" in res.explanation
        assert "Insufficient constraints" in res.explanation

    def test_epistemic_uncertainty_unknown_domain(self, solver):
        # Completely unmodeled query
        res = solver.solve("Calculate quantum wavefunction collapse probability for unmodeled tensor omega.")
        assert res.is_solved is False
        assert "Unable to solve" in res.explanation
        assert "knowledge gap" in res.explanation.lower()
