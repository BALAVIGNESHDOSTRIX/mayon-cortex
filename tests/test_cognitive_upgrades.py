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

"""Unit tests for Autoregressive Text, Symbolic Algebra, Number Theory, Calculus & Puzzle Solvers."""

import numpy as np
import pytest

from mayon_cortex.language.autoregressive_engine import GraphAutoRegressiveEngine, TextGenerationConfig
from mayon_cortex.language.word_graph import WordGraph
from mayon_cortex.reasoning.symbolic_algebra import UniversalEquationSolver, Polynomial, ExprParser, Const, Var, Add, Mul
from mayon_cortex.reasoning.number_theory import NumberTheoryEngine
from mayon_cortex.reasoning.matrix_engine import MatrixEngine
from mayon_cortex.reasoning.calculus_engine import CalculusEngine
from mayon_cortex.reasoning.puzzle_solver import SudokuSolver, CryptarithmSolver, NQueensSolver, MagicSquareEngine


def test_autoregressive_text_engine():
    wg = WordGraph()
    for sentence in ["the cat sat on the mat", "the quick dog jumped over the fence"]:
        wg.add_sentence_tokens(sentence.split())
    engine = GraphAutoRegressiveEngine(word_graph=wg)

    text = engine.generate("the cat", max_tokens=10, temperature=0.7)
    assert len(text) > 0
    assert isinstance(text, str)


def test_symbolic_algebra_and_equations():
    # 1. AST Parser and Evaluation
    ast = ExprParser("2 * x + 3 * x^2").parse()
    val = ast.eval({"x": 2.0})
    assert val == 2 * 2 + 3 * (2 ** 2)

    # 2. Linear Equation
    res_lin = UniversalEquationSolver.solve("2*x - 10 = 0", var="x")
    assert np.isclose(res_lin["real_roots"][0], 5.0, atol=1e-3)

    # 3. Quadratic Equation
    res_quad = UniversalEquationSolver.solve("x^2 - 9 = 0", var="x")
    roots = sorted(res_quad["real_roots"])
    assert np.isclose(roots[0], -3.0, atol=1e-3)
    assert np.isclose(roots[1], 3.0, atol=1e-3)


def test_number_theory():
    # Primality
    assert NumberTheoryEngine.is_prime(2)
    assert NumberTheoryEngine.is_prime(17)
    assert NumberTheoryEngine.is_prime(97)
    assert not NumberTheoryEngine.is_prime(100)

    # Factorization
    factors = NumberTheoryEngine.factorize(360)
    assert factors == {2: 3, 3: 2, 5: 1}

    # Modular Inverse
    inv = NumberTheoryEngine.mod_inverse(3, 11)
    assert inv == 4  # (3 * 4) % 11 == 1

    # Chinese Remainder Theorem: x = 2 mod 3, x = 3 mod 5, x = 2 mod 7 -> x = 23
    crt_res = NumberTheoryEngine.chinese_remainder([2, 3, 2], [3, 5, 7])
    assert crt_res == 23

    # Euler Totient
    assert NumberTheoryEngine.euler_totient(9) == 6  # 1,2,4,5,7,8


def test_matrix_engine():
    # Linear System: 2x + y = 5, x + 3y = 10 -> x = 1, y = 3
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 10], dtype=float)

    res = MatrixEngine.solve_linear_system(A, b)
    assert res["status"] == "UNIQUE"
    assert np.allclose(res["solution"], [1.0, 3.0], atol=1e-4)

    # Determinant
    det = MatrixEngine.determinant(A)
    assert np.isclose(det, 5.0, atol=1e-4)


def test_calculus_engine():
    # Derivative d/dx(x^3 + 2*x) = 3*x^2 + 2
    ast = ExprParser("x^3 + 2*x").parse()
    diff = ast.diff("x").simplify()
    val_at_2 = diff.eval({"x": 2.0})
    assert val_at_2 == 3 * (2 ** 2) + 2

    # Definite integral \int_0^2 x^2 dx = 8/3 ~ 2.6667
    val_int = CalculusEngine.definite_integral(lambda x: x**2, 0.0, 2.0)
    assert np.isclose(val_int, 8.0 / 3.0, atol=1e-3)


def test_puzzle_solvers():
    # Sudoku
    grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    sol = SudokuSolver.solve(grid)
    assert sol is not None
    assert sol[0][2] == 4

    # Cryptarithm: SEND + MORE = MONEY
    sol_crypt = CryptarithmSolver.solve("SEND + MORE = MONEY")
    assert sol_crypt is not None
    assert sol_crypt["M"] == 1
    assert sol_crypt["O"] == 0

    # N-Queens for N=4
    queens = NQueensSolver.solve(4)
    assert len(queens) == 2

    # Magic Square (3x3)
    ms = MagicSquareEngine.generate(3)
    assert MagicSquareEngine.verify(ms)
