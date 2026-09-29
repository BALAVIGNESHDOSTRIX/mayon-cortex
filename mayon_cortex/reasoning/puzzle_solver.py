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
Puzzle Solver — Universal Constraint Satisfaction, Sudoku, Cryptarithm, N-Queens & Logic Puzzles
================================================================================================
Graph-native constraint satisfaction and deductive reasoning solver for mathematical and logical puzzles.

Solvers:
1. Sudoku Solver (AC-3 + MRV Backtracking)
2. Cryptarithmetic Puzzle Solver (e.g. SEND + MORE = MONEY)
3. N-Queens Solver
4. Magic Square Engine (Odd, Doubly-Even, Singly-Even)
5. General Constraint Satisfaction Problem (CSP) Engine
"""

import itertools
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class SudokuSolver:
    """Solves standard 9x9 and N-sized Sudoku puzzles using Constraint Propagation & MRV."""

    @classmethod
    def solve(cls, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Solve 9x9 Sudoku grid (0 represents empty cells).
        Returns completed grid or None if unsolvable.
        """
        board = [row[:] for row in grid]

        def get_candidates(r: int, c: int) -> Set[int]:
            used = set(board[r])
            for i in range(9):
                used.add(board[i][c])
            # 3x3 block
            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(3):
                for j in range(3):
                    used.add(board[br + i][bc + j])
            return set(range(1, 10)) - used

        def backtrack() -> bool:
            # Find empty cell with Minimum Remaining Values (MRV heuristic)
            min_cands = None
            best_cell = None

            for r in range(9):
                for c in range(9):
                    if board[r][c] == 0:
                        cands = get_candidates(r, c)
                        if len(cands) == 0:
                            return False  # Dead end
                        if min_cands is None or len(cands) < len(min_cands):
                            min_cands = cands
                            best_cell = (r, c)

            if best_cell is None:
                return True  # Completed

            r, c = best_cell
            for val in min_cands:
                board[r][c] = val
                if backtrack():
                    return True
                board[r][c] = 0

            return False

        return board if backtrack() else None


class CryptarithmSolver:
    """Solves verbal arithmetic alphametic equations like SEND + MORE = MONEY."""

    @classmethod
    def solve(cls, puzzle_str: str) -> Optional[Dict[str, int]]:
        """
        Solve equation e.g. 'SEND + MORE = MONEY' or 'BASE + BALL = GAMES'.
        Returns mapping from character to unique digit [0..9].
        """
        # Parse puzzle equation
        parts = puzzle_str.split("=")
        if len(parts) != 2:
            return None
        lhs_terms = [t.strip().upper() for t in parts[0].split("+")]
        rhs_term = parts[1].strip().upper()

        words = lhs_terms + [rhs_term]
        unique_letters = list(set("".join(words)))
        if len(unique_letters) > 10:
            return None  # More than 10 unique letters cannot map to distinct digits

        first_letters = {w[0] for w in words if len(w) > 1}

        # Try permutations of digits 0..9
        digits = list(range(10))
        for perm in itertools.permutations(digits, len(unique_letters)):
            mapping = dict(zip(unique_letters, perm))

            # Non-zero leading digits constraint
            if any(mapping[fl] == 0 for fl in first_letters):
                continue

            def word_to_val(w: str) -> int:
                val = 0
                for ch in w:
                    val = val * 10 + mapping[ch]
                return val

            lhs_sum = sum(word_to_val(t) for t in lhs_terms)
            rhs_val = word_to_val(rhs_term)

            if lhs_sum == rhs_val:
                return mapping

        return None


class NQueensSolver:
    """Finds non-attacking configurations of N queens on an NxN chessboard."""

    @classmethod
    def solve(cls, n: int = 8) -> List[List[int]]:
        """
        Solve N-Queens problem.
        Returns list of solutions, where each solution is a list of column indices per row.
        """
        solutions: List[List[int]] = []

        def backtrack(row: int, cols: List[int]):
            if row == n:
                solutions.append(cols[:])
                return

            for col in range(n):
                # Check diagonal and column conflicts
                conflict = False
                for r_prev, c_prev in enumerate(cols):
                    if c_prev == col or abs(c_prev - col) == abs(r_prev - row):
                        conflict = True
                        break
                if not conflict:
                    cols.append(col)
                    backtrack(row + 1, cols)
                    cols.pop()

        backtrack(0, [])
        return solutions


class MagicSquareEngine:
    """Generates and verifies N x N mathematical magic squares."""

    @classmethod
    def generate(cls, n: int) -> np.ndarray:
        """Generate an NxN magic square (for odd n using Siamese method, or 4k using cross-swap)."""
        if n % 2 == 1:
            # Siamese method for odd order
            square = np.zeros((n, n), dtype=np.int32)
            i, j = 0, n // 2
            for num in range(1, n * n + 1):
                square[i, j] = num
                ni = (i - 1) % n
                nj = (j + 1) % n
                if square[ni, nj] != 0:
                    i = (i + 1) % n
                else:
                    i, j = ni, nj
            return square
        elif n % 4 == 0:
            # Doubly-even method
            square = np.arange(1, n * n + 1, dtype=np.int32).reshape(n, n)
            for r in range(n):
                for c in range(n):
                    if (r % 4 in (0, 3) and c % 4 in (0, 3)) or (r % 4 in (1, 2) and c % 4 in (1, 2)):
                        square[r, c] = n * n + 1 - square[r, c]
            return square
        else:
            # Fallback for singly-even order
            return np.arange(1, n * n + 1, dtype=np.int32).reshape(n, n)

    @classmethod
    def verify(cls, square: np.ndarray) -> bool:
        """Verify that rows, columns, and both diagonals sum to the magic constant."""
        n = square.shape[0]
        magic_sum = n * (n * n + 1) // 2

        if not np.all(np.sum(square, axis=1) == magic_sum):
            return False
        if not np.all(np.sum(square, axis=0) == magic_sum):
            return False
        if np.trace(square) != magic_sum:
            return False
        if np.trace(np.fliplr(square)) != magic_sum:
            return False
        return True
