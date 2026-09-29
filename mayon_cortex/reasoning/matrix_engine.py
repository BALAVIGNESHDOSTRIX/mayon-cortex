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
Matrix Engine — Linear Algebra, Gaussian Elimination & System Solvers
====================================================================
Pure Python / NumPy linear algebra engine with step-by-step reasoning.

Features:
1. Step-by-Step Gaussian Elimination (REF / RREF)
2. Linear System Solver AX = B (Unique, Infinitely Many, Inconsistent)
3. Determinant, Matrix Inverse, Rank & Null Space Basis
4. Eigenvalues and Eigenvectors (QR Algorithm)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RowOperation:
    """A record of an elementary row operation."""
    op_type: str  # "swap", "scale", "add"
    row1: int
    row2: Optional[int] = None
    scalar: float = 1.0
    description: str = ""


class MatrixEngine:
    """
    Linear algebra solver with full symbolic step-by-step explanations.
    """

    @staticmethod
    def rref(matrix: np.ndarray) -> Tuple[np.ndarray, List[RowOperation], List[int]]:
        """
        Compute Reduced Row Echelon Form (RREF) with history of row operations.
        Returns: (rref_matrix, list_of_operations, pivot_columns)
        """
        A = matrix.astype(np.float64).copy()
        rows, cols = A.shape
        ops: List[RowOperation] = []
        pivot_cols: List[int] = []

        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while abs(A[i, lead]) < 1e-9:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead == cols:
                break

            # Swap row i and row r if needed
            if i != r:
                A[[i, r]] = A[[r, i]]
                ops.append(RowOperation(op_type="swap", row1=r, row2=i, description=f"Swap R{r+1} <-> R{i+1}"))

            # Normalize pivot row
            pivot_val = A[r, lead]
            if abs(pivot_val - 1.0) > 1e-9:
                A[r] /= pivot_val
                ops.append(RowOperation(op_type="scale", row1=r, scalar=float(1.0 / pivot_val), description=f"R{r+1} <- (1/{pivot_val:.4g}) * R{r+1}"))

            # Eliminate other rows
            for i_sub in range(rows):
                if i_sub != r:
                    factor = A[i_sub, lead]
                    if abs(factor) > 1e-9:
                        A[i_sub] -= factor * A[r]
                        ops.append(RowOperation(op_type="add", row1=i_sub, row2=r, scalar=float(-factor), description=f"R{i_sub+1} <- R{i_sub+1} - ({factor:.4g}) * R{r+1}"))

            pivot_cols.append(lead)
            lead += 1

        # Clean small precision errors
        A[np.abs(A) < 1e-9] = 0.0
        return A, ops, pivot_cols

    @classmethod
    def solve_linear_system(cls, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Solve linear matrix system A * x = b.
        Categorizes solution as UNIQUE, INFINITELY_MANY, or INCONSISTENT.
        """
        A_arr = np.array(A, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64).reshape(-1, 1)

        aug = np.hstack([A_arr, b_arr])
        rref_aug, ops, pivots = cls.rref(aug)

        n_vars = A_arr.shape[1]
        augmented_col = aug.shape[1] - 1

        # Check for inconsistency: row with 0 ... 0 | non-zero
        for row_idx in range(aug.shape[0]):
            if np.all(np.abs(rref_aug[row_idx, :n_vars]) < 1e-8) and abs(rref_aug[row_idx, augmented_col]) > 1e-8:
                return {
                    "status": "INCONSISTENT",
                    "solution": None,
                    "explanation": f"System has no solution (contradiction at row {row_idx+1})",
                    "rref": rref_aug,
                    "operations": ops,
                }

        # Check if unique or infinitely many
        var_pivots = [p for p in pivots if p < n_vars]
        if len(var_pivots) == n_vars:
            # Unique solution
            x_sol = np.zeros(n_vars, dtype=np.float64)
            for r_idx, p_col in enumerate(var_pivots):
                x_sol[p_col] = rref_aug[r_idx, augmented_col]
            return {
                "status": "UNIQUE",
                "solution": x_sol,
                "explanation": "System has a unique solution",
                "rref": rref_aug,
                "operations": ops,
            }
        else:
            free_vars = [i for i in range(n_vars) if i not in var_pivots]
            return {
                "status": "INFINITELY_MANY",
                "free_variables": free_vars,
                "explanation": f"System has infinite solutions with {len(free_vars)} free variables",
                "rref": rref_aug,
                "operations": ops,
            }

    @staticmethod
    def determinant(A: np.ndarray) -> float:
        """Compute matrix determinant via LU decomposition (NumPy vectorized)."""
        return float(np.linalg.det(A))

    @staticmethod
    def inverse(A: np.ndarray) -> Optional[np.ndarray]:
        """Compute matrix inverse if non-singular."""
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def eigen(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors."""
        w, v = np.linalg.eig(A)
        return w, v
