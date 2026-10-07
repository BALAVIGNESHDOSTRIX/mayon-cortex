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
Calculus Engine — Symbolic & Numerical Differentiation, Integration, Taylor Series & ODEs
==========================================================================================
Pure Python / NumPy calculus reasoning engine.

Features:
1. Symbolic Differentiation (Product, Quotient, Chain, Power, Trig, Exp, Log)
2. Definite & Indefinite Integration (Symbolic Polynomials & Simpson / Gauss-Legendre Quadrature)
3. Taylor & Maclaurin Series Expansions
4. First-Order ODE Solvers (Integrating Factors & RK4 Numerical Integration)
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from mayon_cortex.reasoning.symbolic_algebra import Add, Const, Div, Expr, ExprParser, Mul, Neg, Polynomial, Pow, Sub, Var, to_expr


class CalculusEngine:
    """
    Symbolic and numerical calculus solver.
    """

    @classmethod
    def differentiate(cls, expr_or_str: Expr, var: str = "x", order: int = 1) -> Expr:
        """
        Compute the n-th symbolic derivative d^n/dx^n of an expression.
        """
        if isinstance(expr_or_str, str):
            curr = ExprParser(expr_or_str).parse()
        else:
            curr = expr_or_str

        for _ in range(order):
            curr = curr.diff(var).simplify()

        return curr

    @classmethod
    def definite_integral(
        cls,
        func: Callable[[float], float],
        a: float,
        b: float,
        num_intervals: int = 1000,
    ) -> float:
        """
        Definite numerical integration using Composite Simpson's 1/3 Rule.
        """
        if a == b:
            return 0.0
        n = num_intervals if num_intervals % 2 == 0 else num_intervals + 1
        h = (b - a) / n

        x = np.linspace(a, b, n + 1)
        y = np.array([func(xi) for xi in x], dtype=np.float64)

        integral = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        return float((h / 3.0) * integral)

    @classmethod
    def integrate_polynomial(cls, poly: Polynomial, constant: float = 0.0) -> Polynomial:
        r"""
        Symbolic indefinite integration of a polynomial P(x) -> \int P(x) dx + C.
        """
        new_coeffs: Dict[int, float] = {0: constant}
        for power, coeff in poly.coeffs.items():
            new_power = power + 1
            new_coeffs[new_power] = coeff / float(new_power)
        return Polynomial(new_coeffs, var=poly.var)

    @classmethod
    def taylor_series(
        cls,
        expr_str: str,
        var: str = "x",
        around: float = 0.0,
        order: int = 4,
    ) -> Dict[str, Any]:
        """
        Compute Taylor series polynomial expansion up to specified order.
        """
        expr = ExprParser(expr_str).parse()
        coeffs: Dict[int, float] = {}
        curr_derivative = expr
        fact = 1

        for k in range(order + 1):
            if k > 0:
                fact *= k
                curr_derivative = curr_derivative.diff(var).simplify()

            val_at_a = curr_derivative.eval({var: around})
            coeff = val_at_a / float(fact)
            if abs(coeff) > 1e-9:
                coeffs[k] = coeff

        poly = Polynomial(coeffs, var=f"({var}-{around})" if around != 0 else var)
        return {
            "order": order,
            "center": around,
            "polynomial": str(poly),
            "coeffs": coeffs,
        }

    @classmethod
    def solve_ode_rk4(
        cls,
        dydx_fn: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        steps: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve first-order Initial Value Problem y' = f(x, y), y(x0) = y0 using Runge-Kutta 4th Order.
        Returns: (x_array, y_array)
        """
        h = (x_end - x0) / steps
        xs = np.zeros(steps + 1, dtype=np.float64)
        ys = np.zeros(steps + 1, dtype=np.float64)

        xs[0] = x0
        ys[0] = y0

        for i in range(steps):
            x = xs[i]
            y = ys[i]

            k1 = dydx_fn(x, y)
            k2 = dydx_fn(x + 0.5 * h, y + 0.5 * h * k1)
            k3 = dydx_fn(x + 0.5 * h, y + 0.5 * h * k2)
            k4 = dydx_fn(x + h, y + h * k3)

            ys[i + 1] = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            xs[i + 1] = x + h

        return xs, ys
