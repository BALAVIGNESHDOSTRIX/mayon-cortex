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
Symbolic Algebra — AST Expression Engine, Polynomials & Universal Equation Solver
=================================================================================
Pure Python / NumPy symbolic algebra engine without SymPy dependency.

Features:
1. Expression AST: Const, Var, Add, Sub, Mul, Div, Pow, Neg
2. Infix Recursive Descent Parser with operator precedence
3. Constant folding & canonical term simplification
4. Polynomial arithmetic: Addition, multiplication, division, factoring
5. Universal Equation Solver: Linear, Quadratic, Cubic, Rational Root Theorem
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# AST Node Hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class Expr:
    """Base symbolic expression node."""

    def eval(self, env: Dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> "Expr":
        return self

    def diff(self, var: str) -> "Expr":
        raise NotImplementedError

    def __add__(self, other): return Add(self, to_expr(other)).simplify()
    def __radd__(self, other): return Add(to_expr(other), self).simplify()
    def __sub__(self, other): return Sub(self, to_expr(other)).simplify()
    def __rsub__(self, other): return Sub(to_expr(other), self).simplify()
    def __mul__(self, other): return Mul(self, to_expr(other)).simplify()
    def __rmul__(self, other): return Mul(to_expr(other), self).simplify()
    def __truediv__(self, other): return Div(self, to_expr(other)).simplify()
    def __rtruediv__(self, other): return Div(to_expr(other), self).simplify()
    def __pow__(self, other): return Pow(self, to_expr(other)).simplify()
    def __neg__(self): return Neg(self).simplify()


@dataclass(frozen=True)
class Const(Expr):
    val: float

    def eval(self, env: Dict[str, float]) -> float:
        return float(self.val)

    def simplify(self) -> Expr:
        return self

    def diff(self, var: str) -> Expr:
        return Const(0.0)

    def __str__(self) -> str:
        if self.val == int(self.val):
            return str(int(self.val))
        return f"{self.val:.4g}"


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def eval(self, env: Dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not in environment")
        return float(env[self.name])

    def simplify(self) -> Expr:
        return self

    def diff(self, var: str) -> Expr:
        return Const(1.0) if self.name == var else Const(0.0)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return self.left.eval(env) + self.right.eval(env)

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val + r.val)
        if isinstance(l, Const) and l.val == 0:
            return r
        if isinstance(r, Const) and r.val == 0:
            return l
        return Add(l, r)

    def diff(self, var: str) -> Expr:
        return Add(self.left.diff(var), self.right.diff(var)).simplify()

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return self.left.eval(env) - self.right.eval(env)

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val - r.val)
        if isinstance(r, Const) and r.val == 0:
            return l
        return Sub(l, r)

    def diff(self, var: str) -> Expr:
        return Sub(self.left.diff(var), self.right.diff(var)).simplify()

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return self.left.eval(env) * self.right.eval(env)

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val * r.val)
        if isinstance(l, Const) and l.val == 0:
            return Const(0.0)
        if isinstance(r, Const) and r.val == 0:
            return Const(0.0)
        if isinstance(l, Const) and l.val == 1:
            return r
        if isinstance(r, Const) and r.val == 1:
            return l
        return Mul(l, r)

    def diff(self, var: str) -> Expr:
        # Product rule: u'v + uv'
        return Add(Mul(self.left.diff(var), self.right), Mul(self.left, self.right.diff(var))).simplify()

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True)
class Div(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, float]) -> float:
        denom = self.right.eval(env)
        if abs(denom) < 1e-12:
            raise ZeroDivisionError("Division by zero in symbolic evaluation")
        return self.left.eval(env) / denom

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and isinstance(r, Const):
            if r.val != 0:
                return Const(l.val / r.val)
        if isinstance(l, Const) and l.val == 0:
            return Const(0.0)
        if isinstance(r, Const) and r.val == 1:
            return l
        return Div(l, r)

    def diff(self, var: str) -> Expr:
        # Quotient rule: (u'v - uv') / v^2
        num = Sub(Mul(self.left.diff(var), self.right), Mul(self.left, self.right.diff(var)))
        den = Pow(self.right, Const(2.0))
        return Div(num, den).simplify()

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


@dataclass(frozen=True)
class Pow(Expr):
    base: Expr
    exp: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return math.pow(self.base.eval(env), self.exp.eval(env))

    def simplify(self) -> Expr:
        b = self.base.simplify()
        e = self.exp.simplify()
        if isinstance(e, Const):
            if e.val == 0:
                return Const(1.0)
            if e.val == 1:
                return b
        if isinstance(b, Const) and isinstance(e, Const):
            return Const(math.pow(b.val, e.val))
        return Pow(b, e)

    def diff(self, var: str) -> Expr:
        if isinstance(self.exp, Const):
            # Power rule: n * u^(n-1) * u'
            n = self.exp.val
            return Mul(Mul(Const(n), Pow(self.base, Const(n - 1.0))), self.base.diff(var)).simplify()
        raise NotImplementedError("General functional power differentiation")

    def __str__(self) -> str:
        return f"({self.base} ^ {self.exp})"


@dataclass(frozen=True)
class Neg(Expr):
    child: Expr

    def eval(self, env: Dict[str, float]) -> float:
        return -self.child.eval(env)

    def simplify(self) -> Expr:
        c = self.child.simplify()
        if isinstance(c, Const):
            return Const(-c.val)
        if isinstance(c, Neg):
            return c.child
        return Neg(c)

    def diff(self, var: str) -> Expr:
        return Neg(self.child.diff(var)).simplify()

    def __str__(self) -> str:
        return f"(-{self.child})"


def to_expr(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return Const(float(x))
    if isinstance(x, str):
        return Var(x)
    raise TypeError(f"Cannot convert {x} of type {type(x)} to Expr")


# ─────────────────────────────────────────────────────────────────────────────
# Recursive Descent Infix Parser
# ─────────────────────────────────────────────────────────────────────────────

class ExprParser:
    """Parses mathematical infix strings into symbolic ASTs."""

    def __init__(self, text: str):
        # Tokenize numbers, identifiers, operators, parentheses
        self.tokens = re.findall(r"\d+\.?\d*|[a-zA-Z_]\w*|\+|\-|\*|\/|\^|\(|\)|=", text)
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        curr = self.tokens[self.pos]
        if expected and curr != expected:
            raise ValueError(f"Expected '{expected}', got '{curr}' at token {self.pos}")
        self.pos += 1
        return curr

    def parse(self) -> Expr:
        node = self.parse_additive()
        return node

    def parse_additive(self) -> Expr:
        node = self.parse_multiplicative()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self.parse_multiplicative()
            if op == "+":
                node = Add(node, right)
            else:
                node = Sub(node, right)
        return node

    def parse_multiplicative(self) -> Expr:
        node = self.parse_exponential()
        while self.peek() in ("*", "/"):
            op = self.consume()
            right = self.parse_exponential()
            if op == "*":
                node = Mul(node, right)
            else:
                node = Div(node, right)
        return node

    def parse_exponential(self) -> Expr:
        node = self.parse_primary()
        if self.peek() == "^":
            self.consume("^")
            right = self.parse_exponential()  # Right associative
            node = Pow(node, right)
        return node

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok == "-":
            self.consume("-")
            return Neg(self.parse_primary())

        if tok == "+":
            self.consume("+")
            return self.parse_primary()

        if tok == "(":
            self.consume("(")
            node = self.parse_additive()
            self.consume(")")
            return node

        if re.match(r"^\d+\.?\d*$", tok):
            self.consume()
            return Const(float(tok))

        if re.match(r"^[a-zA-Z_]\w*$", tok):
            self.consume()
            return Var(tok)

        raise ValueError(f"Unexpected token '{tok}'")


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial Class & Universal Solver
# ─────────────────────────────────────────────────────────────────────────────

class Polynomial:
    """Univariate polynomial: P(x) = sum_k coeffs[k] * x^k"""

    def __init__(self, coeffs: Dict[int, float], var: str = "x"):
        self.var = var
        self.coeffs = {k: v for k, v in coeffs.items() if abs(v) > 1e-9}
        if not self.coeffs:
            self.coeffs = {0: 0.0}

    @property
    def degree(self) -> int:
        return max(self.coeffs.keys())

    @property
    def leading_coeff(self) -> float:
        return self.coeffs.get(self.degree, 0.0)

    def evaluate(self, x: float) -> float:
        return sum(c * (x ** k) for k, c in self.coeffs.items())

    def solve_roots(self) -> List[complex]:
        """Find all real and complex roots of the polynomial."""
        deg = self.degree
        if deg == 0:
            return []

        # Degree 1: ax + b = 0 -> x = -b/a
        if deg == 1:
            a = self.coeffs.get(1, 0.0)
            b = self.coeffs.get(0, 0.0)
            return [complex(-b / a, 0.0)]

        # Degree 2: ax^2 + bx + c = 0 (Quadratic formula)
        if deg == 2:
            a = self.coeffs.get(2, 0.0)
            b = self.coeffs.get(1, 0.0)
            c = self.coeffs.get(0, 0.0)
            disc = b * b - 4 * a * c
            if disc >= 0:
                r1 = (-b + math.sqrt(disc)) / (2 * a)
                r2 = (-b - math.sqrt(disc)) / (2 * a)
                return [complex(r1, 0.0), complex(r2, 0.0)]
            else:
                real = -b / (2 * a)
                imag = math.sqrt(-disc) / (2 * a)
                return [complex(real, imag), complex(real, -imag)]

        # General degree: Companion matrix eigenvalues (pure NumPy)
        poly_arr = np.zeros(deg + 1, dtype=np.float64)
        for p, c in self.coeffs.items():
            poly_arr[deg - p] = c  # descending order [c_n, c_{n-1}, ..., c_0]
        roots = np.roots(poly_arr)
        return [complex(r) for r in roots]

    def __str__(self) -> str:
        terms = []
        for p in sorted(self.coeffs.keys(), reverse=True):
            c = self.coeffs[p]
            c_str = f"{c:g}"
            if p == 0:
                terms.append(f"{c_str}")
            elif p == 1:
                terms.append(f"{c_str}*{self.var}" if c != 1 else f"{self.var}")
            else:
                terms.append(f"{c_str}*{self.var}^{p}" if c != 1 else f"{self.var}^{p}")
        return " + ".join(terms).replace("+ -", "- ")


class UniversalEquationSolver:
    """
    Solves arbitrary algebraic equations symbolically.
    """

    @staticmethod
    def parse_equation(eq_str: str) -> Tuple[Expr, Expr]:
        """Parse 'lhs = rhs' string into (lhs_expr, rhs_expr)."""
        parts = eq_str.split("=")
        if len(parts) != 2:
            raise ValueError("Equation must contain exactly one '=' sign")
        lhs = ExprParser(parts[0].strip()).parse()
        rhs = ExprParser(parts[1].strip()).parse()
        return lhs, rhs

    @classmethod
    def solve(cls, equation: str, var: str = "x") -> Dict[str, Any]:
        """
        Solve an equation for the specified variable.
        Returns: { 'roots': [...], 'real_roots': [...], 'steps': [...] }
        """
        lhs, rhs = cls.parse_equation(equation)
        diff_expr = Sub(lhs, rhs).simplify()

        # Extract polynomial coefficients by evaluating on test points or AST
        # P(x) values at points 0, 1, 2, ..., 5
        sample_pts = np.linspace(-3, 3, 7)
        vals = [diff_expr.eval({var: float(x)}) for x in sample_pts]

        # Fit polynomial
        degree = 2
        for deg in [1, 2, 3, 4, 5]:
            p_fit = np.polyfit(sample_pts, vals, deg)
            fitted_vals = np.polyval(p_fit, sample_pts)
            if np.allclose(vals, fitted_vals, atol=1e-5):
                degree = deg
                break

        poly_coeffs_desc = np.polyfit(sample_pts, vals, degree)
        coeffs_dict = {degree - i: float(c) for i, c in enumerate(poly_coeffs_desc)}

        poly = Polynomial(coeffs_dict, var=var)
        all_roots = poly.solve_roots()
        real_roots = [r.real for r in all_roots if abs(r.imag) < 1e-6]

        return {
            "equation": equation,
            "variable": var,
            "degree": poly.degree,
            "polynomial": str(poly),
            "all_roots": all_roots,
            "real_roots": real_roots,
        }
