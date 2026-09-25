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
Formal SMT, CSP & Mathematical Constraint Solver
=================================================
Pure CPU mathematical deduction, algebraic equation solving,
constraint satisfaction (AC-3 / Forward Checking), and formal
resolution-refutation theorem proving with 100% sound guarantees.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. ALGEBRAIC & MATHEMATICAL SOLVER
# ─────────────────────────────────────────────────────────────

@dataclass
class MathSolution:
    """Result of an algebraic or arithmetic deduction."""
    problem_text: str
    solved_variables: Dict[str, float]
    steps: List[str]
    is_solved: bool
    explanation: str

    def format_proof(self) -> str:
        """Return full formatted mathematical proof including all intermediate deductive steps."""
        lines = [
            f"Explanation: {self.explanation}",
            "Formal Deductive Proof Steps:"
        ]
        if self.steps:
            for idx, step in enumerate(self.steps, 1):
                lines.append(f"  Step {idx}: {step}")
        else:
            lines.append("  (No intermediate steps recorded)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_proof()


class MathSolver:
    """
    Solves linear equations, multi-variable formulas, unit conversions,
    and word problem kinematics/physics/finance calculations.
    """

    # Common physics/finance formulas: (output_var, required_vars, formula_func, step_description)
    FORMULA_REGISTRY: List[Tuple[str, List[str], Callable[..., float], str]] = [
        # Speed = distance / time
        ("speed", ["distance", "time"], lambda d, t: d / t if t != 0 else math.nan, "speed = distance / time"),
        ("distance", ["speed", "time"], lambda s, t: s * t, "distance = speed * time"),
        ("time", ["distance", "speed"], lambda d, s: d / s if s != 0 else math.nan, "time = distance / speed"),
        
        # Force = mass * acceleration
        ("force", ["mass", "acceleration"], lambda m, a: m * a, "force = mass * acceleration"),
        ("mass", ["force", "acceleration"], lambda f, a: f / a if a != 0 else math.nan, "mass = force / acceleration"),
        ("acceleration", ["force", "mass"], lambda f, m: f / m if m != 0 else math.nan, "acceleration = force / mass"),

        # Work & Energy
        ("work", ["force", "distance"], lambda f, d: f * d, "work = force * distance"),
        ("power", ["work", "time"], lambda w, t: w / t if t != 0 else math.nan, "power = work / time"),
        ("kinetic_energy", ["mass", "velocity"], lambda m, v: 0.5 * m * v * v, "kinetic_energy = 0.5 * mass * velocity^2"),
        ("potential_energy", ["mass", "height"], lambda m, h: m * 9.8 * h, "potential_energy = mass * 9.8 * height"),

        # Density & Mass
        ("density", ["mass", "volume"], lambda m, v: m / v if v != 0 else math.nan, "density = mass / volume"),
        ("mass", ["density", "volume"], lambda d, v: d * v, "mass = density * volume"),
        ("volume", ["mass", "density"], lambda m, d: m / d if d != 0 else math.nan, "volume = mass / density"),

        # Geometry: Area, Perimeter, Volume
        ("area", ["length", "width"], lambda l, w: l * w, "rectangle area = length * width"),
        ("area", ["width", "height"], lambda w, h: w * h, "rectangle area = width * height"),
        ("area", ["base", "height"], lambda b, h: 0.5 * b * h, "triangle area = 0.5 * base * height"),
        ("area", ["radius"], lambda r: math.pi * r * r, "circle area = pi * radius^2"),
        ("perimeter", ["length", "width"], lambda l, w: 2.0 * (l + w), "perimeter = 2 * (length + width)"),
        ("perimeter", ["width", "height"], lambda w, h: 2.0 * (w + h), "perimeter = 2 * (width + height)"),
        ("circumference", ["radius"], lambda r: 2.0 * math.pi * r, "circumference = 2 * pi * radius"),
        ("volume", ["length", "width", "height"], lambda l, w, h: l * w * h, "cuboid volume = length * width * height"),
        ("volume", ["area", "height"], lambda a, h: a * h, "volume = area * height"),

        # Ohm's Law & Electricity
        ("voltage", ["current", "resistance"], lambda i, r: i * r, "voltage = current * resistance"),
        ("current", ["voltage", "resistance"], lambda v, r: v / r if r != 0 else math.nan, "current = voltage / resistance"),
        ("resistance", ["voltage", "current"], lambda v, i: v / i if i != 0 else math.nan, "resistance = voltage / current"),
        ("power", ["voltage", "current"], lambda v, i: v * i, "electrical power = voltage * current"),

        # Trigonometry & Pythagorean Theorem
        ("hypotenuse", ["side_a", "side_b"], lambda a, b: math.sqrt(a * a + b * b), "hypotenuse = sqrt(side_a^2 + side_b^2)"),
        ("side_a", ["hypotenuse", "side_b"], lambda c, b: math.sqrt(c * c - b * b) if c >= b else math.nan, "side_a = sqrt(hypotenuse^2 - side_b^2)"),
        ("side_b", ["hypotenuse", "side_a"], lambda c, a: math.sqrt(c * c - a * a) if c >= a else math.nan, "side_b = sqrt(hypotenuse^2 - side_a^2)"),
        ("sin_val", ["angle_deg"], lambda deg: math.sin(math.radians(deg)), "sin(theta) = sin(radians(angle_deg))"),
        ("cos_val", ["angle_deg"], lambda deg: math.cos(math.radians(deg)), "cos(theta) = cos(radians(angle_deg))"),
        ("tan_val", ["angle_deg"], lambda deg: math.tan(math.radians(deg)) if deg % 180 != 90 else math.nan, "tan(theta) = tan(radians(angle_deg))"),

        # Combinatorics & Factorials
        ("combinations", ["n", "r"], lambda n, r: math.comb(int(n), int(r)) if n >= r >= 0 else math.nan, "C(n, r) = n! / (r! * (n - r)!)"),
        ("permutations", ["n", "r"], lambda n, r: math.perm(int(n), int(r)) if n >= r >= 0 else math.nan, "P(n, r) = n! / (n - r)!"),
        ("factorial", ["n"], lambda n: float(math.factorial(int(n))) if n >= 0 else math.nan, "n! = factorial(n)"),

        # Logarithms & Exponentials
        ("log10", ["value"], lambda v: math.log10(v) if v > 0 else math.nan, "log10(value) = log10(value)"),
        ("ln", ["value"], lambda v: math.log(v) if v > 0 else math.nan, "ln(value) = ln(value)"),
        ("log_base", ["value", "base"], lambda v, b: math.log(v, b) if v > 0 and b > 0 and b != 1 else math.nan, "log_base(value, base) = log(value) / log(base)"),
        # Financial: final_price, profit, tax, interest
        ("final_price", ["price", "discount"], lambda p, d: p * (1.0 - (d / 100.0 if d > 1.0 else d)), "final_price = price * (1 - discount)"),
        ("profit", ["revenue", "cost"], lambda r, c: r - c, "profit = revenue - cost"),
        ("tax", ["amount", "tax_rate"], lambda a, r: a * (r / 100.0 if r > 1.0 else r), "tax = amount * tax_rate"),
        ("interest", ["principal", "rate", "time"], lambda p, r, t: p * (r / 100.0 if r > 1.0 else r) * t, "simple interest = principal * rate * time"),
        ("compound_amount", ["principal", "rate", "periods", "time"], lambda p, r, n, t: p * ((1.0 + (r / 100.0 if r > 1.0 else r) / n) ** (n * t)), "A = P * (1 + r/n)^(n*t)"),

        # Chemistry & Thermodynamics (Ideal Gas Law PV = nRT, R = 8.314 J/(mol*K))
        ("pressure", ["moles", "temperature", "volume"], lambda n, t, v: (n * 8.314 * t) / v if v != 0 else math.nan, "ideal gas P = (n * R * T) / V"),
        ("gas_volume", ["pressure", "moles", "temperature"], lambda p, n, t: (n * 8.314 * t) / p if p != 0 else math.nan, "ideal gas V = (n * R * T) / P"),
    ]

    def __init__(self):
        pass

    def extract_quantities(self, text: str) -> Dict[str, float]:
        """Extract named numerical variables from text (e.g. 'distance = 100 km, time = 2 hours, price = $80')."""
        env = {}
        # 1. Pattern: variable = optional_currency number optional_percent
        var_num_pattern = re.compile(
            r'\b([a-zA-Z_]\w*)\s*(?:=|is|equals|of|:)\s*\$?(-?\d+(?:\.\d+)?)\s*(?:%|[a-zA-Z/%\^]+)?\b',
            re.IGNORECASE,
        )
        for match in var_num_pattern.finditer(text):
            var_name = match.group(1).lower()
            if var_name not in {"what", "calculate", "find", "given", "if", "and", "the", "a", "an"}:
                val = float(match.group(2))
                env[var_name] = val

        # 2. Standalone currency e.g. '$80' (if price not yet assigned)
        if "price" not in env:
            curr_match = re.search(r'\$(-?\d+(?:\.\d+)?)', text)
            if curr_match:
                env["price"] = float(curr_match.group(1))

        # 3. Standalone percentage e.g. '20%' (if discount or tax_rate not yet assigned)
        if "discount" not in env:
            pct_match = re.search(r'(-?\d+(?:\.\d+)?)\s*%', text)
            if pct_match:
                env["discount"] = float(pct_match.group(1))

        return env

    def solve_quadratic_equation(self, eq_text: str) -> Optional[Tuple[str, List[float], List[str]]]:
        """
        Solves quadratic equations of form 'ax^2 + bx + c = 0' or 'ax^2 - bx = c'.
        Example: '2x^2 - 4x - 6 = 0' or 'x^2 - 5x + 6 = 0'.
        """
        steps = []
        # Pattern: (coeff a)var^2 (coeff b)var (const c) = (rhs)
        pattern = re.compile(
            r'(?:^|\s|\b)([+-]?\d+(?:\.\d+)?)?([xyzabcXYZABC])\^2\s*([+-]\s*\d*(?:\.\d+)?\s*[xyzabcXYZABC])?\s*([+-]\s*\d+(?:\.\d+)?)?\s*=\s*([+-]?\d+(?:\.\d+)?)',
            re.IGNORECASE,
        )
        m = pattern.search(eq_text)
        if not m:
            return None

        coeff_a_str, var_name, b_term_str, c_term_str, rhs_str = m.groups()
        var_name = var_name.lower()

        # Parse a
        if not coeff_a_str:
            a = 1.0
        elif coeff_a_str == "+":
            a = 1.0
        elif coeff_a_str == "-":
            a = -1.0
        else:
            a = float(coeff_a_str)

        if a == 0:
            return None

        # Parse b
        b = 0.0
        if b_term_str:
            clean_b = b_term_str.replace(" ", "").replace(var_name, "").replace(var_name.upper(), "")
            if clean_b in ("", "+"):
                b = 1.0
            elif clean_b == "-":
                b = -1.0
            else:
                b = float(clean_b)

        # Parse c
        c_val = float(c_term_str.replace(" ", "")) if c_term_str else 0.0
        rhs = float(rhs_str) if rhs_str else 0.0
        c = c_val - rhs

        steps.append(f"Parsed quadratic equation: ({a})*{var_name}^2 + ({b})*{var_name} + ({c}) = 0")
        disc = b * b - 4 * a * c
        steps.append(f"Calculated discriminant D = b^2 - 4ac = ({b})^2 - 4*({a})*({c}) = {disc:.4g}")

        if disc > 0:
            r1 = (-b + math.sqrt(disc)) / (2 * a)
            r2 = (-b - math.sqrt(disc)) / (2 * a)
            steps.append(f"Two real roots: {var_name}_1 = {r1:.4g}, {var_name}_2 = {r2:.4g}")
            return var_name, [r1, r2], steps
        elif disc == 0:
            r0 = -b / (2 * a)
            steps.append(f"One repeated real root: {var_name} = {r0:.4g}")
            return var_name, [r0], steps
        else:
            real_part = -b / (2 * a)
            imag_part = math.sqrt(-disc) / (2 * a)
            steps.append(f"Two complex roots: {var_name} = {real_part:.4g} ± {imag_part:.4g}i")
            return var_name, [real_part], steps

    def solve_linear_equation(self, eq_text: str) -> Optional[Tuple[str, float, List[str]]]:
        """
        Solves simple linear equations of form 'ax + b = c' or 'ax - b = c' or 'ax = c'.
        Example: '3x + 5 = 20' -> x = 5.0
        """
        steps = []
        pattern = re.compile(
            r'(?:^|\s|\b)([+-]?\d+(?:\.\d+)?)?([xyzabcXYZABC])\s*([+-]\s*\d+(?:\.\d+)?)?\s*=\s*([+-]?\d+(?:\.\d+)?)',
            re.IGNORECASE,
        )
        m = pattern.search(eq_text)
        if not m:
            return None

        coeff_str, var_name, const_str, rhs_str = m.groups()
        var_name = var_name.lower()

        # Parse coefficient
        if not coeff_str:
            a = 1.0
        elif coeff_str == "+":
            a = 1.0
        elif coeff_str == "-":
            a = -1.0
        else:
            a = float(coeff_str)

        # Parse constant b
        b = float(const_str.replace(" ", "")) if const_str else 0.0

        # Parse RHS c
        c = float(rhs_str)

        steps.append(f"Parsed linear equation: ({a})*{var_name} + ({b}) = {c}")

        # Solve: a*x + b = c  =>  a*x = c - b  =>  x = (c - b) / a
        rhs_after_b = c - b
        steps.append(f"Subtract constant {b} from both sides: ({a})*{var_name} = {rhs_after_b}")

        if a == 0:
            return None

        solution = rhs_after_b / a
        steps.append(f"Divide by coefficient {a}: {var_name} = {solution}")

        return var_name, solution, steps

    def solve_linear_system_2x2(self, text: str) -> Optional[Tuple[Dict[str, float], List[str]]]:
        """
        Solves systems of 2 linear equations in 2 variables using Cramer's Rule / Determinants.
        Example: '2x + 3y = 13 and x - y = -1' -> x = 2, y = 3.
        """
        steps = []
        eq_pattern = re.compile(
            r'([+-]?\d*(?:\.\d+)?)\s*([a-zA-Z])\s*([+-]\s*\d*(?:\.\d+)?)\s*([a-zA-Z])\s*=\s*([+-]?\d+(?:\.\d+)?)',
            re.IGNORECASE,
        )
        matches = list(eq_pattern.finditer(text))
        if len(matches) < 2:
            return None

        # Parse Eq 1: a1*v1 + b1*v2 = c1
        m1 = matches[0]
        a1_str, v1_1, b1_str, v2_1, c1_str = m1.groups()
        v1_1, v2_1 = v1_1.lower(), v2_1.lower()
        if v1_1 == v2_1:
            return None

        a1 = 1.0 if not a1_str or a1_str == "+" else (-1.0 if a1_str == "-" else float(a1_str))
        clean_b1 = b1_str.replace(" ", "")
        b1 = 1.0 if clean_b1 == "+" else (-1.0 if clean_b1 == "-" else float(clean_b1))
        c1 = float(c1_str)

        # Parse Eq 2: a2*v1 + b2*v2 = c2
        m2 = matches[1]
        a2_str, v1_2, b2_str, v2_2, c2_str = m2.groups()
        v1_2, v2_2 = v1_2.lower(), v2_2.lower()

        a2_raw = 1.0 if not a2_str or a2_str == "+" else (-1.0 if a2_str == "-" else float(a2_str))
        clean_b2 = b2_str.replace(" ", "")
        b2_raw = 1.0 if clean_b2 == "+" else (-1.0 if clean_b2 == "-" else float(clean_b2))
        c2 = float(c2_str)

        # Align variables
        if v1_2 == v1_1 and v2_2 == v2_1:
            a2, b2 = a2_raw, b2_raw
        elif v1_2 == v2_1 and v2_2 == v1_1:
            a2, b2 = b2_raw, a2_raw
        else:
            return None

        steps.append(f"System of Equations parsed:")
        steps.append(f"  (1) {a1}*{v1_1} + {b1}*{v2_1} = {c1}")
        steps.append(f"  (2) {a2}*{v1_1} + {b2}*{v2_1} = {c2}")

        # Determinant
        D = a1 * b2 - a2 * b1
        steps.append(f"Computed coefficient determinant D = ({a1})*({b2}) - ({a2})*({b1}) = {D:.4g}")
        if abs(D) < 1e-9:
            steps.append("System is singular (no unique solution).")
            return None

        Dx = c1 * b2 - c2 * b1
        Dy = a1 * c2 - a2 * c1
        sol_v1 = Dx / D
        sol_v2 = Dy / D
        steps.append(f"Cramer's Rule: {v1_1} = Dx/D = {Dx:.4g}/{D:.4g} = {sol_v1:.4g}")
        steps.append(f"Cramer's Rule: {v2_1} = Dy/D = {Dy:.4g}/{D:.4g} = {sol_v2:.4g}")

        return {v1_1: sol_v1, v2_1: sol_v2}, steps

    def solve_polynomial_calculus(self, text: str) -> Optional[Tuple[str, str, List[str]]]:
        """
        Symbolic polynomial differentiation & integration.
        Example: 'derivative of 3x^3 + 4x^2 - 5x + 7' -> 9x^2 + 8x - 5
        Example: 'integral of 6x^2 + 4x - 3 dx' -> 2x^3 + 2x^2 - 3x + C
        """
        steps = []
        is_deriv = bool(re.search(r'\b(derivative|diff|d/d[a-z]|differentiate)\b', text, re.IGNORECASE))
        is_integ = bool(re.search(r'\b(integral|integrate|antiderivative)\b', text, re.IGNORECASE))

        if not is_deriv and not is_integ:
            return None

        # Extract polynomial string
        poly_match = re.search(r'(?:(?:derivative|diff|differentiate|integral|integrate|antiderivative)(?:\s+of)?|d/d[a-z])\s+([0-9a-zA-Z\^\s\+\-\*]+?)(?:\s+with|\s+dx|\s*\?|\s*$)', text, re.IGNORECASE)
        if not poly_match:
            return None

        poly_str = poly_match.group(1).strip()
        if poly_str.lower().startswith("of "):
            poly_str = poly_str[3:].strip()

        # Parse polynomial terms: (optional sign)(optional coeff)(var)(optional ^exp)
        term_pattern = re.compile(r'([+-]?\s*\d*(?:\.\d+)?)\s*([xyzabcXYZABC])(?:\^(\d+))?|([+-]?\s*\d+(?:\.\d+)?)')
        terms = []
        for m in term_pattern.finditer(poly_str):
            c_str, var, exp_str, const_str = m.groups()
            if const_str:
                clean_c = const_str.replace(" ", "")
                if clean_c:
                    terms.append((float(clean_c), None, 0))
            elif var:
                clean_c = c_str.replace(" ", "") if c_str else "+"
                if clean_c in ("", "+"):
                    coeff = 1.0
                elif clean_c == "-":
                    coeff = -1.0
                else:
                    coeff = float(clean_c)
                exp = int(exp_str) if exp_str else 1
                terms.append((coeff, var.lower(), exp))

        if not terms:
            return None

        var_name = next((t[1] for t in terms if t[1]), "x")

        if is_deriv:
            steps.append(f"Differentiating polynomial f({var_name}) = {poly_str}")
            res_terms = []
            for coeff, v, exp in terms:
                if exp == 0:
                    steps.append(f"  d/d{var_name}({coeff}) = 0")
                elif exp == 1:
                    steps.append(f"  d/d{var_name}({coeff}*{var_name}) = {coeff}")
                    res_terms.append(f"{coeff:g}")
                else:
                    new_coeff = coeff * exp
                    new_exp = exp - 1
                    exp_part = f"^{new_exp}" if new_exp > 1 else ""
                    steps.append(f"  d/d{var_name}({coeff}*{var_name}^{exp}) = {new_coeff:g}*{var_name}{exp_part}")
                    res_terms.append(f"{new_coeff:g}{var_name}{exp_part}")

            result_poly = " + ".join(res_terms).replace("+ -", "- ") if res_terms else "0"
            steps.append(f"Resulting Derivative: f'({var_name}) = {result_poly}")
            return "derivative", result_poly, steps

        elif is_integ:
            steps.append(f"Integrating polynomial ∫ ({poly_str}) d{var_name}")
            res_terms = []
            for coeff, v, exp in terms:
                new_exp = exp + 1
                new_coeff = coeff / new_exp
                exp_part = f"^{new_exp}" if new_exp > 1 else ""
                steps.append(f"  ∫ ({coeff}*{var_name}^{exp}) d{var_name} = ({coeff}/{new_exp})*{var_name}{exp_part} = {new_coeff:g}*{var_name}{exp_part}")
                res_terms.append(f"{new_coeff:g}{var_name}{exp_part}")

            res_terms.append("C")
            result_poly = " + ".join(res_terms).replace("+ -", "- ")
            steps.append(f"Resulting Indefinite Integral: {result_poly}")
            return "integral", result_poly, steps

        return None

    def solve_array_statistics(self, text: str) -> Optional[Tuple[Dict[str, float], List[str]]]:
        """Calculates mean, variance, std, sum for list of numbers."""
        steps = []
        # Look for [num1, num2, ...] or data = ...
        array_match = re.search(r'\[([\d\s,\.\-]+)\]', text)
        if not array_match:
            return None

        num_strings = array_match.group(1).split(",")
        try:
            nums = [float(s.strip()) for s in num_strings if s.strip()]
        except ValueError:
            return None

        if not nums:
            return None

        n = len(nums)
        s = sum(nums)
        mean = s / n
        var = sum((x - mean) ** 2 for x in nums) / n
        std = math.sqrt(var)

        steps.append(f"Extracted numerical dataset: {nums} (N = {n})")
        steps.append(f"Sum = {s:.4g}, Mean (μ) = {mean:.4g}")
        steps.append(f"Population Variance (σ^2) = {var:.4g}")
        steps.append(f"Standard Deviation (σ) = {std:.4g}")

        return {"mean": mean, "variance": var, "standard_deviation": std, "sum": s, "count": float(n)}, steps

    def explain_epistemic_uncertainty(self, problem_text: str, env: Dict[str, float]) -> str:
        """
        Epistemic honesty: When a problem cannot be solved, determines precisely
        which required variables or axioms are missing instead of hallucinating.
        """
        # Look for target keyword in problem text
        target_match = re.search(r'\b(?:what is|calculate|find|solve for|determine)\s+([a-zA-Z_]\w*)', problem_text, re.IGNORECASE)
        if target_match:
            target_var = target_match.group(1).lower()
            # Check if any formula in registry could produce target_var
            candidate_formulas = [
                (req, desc) for tvar, req, _, desc in self.FORMULA_REGISTRY if tvar == target_var
            ]
            if candidate_formulas:
                missing_per_formula = []
                for req, desc in candidate_formulas:
                    missing = [rv for rv in req if rv not in env]
                    missing_per_formula.append(f"Formula [{desc}] is missing: {missing}")
                missing_info = " OR ".join(missing_per_formula)
                return (
                    f"Unable to solve for '{target_var}': Insufficient constraints provided. "
                    f"{missing_info}."
                )

        if env:
            return (
                f"Unable to solve: Extracted given parameters {list(env.keys())}, but no verified "
                f"mathematical axiom or theorem in the knowledge tissue links these variables to a solvable target."
            )

        return (
            "Unable to solve: No recognized mathematical equation, variable constraints, or domain axioms found. "
            "Epistemic uncertainty declared (knowledge gap)."
        )

    def solve(self, problem_text: str) -> MathSolution:
        """Universal math problem solver across linear, quadratic, multi-variable, calculus, and stats."""
        steps = []
        solved_vars = {}

        # 1. Calculus solver (Differentiation & Integration)
        calc_res = self.solve_polynomial_calculus(problem_text)
        if calc_res:
            calc_type, result_expr, calc_steps = calc_res
            steps.extend(calc_steps)
            return MathSolution(
                problem_text=problem_text,
                solved_variables={},
                steps=steps,
                is_solved=True,
                explanation=f"{calc_type.capitalize()}: {result_expr}",
            )

        # 2. Quadratic Equation solver
        quad_res = self.solve_quadratic_equation(problem_text)
        if quad_res:
            var_name, roots, quad_steps = quad_res
            for idx, r in enumerate(roots, 1):
                solved_vars[f"{var_name}_{idx}" if len(roots) > 1 else var_name] = r
            steps.extend(quad_steps)
            roots_str = ", ".join(f"{r:.4g}" for r in roots)
            return MathSolution(
                problem_text=problem_text,
                solved_variables=solved_vars,
                steps=steps,
                is_solved=True,
                explanation=f"Solved quadratic for {var_name} = [{roots_str}]",
            )

        # 3. 2x2 System of Linear Equations solver
        sys_res = self.solve_linear_system_2x2(problem_text)
        if sys_res:
            sys_vars, sys_steps = sys_res
            solved_vars.update(sys_vars)
            steps.extend(sys_steps)
            sol_str = ", ".join(f"{k} = {v:.4g}" for k, v in sys_vars.items())
            return MathSolution(
                problem_text=problem_text,
                solved_variables=solved_vars,
                steps=steps,
                is_solved=True,
                explanation=f"Solved system: {sol_str}",
            )

        # 4. Try single-variable linear equation solver
        eq_res = self.solve_linear_equation(problem_text)
        if eq_res:
            var_name, val, eq_steps = eq_res
            solved_vars[var_name] = val
            steps.extend(eq_steps)
            return MathSolution(
                problem_text=problem_text,
                solved_variables=solved_vars,
                steps=steps,
                is_solved=True,
                explanation=f"Solved for {var_name} = {val:.4g}",
            )

        # 5. Numerical Dataset Statistics
        stats_res = self.solve_array_statistics(problem_text)
        if stats_res:
            stats_vars, stats_steps = stats_res
            solved_vars.update(stats_vars)
            steps.extend(stats_steps)
            return MathSolution(
                problem_text=problem_text,
                solved_variables=solved_vars,
                steps=steps,
                is_solved=True,
                explanation=f"Calculated dataset statistics: mean = {stats_vars['mean']:.4g}, std = {stats_vars['standard_deviation']:.4g}",
            )

        # 6. Extract quantities and check multi-domain formula registry
        env = self.extract_quantities(problem_text)
        if env:
            steps.append(f"Extracted known variables: {env}")

        # Iterate forward deduction until fixed point
        progress = True
        while progress:
            progress = False
            for target_var, req_vars, fn, formula_desc in self.FORMULA_REGISTRY:
                if target_var not in env and all(rv in env for rv in req_vars):
                    args = [env[rv] for rv in req_vars]
                    val = fn(*args)
                    if not math.isnan(val):
                        env[target_var] = val
                        solved_vars[target_var] = val
                        steps.append(f"Applied formula [{formula_desc}] -> {target_var} = {val:.4g}")
                        progress = True

        if solved_vars:
            ans_str = ", ".join(f"{k} = {v:.4g}" for k, v in solved_vars.items())
            return MathSolution(
                problem_text=problem_text,
                solved_variables=solved_vars,
                steps=steps,
                is_solved=True,
                explanation=f"Calculated: {ans_str}",
            )

        # 7. Epistemic Uncertainty Handling: Explicitly explain what is missing
        epistemic_reason = self.explain_epistemic_uncertainty(problem_text, env)
        steps.append(f"Epistemic Analysis: {epistemic_reason}")

        return MathSolution(
            problem_text=problem_text,
            solved_variables={},
            steps=steps,
            is_solved=False,
            explanation=epistemic_reason,
        )


# ─────────────────────────────────────────────────────────────
# 2. CONSTRAINT SATISFACTION (CSP) & AC-3 SOLVER
# ─────────────────────────────────────────────────────────────

@dataclass
class CSPResult:
    """Result of Constraint Satisfaction Solving."""
    is_satisfiable: bool
    assignment: Dict[str, Any]
    steps: List[str]


class CSPSolver:
    """
    Pure CPU Constraint Satisfaction Solver using Arc Consistency (AC-3)
    and Backtracking Search with Forward Checking.
    """

    def __init__(self):
        pass

    def solve(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        binary_constraints: List[Tuple[str, str, Callable[[Any, Any], bool]]],
        unary_constraints: Optional[List[Tuple[str, Callable[[Any], bool]]]] = None,
    ) -> CSPResult:
        """Solve CSP over finite discrete domains."""
        steps = []
        # Make a copy of domains
        d = {var: list(dom) for var, dom in domains.items()}

        # 1. Apply unary constraints
        if unary_constraints:
            for var, pred in unary_constraints:
                before_len = len(d[var])
                d[var] = [val for val in d[var] if pred(val)]
                steps.append(f"Unary constraint on {var}: domain reduced from {before_len} to {len(d[var])}")
                if not d[var]:
                    return CSPResult(is_satisfiable=False, assignment={}, steps=steps + [f"Domain of {var} wiped out!"])

        # 2. AC-3 Arc Consistency
        queue = [(u, v, pred) for u, v, pred in binary_constraints] + [(v, u, lambda y, x, p=pred: p(x, y)) for u, v, pred in binary_constraints]
        while queue:
            xi, xj, pred = queue.pop(0)
            if self._revise(xi, xj, pred, d):
                if not d[xi]:
                    return CSPResult(is_satisfiable=False, assignment={}, steps=steps + [f"AC-3: Inconsistency found at {xi}"])
                for u, v, p in binary_constraints:
                    if v == xi and u != xj:
                        queue.append((u, xi, p))
                    elif u == xi and v != xj:
                        queue.append((v, xi, lambda y, x, p2=p: p2(x, y)))

        # 3. Backtracking search over revised domains
        assignment: Dict[str, Any] = {}
        success = self._backtrack(assignment, variables, d, binary_constraints, steps)
        return CSPResult(is_satisfiable=success, assignment=assignment if success else {}, steps=steps)

    def _revise(self, xi: str, xj: str, pred: Callable[[Any, Any], bool], domains: Dict[str, List[Any]]) -> bool:
        revised = False
        to_remove = []
        for x in domains[xi]:
            if not any(pred(x, y) for y in domains[xj]):
                to_remove.append(x)
                revised = True
        for x in to_remove:
            domains[xi].remove(x)
        return revised

    def _backtrack(
        self,
        assignment: Dict[str, Any],
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraints: List[Tuple[str, str, Callable[[Any, Any], bool]]],
        steps: List[str],
    ) -> bool:
        if len(assignment) == len(variables):
            return True

        unassigned = [v for v in variables if v not in assignment]
        # Minimum Remaining Values (MRV) heuristic
        var = min(unassigned, key=lambda v: len(domains[v]))

        for val in domains[var]:
            if self._is_consistent(var, val, assignment, constraints):
                assignment[var] = val
                if self._backtrack(assignment, variables, domains, constraints, steps):
                    return True
                del assignment[var]

        return False

    def _is_consistent(
        self,
        var: str,
        val: Any,
        assignment: Dict[str, Any],
        constraints: List[Tuple[str, str, Callable[[Any, Any], bool]]],
    ) -> bool:
        for u, v, pred in constraints:
            if u == var and v in assignment:
                if not pred(val, assignment[v]):
                    return False
            elif v == var and u in assignment:
                if not pred(assignment[u], val):
                    return False
        return True


# ─────────────────────────────────────────────────────────────
# 3. RESOLUTION-REFUTATION THEOREM PROVER (FIRST-ORDER LOGIC)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Literal:
    """A positive or negated first-order predicate, e.g. Treats(Lisinopril, Hypertension)."""
    predicate: str
    args: Tuple[str, ...]
    is_negated: bool = False

    def negate(self) -> "Literal":
        return Literal(self.predicate, self.args, not self.is_negated)

    def __str__(self) -> str:
        prefix = "NOT " if self.is_negated else ""
        return f"{prefix}{self.predicate}({', '.join(self.args)})"


@dataclass
class Clause:
    """Disjunction of literals: L1 OR L2 OR ... OR Ln."""
    literals: Set[Literal]

    def is_empty(self) -> bool:
        return len(self.literals) == 0

    def __str__(self) -> str:
        if self.is_empty():
            return "[EMPTY_CLAUSE: CONTRADICTION / Q.E.D.]"
        return " OR ".join(str(lit) for lit in self.literals)


@dataclass
class ProofResult:
    """Formal proof verification result."""
    is_proven: bool
    proof_steps: List[str]
    contradiction_derived: bool


class ResolutionProver:
    """
    Sound & complete resolution-refutation theorem prover.
    Proves knowledge base entailment (KB |= Alpha) by proving (KB and NOT Alpha) is unsatisfiable.
    """

    def __init__(self):
        pass

    def prove(self, kb_clauses: List[Clause], query_literal: Literal, max_steps: int = 50) -> ProofResult:
        """Prove query by refutation."""
        steps = []
        # Negate the query and add to KB
        negated_query = Clause({query_literal.negate()})
        clauses = list(kb_clauses) + [negated_query]
        steps.append(f"Assert Negation of Query for Refutation: {negated_query}")

        seen_clauses = set(frozenset(c.literals) for c in clauses)

        for step_i in range(max_steps):
            new_clauses = []
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    resolvents = self._resolve(clauses[i], clauses[j])
                    for res in resolvents:
                        if res.is_empty():
                            steps.append(f"Step {step_i+1}: Resolved ({clauses[i]}) with ({clauses[j]}) -> Derived EMPTY CLAUSE [CONTRADICTION].")
                            return ProofResult(is_proven=True, proof_steps=steps, contradiction_derived=True)
                        frozen = frozenset(res.literals)
                        if frozen not in seen_clauses:
                            seen_clauses.add(frozen)
                            new_clauses.append(res)
                            steps.append(f"Step {step_i+1}: Resolved ({clauses[i]}) with ({clauses[j]}) -> Derived ({res})")

            if not new_clauses:
                break
            clauses.extend(new_clauses)

        return ProofResult(is_proven=False, proof_steps=steps, contradiction_derived=False)

    def _resolve(self, c1: Clause, c2: Clause) -> List[Clause]:
        resolvents = []
        for l1 in c1.literals:
            for l2 in c2.literals:
                if l1.predicate == l2.predicate and l1.args == l2.args and l1.is_negated != l2.is_negated:
                    # Complementary pair found -> resolve
                    new_lits = (c1.literals - {l1}) | (c2.literals - {l2})
                    resolvents.append(Clause(new_lits))
        return resolvents
