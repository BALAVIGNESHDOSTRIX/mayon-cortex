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
Number Theory Engine — Primes, Modular Arithmetic, Diophantine & Discrete Math
==============================================================================
Pure Python / NumPy number theory and discrete mathematics solver.

Features:
1. Deterministic Miller-Rabin Primality Test
2. Pollard's Rho Prime Factorization
3. Extended GCD & Modular Inverse
4. Chinese Remainder Theorem (CRT)
5. Euler's Totient Function phi(n)
6. Discrete Logarithm (Baby-step Giant-step)
7. Linear Diophantine Equation Solver
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class NumberTheoryEngine:
    """
    Mathematical number theory reasoning engine.
    """

    @staticmethod
    def is_prime(n: int) -> bool:
        """Deterministic Miller-Rabin primality test for n < 3.3 x 10^14."""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            return False

        # Write n - 1 as 2^s * d
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1

        # Test bases
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in bases:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            composite = True
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    composite = False
                    break
            if composite:
                return False
        return True

    @classmethod
    def factorize(cls, n: int) -> Dict[int, int]:
        """Prime factorization returning {prime_factor: exponent}."""
        if n <= 1:
            return {}

        factors: Dict[int, int] = {}
        # Trial division for small primes
        d = 2
        while d * d <= n and d <= 100:
            if n % d == 0:
                count = 0
                while n % d == 0:
                    count += 1
                    n //= d
                factors[d] = count
            d += 1 if d == 2 else 2

        if n == 1:
            return factors

        if cls.is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            return factors

        # Pollard's rho for remaining composite parts
        def pollard_rho(val: int) -> int:
            if val % 2 == 0:
                return 2
            x = 2
            y = 2
            d = 1
            c = 1
            f = lambda x_val: (pow(x_val, 2, val) + c) % val
            while d == 1:
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), val)
            return d

        sub_factor = pollard_rho(n)
        while sub_factor == n:
            sub_factor = pollard_rho(n)

        f1 = cls.factorize(sub_factor)
        f2 = cls.factorize(n // sub_factor)
        for p, exp in f1.items():
            factors[p] = factors.get(p, 0) + exp
        for p, exp in f2.items():
            factors[p] = factors.get(p, 0) + exp

        return dict(sorted(factors.items()))

    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm: returns (gcd, x, y) such that a*x + b*y = gcd.
        """
        if a == 0:
            return b, 0, 1
        g, x1, y1 = NumberTheoryEngine.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return g, x, y

    @classmethod
    def mod_inverse(cls, a: int, m: int) -> Optional[int]:
        """Compute modular multiplicative inverse a^-1 mod m."""
        g, x, _ = cls.extended_gcd(a % m, m)
        if g != 1:
            return None  # Inverse does not exist
        return (x % m + m) % m

    @classmethod
    def chinese_remainder(cls, remainders: List[int], moduli: List[int]) -> Optional[int]:
        """
        Chinese Remainder Theorem: Solves x = r_i (mod m_i) for pairwise coprime moduli.
        """
        if len(remainders) != len(moduli):
            return None

        total_prod = 1
        for m in moduli:
            total_prod *= m

        x = 0
        for r_i, m_i in zip(remainders, moduli):
            n_i = total_prod // m_i
            inv = cls.mod_inverse(n_i, m_i)
            if inv is None:
                return None
            x = (x + r_i * n_i * inv) % total_prod

        return (x % total_prod + total_prod) % total_prod

    @classmethod
    def euler_totient(cls, n: int) -> int:
        """Euler's Totient function phi(n) = count of coprime integers <= n."""
        if n <= 0:
            return 0
        factors = cls.factorize(n)
        result = n
        for p in factors:
            result = (result // p) * (p - 1)
        return result

    @classmethod
    def solve_diophantine(cls, a: int, b: int, c: int) -> Optional[Dict[str, Any]]:
        """
        Solve linear Diophantine equation a*x + b*y = c.
        Returns particular solution (x0, y0) and parameter steps (dx, dy).
        """
        g, x0, y0 = cls.extended_gcd(abs(a), abs(b))
        if c % g != 0:
            return None  # No integer solution

        scale = c // g
        x_part = x0 * scale * (1 if a >= 0 else -1)
        y_part = y0 * scale * (1 if b >= 0 else -1)

        dx = b // g
        dy = -a // g

        return {
            "gcd": g,
            "x0": x_part,
            "y0": y_part,
            "dx": dx,
            "dy": dy,
            "general_x": f"{x_part} + {dx}*k",
            "general_y": f"{y_part} + {dy}*k",
        }
