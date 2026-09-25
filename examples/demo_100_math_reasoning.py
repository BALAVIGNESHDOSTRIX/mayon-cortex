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
Mayon-Cortex: 100 Math Problems Ingestion & Neuro-Symbolic Reasoning
=====================================================================
Comprehensive real-world demonstration:
1. Ingests 100 mathematical theorems, formulas, and structured word problems
   spanning Algebra, Geometry, Kinematics, Finance, Electricity, Energy, and Multi-Hop Logic.
2. Queries Mayon-Cortex to solve each problem with 100% deterministic proofs.
3. Displays step-by-step algebraic deductions, latency metrics, and verification scorecards.
4. Supports live interactive math querying.

Run with:
    & "x:/BalaDev/codex-net/pyenv/Scripts/python.exe" examples/demo_100_math_reasoning.py
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mayon_cortex.engine import MayonCortex
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.reasoning.smt_solver import MathSolution


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────
# 100 CURATED MATHEMATICAL PROBLEMS (7 CATEGORIES)
# ─────────────────────────────────────────────────────────────

MATH_DATASET_100: List[Dict[str, Any]] = [
    # ── Category 1: Linear Algebraic Equations (20 Problems) ──
    {"id": 1, "cat": "Algebra", "q": "If 3x + 5 = 20, what is x?", "target_var": "x", "expected": 5.0},
    {"id": 2, "cat": "Algebra", "q": "If 2x - 8 = 14, what is x?", "target_var": "x", "expected": 11.0},
    {"id": 3, "cat": "Algebra", "q": "If 5y + 15 = 45, what is y?", "target_var": "y", "expected": 6.0},
    {"id": 4, "cat": "Algebra", "q": "If 4z - 12 = 28, what is z?", "target_var": "z", "expected": 10.0},
    {"id": 5, "cat": "Algebra", "q": "If 7a + 14 = 49, what is a?", "target_var": "a", "expected": 5.0},
    {"id": 6, "cat": "Algebra", "q": "If 6b - 18 = 30, what is b?", "target_var": "b", "expected": 8.0},
    {"id": 7, "cat": "Algebra", "q": "If 8c + 24 = 64, what is c?", "target_var": "c", "expected": 5.0},
    {"id": 8, "cat": "Algebra", "q": "If 9x - 27 = 54, what is x?", "target_var": "x", "expected": 9.0},
    {"id": 9, "cat": "Algebra", "q": "If 10y + 50 = 120, what is y?", "target_var": "y", "expected": 7.0},
    {"id": 10, "cat": "Algebra", "q": "If 12z - 36 = 60, what is z?", "target_var": "z", "expected": 8.0},
    {"id": 11, "cat": "Algebra", "q": "If 2x + 10 = 30, what is x?", "target_var": "x", "expected": 10.0},
    {"id": 12, "cat": "Algebra", "q": "If 3a - 9 = 21, what is a?", "target_var": "a", "expected": 10.0},
    {"id": 13, "cat": "Algebra", "q": "If 4b + 16 = 56, what is b?", "target_var": "b", "expected": 10.0},
    {"id": 14, "cat": "Algebra", "q": "If 5c - 25 = 50, what is c?", "target_var": "c", "expected": 15.0},
    {"id": 15, "cat": "Algebra", "q": "If 6x + 30 = 90, what is x?", "target_var": "x", "expected": 10.0},
    {"id": 16, "cat": "Algebra", "q": "If 7y - 21 = 49, what is y?", "target_var": "y", "expected": 10.0},
    {"id": 17, "cat": "Algebra", "q": "If 8z + 40 = 120, what is z?", "target_var": "z", "expected": 10.0},
    {"id": 18, "cat": "Algebra", "q": "If 9a - 45 = 45, what is a?", "target_var": "a", "expected": 10.0},
    {"id": 19, "cat": "Algebra", "q": "If 10b + 100 = 250, what is b?", "target_var": "b", "expected": 15.0},
    {"id": 20, "cat": "Algebra", "q": "If 15x - 45 = 90, what is x?", "target_var": "x", "expected": 9.0},

    # ── Category 2: Kinematics & Physics Motion (15 Problems) ──
    {"id": 21, "cat": "Kinematics", "q": "Given distance = 100 km and time = 2 hours, calculate speed.", "target_var": "speed", "expected": 50.0},
    {"id": 22, "cat": "Kinematics", "q": "Given distance = 300 km and time = 5 hours, calculate speed.", "target_var": "speed", "expected": 60.0},
    {"id": 23, "cat": "Kinematics", "q": "Given speed = 80 km/h and time = 4 hours, calculate distance.", "target_var": "distance", "expected": 320.0},
    {"id": 24, "cat": "Kinematics", "q": "Given speed = 120 km/h and time = 2.5 hours, calculate distance.", "target_var": "distance", "expected": 300.0},
    {"id": 25, "cat": "Kinematics", "q": "Given distance = 450 km and speed = 90 km/h, calculate time.", "target_var": "time", "expected": 5.0},
    {"id": 26, "cat": "Kinematics", "q": "Given distance = 600 km and speed = 75 km/h, calculate time.", "target_var": "time", "expected": 8.0},
    {"id": 27, "cat": "Kinematics", "q": "Given mass = 50 kg and acceleration = 9.8 m/s^2, find force.", "target_var": "force", "expected": 490.0},
    {"id": 28, "cat": "Kinematics", "q": "Given mass = 1200 kg and acceleration = 3.5 m/s^2, find force.", "target_var": "force", "expected": 4200.0},
    {"id": 29, "cat": "Kinematics", "q": "Given force = 500 N and acceleration = 10 m/s^2, find mass.", "target_var": "mass", "expected": 50.0},
    {"id": 30, "cat": "Kinematics", "q": "Given force = 2400 N and mass = 800 kg, find acceleration.", "target_var": "acceleration", "expected": 3.0},
    {"id": 31, "cat": "Kinematics", "q": "Given distance = 150 km and time = 3 hours, calculate speed.", "target_var": "speed", "expected": 50.0},
    {"id": 32, "cat": "Kinematics", "q": "Given speed = 65 km/h and time = 6 hours, calculate distance.", "target_var": "distance", "expected": 390.0},
    {"id": 33, "cat": "Kinematics", "q": "Given distance = 800 km and speed = 100 km/h, calculate time.", "target_var": "time", "expected": 8.0},
    {"id": 34, "cat": "Kinematics", "q": "Given mass = 75 kg and acceleration = 4.0 m/s^2, find force.", "target_var": "force", "expected": 300.0},
    {"id": 35, "cat": "Kinematics", "q": "Given force = 1500 N and acceleration = 5.0 m/s^2, find mass.", "target_var": "mass", "expected": 300.0},

    # ── Category 3: Financial, Discounts & Profit (15 Problems) ──
    {"id": 36, "cat": "Finance", "q": "If price = $80 and discount = 20%, calculate final_price.", "target_var": "final_price", "expected": 64.0},
    {"id": 37, "cat": "Finance", "q": "If price = $150 and discount = 10%, calculate final_price.", "target_var": "final_price", "expected": 135.0},
    {"id": 38, "cat": "Finance", "q": "If price = $200 and discount = 25%, calculate final_price.", "target_var": "final_price", "expected": 150.0},
    {"id": 39, "cat": "Finance", "q": "If price = $500 and discount = 30%, calculate final_price.", "target_var": "final_price", "expected": 350.0},
    {"id": 40, "cat": "Finance", "q": "If revenue = $10000 and cost = $6500, calculate profit.", "target_var": "profit", "expected": 3500.0},
    {"id": 41, "cat": "Finance", "q": "If revenue = $250000 and cost = $180000, calculate profit.", "target_var": "profit", "expected": 70000.0},
    {"id": 42, "cat": "Finance", "q": "If amount = $1200 and tax_rate = 8%, calculate tax.", "target_var": "tax", "expected": 96.0},
    {"id": 43, "cat": "Finance", "q": "If amount = $5000 and tax_rate = 15%, calculate tax.", "target_var": "tax", "expected": 750.0},
    {"id": 44, "cat": "Finance", "q": "If principal = $1000, rate = 5%, and time = 3 years, calculate interest.", "target_var": "interest", "expected": 150.0},
    {"id": 45, "cat": "Finance", "q": "If principal = $5000, rate = 6%, and time = 4 years, calculate interest.", "target_var": "interest", "expected": 1200.0},
    {"id": 46, "cat": "Finance", "q": "If price = $40 and discount = 15%, calculate final_price.", "target_var": "final_price", "expected": 34.0},
    {"id": 47, "cat": "Finance", "q": "If revenue = $80000 and cost = $52000, calculate profit.", "target_var": "profit", "expected": 28000.0},
    {"id": 48, "cat": "Finance", "q": "If amount = $3500 and tax_rate = 10%, calculate tax.", "target_var": "tax", "expected": 350.0},
    {"id": 49, "cat": "Finance", "q": "If principal = $20000, rate = 4%, and time = 5 years, calculate interest.", "target_var": "interest", "expected": 4000.0},
    {"id": 50, "cat": "Finance", "q": "If price = $1200 and discount = 50%, calculate final_price.", "target_var": "final_price", "expected": 600.0},

    # ── Category 4: Ohm's Law & Circuit Electricity (10 Problems) ──
    {"id": 51, "cat": "Electricity", "q": "Given current = 5 A and resistance = 20 ohms, find voltage.", "target_var": "voltage", "expected": 100.0},
    {"id": 52, "cat": "Electricity", "q": "Given current = 2.5 A and resistance = 48 ohms, find voltage.", "target_var": "voltage", "expected": 120.0},
    {"id": 53, "cat": "Electricity", "q": "Given voltage = 230 V and resistance = 46 ohms, find current.", "target_var": "current", "expected": 5.0},
    {"id": 54, "cat": "Electricity", "q": "Given voltage = 12 V and current = 3 A, find resistance.", "target_var": "resistance", "expected": 4.0},
    {"id": 55, "cat": "Electricity", "q": "Given voltage = 240 V and current = 10 A, find power.", "target_var": "power", "expected": 2400.0},
    {"id": 56, "cat": "Electricity", "q": "Given voltage = 120 V and current = 2.0 A, find power.", "target_var": "power", "expected": 240.0},
    {"id": 57, "cat": "Electricity", "q": "Given current = 8 A and resistance = 15 ohms, find voltage.", "target_var": "voltage", "expected": 120.0},
    {"id": 58, "cat": "Electricity", "q": "Given voltage = 48 V and resistance = 12 ohms, find current.", "target_var": "current", "expected": 4.0},
    {"id": 59, "cat": "Electricity", "q": "Given voltage = 24 V and current = 6 A, find resistance.", "target_var": "resistance", "expected": 4.0},
    {"id": 60, "cat": "Electricity", "q": "Given voltage = 12 V and current = 0.5 A, find power.", "target_var": "power", "expected": 6.0},

    # ── Category 5: Geometry, Areas, Perimeters & Volumes (15 Problems) ──
    {"id": 61, "cat": "Geometry", "q": "Given width = 12 m and height = 8 m, calculate area.", "target_var": "area", "expected": 96.0},
    {"id": 62, "cat": "Geometry", "q": "Given width = 25 cm and height = 14 cm, calculate area.", "target_var": "area", "expected": 350.0},
    {"id": 63, "cat": "Geometry", "q": "Given base = 16 m and height = 10 m, calculate area.", "target_var": "area", "expected": 80.0},
    {"id": 64, "cat": "Geometry", "q": "Given base = 30 cm and height = 12 cm, calculate area.", "target_var": "area", "expected": 180.0},
    {"id": 65, "cat": "Geometry", "q": "Given radius = 7 m, calculate area.", "target_var": "area", "expected": 153.938},
    {"id": 66, "cat": "Geometry", "q": "Given radius = 10 cm, calculate area.", "target_var": "area", "expected": 314.159},
    {"id": 67, "cat": "Geometry", "q": "Given width = 15 m and height = 20 m, calculate perimeter.", "target_var": "perimeter", "expected": 70.0},
    {"id": 68, "cat": "Geometry", "q": "Given width = 50 cm and height = 30 cm, calculate perimeter.", "target_var": "perimeter", "expected": 160.0},
    {"id": 69, "cat": "Geometry", "q": "Given radius = 5 m, calculate circumference.", "target_var": "circumference", "expected": 31.4159},
    {"id": 70, "cat": "Geometry", "q": "Given radius = 14 cm, calculate circumference.", "target_var": "circumference", "expected": 87.9645},
    {"id": 71, "cat": "Geometry", "q": "Given area = 50 m^2 and height = 6 m, calculate volume.", "target_var": "volume", "expected": 300.0},
    {"id": 72, "cat": "Geometry", "q": "Given length = 8 m, width = 5 m, and height = 4 m, calculate volume.", "target_var": "volume", "expected": 160.0},
    {"id": 73, "cat": "Geometry", "q": "Given length = 10 cm, width = 6 cm, and height = 3 cm, calculate volume.", "target_var": "volume", "expected": 180.0},
    {"id": 74, "cat": "Geometry", "q": "Given width = 40 m and height = 25 m, calculate area.", "target_var": "area", "expected": 1000.0},
    {"id": 75, "cat": "Geometry", "q": "Given base = 22 m and height = 14 m, calculate area.", "target_var": "area", "expected": 154.0},

    # ── Category 6: Energy, Work, Power & Density (10 Problems) ──
    {"id": 76, "cat": "Energy", "q": "Given force = 200 N and distance = 15 m, calculate work.", "target_var": "work", "expected": 3000.0},
    {"id": 77, "cat": "Energy", "q": "Given force = 450 N and distance = 8 m, calculate work.", "target_var": "work", "expected": 3600.0},
    {"id": 78, "cat": "Energy", "q": "Given work = 6000 J and time = 12 s, calculate power.", "target_var": "power", "expected": 500.0},
    {"id": 79, "cat": "Energy", "q": "Given work = 15000 J and time = 30 s, calculate power.", "target_var": "power", "expected": 500.0},
    {"id": 80, "cat": "Energy", "q": "Given mass = 1000 kg and velocity = 20 m/s, calculate kinetic_energy.", "target_var": "kinetic_energy", "expected": 200000.0},
    {"id": 81, "cat": "Energy", "q": "Given mass = 80 kg and velocity = 15 m/s, calculate kinetic_energy.", "target_var": "kinetic_energy", "expected": 9000.0},
    {"id": 82, "cat": "Energy", "q": "Given mass = 50 kg and height = 10 m, calculate potential_energy.", "target_var": "potential_energy", "expected": 4900.0},
    {"id": 83, "cat": "Energy", "q": "Given mass = 500 kg and volume = 2 m^3, calculate density.", "target_var": "density", "expected": 250.0},
    {"id": 84, "cat": "Energy", "q": "Given density = 800 kg/m^3 and volume = 5 m^3, calculate mass.", "target_var": "mass", "expected": 4000.0},
    {"id": 85, "cat": "Energy", "q": "Given mass = 1500 kg and density = 300 kg/m^3, calculate volume.", "target_var": "volume", "expected": 5.0},

    # ── Category 7: Multi-Hop Mathematical & Causal Chains (15 Problems) ──
    {"id": 86, "cat": "Multi-Hop", "q": "Given distance = 240 km, time = 4 hours, and speed = distance / time, calculate speed.", "target_var": "speed", "expected": 60.0},
    {"id": 87, "cat": "Multi-Hop", "q": "Given mass = 100 kg, acceleration = 9.8 m/s^2, and force = mass * acceleration, calculate force.", "target_var": "force", "expected": 980.0},
    {"id": 88, "cat": "Multi-Hop", "q": "Given force = 500 N, distance = 20 m, calculate work.", "target_var": "work", "expected": 10000.0},
    {"id": 89, "cat": "Multi-Hop", "q": "If price = $300 and discount = 10%, calculate final_price.", "target_var": "final_price", "expected": 270.0},
    {"id": 90, "cat": "Multi-Hop", "q": "Given width = 18 m and height = 5 m, calculate area.", "target_var": "area", "expected": 90.0},
    {"id": 91, "cat": "Multi-Hop", "q": "If 8x - 16 = 48, what is x?", "target_var": "x", "expected": 8.0},
    {"id": 92, "cat": "Multi-Hop", "q": "Given current = 4 A and resistance = 30 ohms, find voltage.", "target_var": "voltage", "expected": 120.0},
    {"id": 93, "cat": "Multi-Hop", "q": "If revenue = $50000 and cost = $32000, calculate profit.", "target_var": "profit", "expected": 18000.0},
    {"id": 94, "cat": "Multi-Hop", "q": "Given mass = 2000 kg and velocity = 10 m/s, calculate kinetic_energy.", "target_var": "kinetic_energy", "expected": 100000.0},
    {"id": 95, "cat": "Multi-Hop", "q": "Given width = 10 m and height = 25 m, calculate perimeter.", "target_var": "perimeter", "expected": 70.0},
    {"id": 96, "cat": "Multi-Hop", "q": "Given distance = 720 km and speed = 80 km/h, calculate time.", "target_var": "time", "expected": 9.0},
    {"id": 97, "cat": "Multi-Hop", "q": "If 11y + 33 = 99, what is y?", "target_var": "y", "expected": 6.0},
    {"id": 98, "cat": "Multi-Hop", "q": "If principal = $15000, rate = 8%, and time = 2 years, calculate interest.", "target_var": "interest", "expected": 2400.0},
    {"id": 99, "cat": "Multi-Hop", "q": "Given density = 1200 kg/m^3 and volume = 3 m^3, calculate mass.", "target_var": "mass", "expected": 3600.0},
    {"id": 100, "cat": "Multi-Hop", "q": "If 20z - 100 = 300, what is z?", "target_var": "z", "expected": 20.0},
]


def main():
    print_banner("Mayon-Cortex: 100 Math Problems Ingestion & Reasoning Engine")
    print("[*] Initializing Pure CPU Neuro-Symbolic Cognitive Brain...")
    
    t_init = time.perf_counter()
    config = CortexConfig()
    cortex = MayonCortex(config)
    print(f"[OK] Brain initialized in {(time.perf_counter() - t_init)*1000:.2f} ms on CPU.")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: INGEST MATHEMATICAL KNOWLEDGE & RULES INTO GRAPH
    # ─────────────────────────────────────────────────────────────
    print_banner("1. Ingesting Mathematical Knowledge Base into MayonGraph")
    
    math_knowledge_base = [
        "Linear algebra defines relationship ax + b = c for unknown variable x.",
        "Kinematics specifies speed equals distance divided by elapsed time.",
        "Newtonian mechanics proves force equals mass multiplied by acceleration.",
        "Electrical Ohm's law defines voltage equals electric current multiplied by resistance.",
        "Electrical power equals voltage multiplied by electric current.",
        "Work done equals applied mechanical force multiplied by displacement distance.",
        "Mechanical power equals work done divided by elapsed time.",
        "Kinetic energy equals one half mass multiplied by velocity squared.",
        "Gravitational potential energy equals mass multiplied by gravity 9.8 and height.",
        "Material density equals object mass divided by spatial volume.",
        "Geometry defines rectangle area equals width multiplied by height.",
        "Geometry defines triangle area equals one half base multiplied by height.",
        "Geometry defines circle area equals pi multiplied by radius squared.",
        "Geometry defines rectangle perimeter equals two times width plus height.",
        "Financial economics defines final price equals price times one minus discount rate.",
        "Financial accounting defines profit equals total revenue minus total cost.",
        "Tax calculation defines tax amount equals taxable total multiplied by tax rate.",
        "Financial banking defines simple interest equals principal times annual rate times time.",
    ]

    t0 = time.perf_counter()
    total_edges_added = 0
    for fact in math_knowledge_base:
        edges = cortex.ingest_hyper(fact, sector="math", provenance="math_knowledge_base")
        total_edges_added += len(edges)
    dt_ingest = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Ingested {len(math_knowledge_base)} foundational mathematical axioms.")
    print(f"[+] Graph Tissue Status: {cortex.graph.num_nodes} Concept Nodes, {len(cortex.graph.edges)} Relational Edges.")
    print(f"[+] Ingestion completed in {dt_ingest:.2f} ms on CPU.\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: SOLVE & REASON ACROSS ALL 100 PROBLEMS
    # ─────────────────────────────────────────────────────────────
    print_banner("2. Executing Automated Reasoning Across 100 Math Problems")

    passed_count = 0
    total_time_ms = 0.0
    category_stats: Dict[str, Dict[str, Any]] = {}

    print(f"{'ID':<4} | {'Category':<12} | {'Problem Preview':<40} | {'Answer Derived':<15} | {'Status':<6}")
    print("-" * 88)

    for item in MATH_DATASET_100:
        p_id = item["id"]
        cat = item["cat"]
        q = item["q"]
        target = item["target_var"]
        expected = item["expected"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "latencies": []}
        category_stats[cat]["total"] += 1

        t_start = time.perf_counter()
        solution = cortex.solve(q)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        total_time_ms += lat_ms
        category_stats[cat]["latencies"].append(lat_ms)

        # Verification check
        is_correct = False
        derived_str = "None"
        if isinstance(solution, MathSolution) and solution.is_solved:
            if target in solution.solved_variables:
                derived_val = solution.solved_variables[target]
                derived_str = f"{derived_val:.4g}"
                if np.isclose(derived_val, expected, rtol=1e-2, atol=1e-2):
                    is_correct = True
        
        status_str = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed_count += 1
            category_stats[cat]["passed"] += 1

        preview = (q[:38] + "..") if len(q) > 40 else q
        print(f"#{p_id:<3} | {cat:<12} | {preview:<40} | {derived_str:<15} | {status_str:<6}")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: DETAILED STEP-BY-STEP REASONING BREAKDOWN
    # ─────────────────────────────────────────────────────────────
    print_banner("3. Sample Mathematical Step-by-Step Proof Traces")
    
    sample_ids = [1, 21, 36, 51, 61, 76, 86]
    for sid in sample_ids:
        item = MATH_DATASET_100[sid - 1]
        res = cortex.solve(item["q"])
        print(f"\n[Problem #{item['id']} - {item['cat']}] \"{item['q']}\"")
        print(f"  • Explanation: {res.explanation}")
        print("  • Formal Deductive Proof Steps:")
        for idx, step in enumerate(res.steps, 1):
            print(f"    Step {idx}: {step}")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: CATEGORY BREAKDOWN & FINAL SCORECARD
    # ─────────────────────────────────────────────────────────────
    print_banner("4. 100-Problem Mathematical Benchmark Scorecard")
    print(f"{'Category':<15} | {'Passed':<10} | {'Success Rate':<14} | {'Avg Latency (ms)':<16}")
    print("-" * 65)
    for cat, data in category_stats.items():
        rate = (data["passed"] / data["total"]) * 100.0
        avg_lat = np.mean(data["latencies"])
        print(f"{cat:<15} | {data['passed']}/{data['total']:<8} | {rate:>10.1f} % | {avg_lat:>12.2f} ms")
    
    print("=" * 65)
    overall_rate = (passed_count / len(MATH_DATASET_100)) * 100.0
    overall_avg_lat = total_time_ms / len(MATH_DATASET_100)
    print(f"{'TOTAL / OVERALL':<15} | {passed_count}/{len(MATH_DATASET_100):<8} | {overall_rate:>10.1f} % | {overall_avg_lat:>12.2f} ms")
    print("=" * 65)
    print(f"  FINAL GRADE: {overall_rate:.1f}% -- 100% DETERMINISTIC CPU EXECUTION (Zero GPU)")
    print("=" * 65 + "\n")
    print_banner("5. Arbitrary & Unscripted Problem Reasoning (Zero Template)")
    unseen_query = "If 47.5x - 12.3 = 98.7, what is x?"
    res1 = cortex.solve(unseen_query)
    print(f"\n[Query]: \"{unseen_query}\"")
    print(res1.format_proof())


if __name__ == "__main__":
    main()
