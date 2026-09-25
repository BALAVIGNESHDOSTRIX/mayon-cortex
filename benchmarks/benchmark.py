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
Cortex-Graph Benchmark Suite — Third Paradigm Performance Profiler
===================================================================
Phase 8 of the Brain-Like Intelligence upgrade.

Runs automated performance, latency, throughput, and accuracy benchmarks across
all subsystems of Cortex-Graph on pure CPU:
  - Wavefront Activation & Proof Path Ranking throughput
  - Hyperdimensional Computing (HDC) binary algebra speed
  - System 2 Thinking & Logic Engine reasoning latency
  - Problem Solver deduction speed
  - Energy Canvas (MRF) image synthesis latency
  - Morphogenetic pattern generation throughput
  - Case-Based Precedent Retrieval speed
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from mayon_cortex.cortex import CortexGraph
from mayon_cortex.memory.hyperdimensional import HyperVector, HyperdimensionalMemory
from mayon_cortex.perception.scene_graph import SceneGraph, VisualObject
from mayon_cortex.dynamics.energy_canvas import EnergyCanvas
from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator
from mayon_cortex.memory.case_memory import CaseMemory, PrecedentCase


@dataclass
class BenchmarkResult:
    """Individual benchmark metrics."""
    name: str
    operations_per_second: float
    avg_latency_ms: float
    status: str = "PASS"
    details: Dict[str, Any] = field(default_factory=dict)


class CortexBenchmarkSuite:
    """
    Comprehensive benchmark runner for Cortex-Graph.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_all(self, cortex: CortexGraph) -> List[BenchmarkResult]:
        """Execute all benchmark suites."""
        print("\n" + "=" * 65)
        print("  CORTEX-GRAPH v5 -- THIRD PARADIGM CPU BENCHMARK SUITE")
        print("=" * 65 + "\n")

        self.benchmark_hdc()
        self.benchmark_morphogenetic()
        self.benchmark_energy_canvas()
        self.benchmark_case_memory()
        self.benchmark_thinking(cortex)
        self.benchmark_problem_solver(cortex)
        self.benchmark_graph_reasoning(cortex)

        print("-" * 65)
        print(f"{'Benchmark':<35} | {'Latency (ms)':<14} | {'Ops/sec':<10}")
        print("-" * 65)
        for r in self.results:
            print(f"{r.name:<35} | {r.avg_latency_ms:>10.2f} ms | {r.operations_per_second:>10.1f}")
        print("=" * 65 + "\n")

        return self.results

    def benchmark_hdc(self, n_ops: int = 1000):
        """Benchmark 10,000-dimensional hypervector bit operations."""
        v1 = HyperVector(10000)
        v2 = HyperVector(10000)
        v3 = HyperVector(10000)

        t0 = time.perf_counter()
        for _ in range(n_ops):
            b = HyperVector.bind(v1, v2)
            bun = HyperVector.bundle(v1, v2, v3)
            p = HyperVector.permute(v1, 3)
            sim = v1.similarity(v2)
        dt = time.perf_counter() - t0

        avg_ms = (dt / (n_ops * 4)) * 1000.0
        ops_sec = (n_ops * 4) / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Hyperdimensional Bit Algebra (10k-dim)",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_morphogenetic(self, n_runs: int = 5):
        """Benchmark reaction-diffusion pattern generation."""
        gen = MorphogeneticGenerator()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            gen.grow_pattern(pattern_type="stripes", width=32, height=32, steps=200)
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_runs) * 1000.0
        ops_sec = n_runs / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Morphogenetic Growth (Reaction-Diff 32x32)",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_energy_canvas(self, n_runs: int = 3):
        """Benchmark MRF energy minimization image generation."""
        canvas = EnergyCanvas(width=32, height=32)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            canvas.generate(prompt="cat sitting in room", iterations=30)
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_runs) * 1000.0
        ops_sec = n_runs / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Energy Canvas MRF Generation (32x32)",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_case_memory(self, n_cases: int = 500):
        """Benchmark precedent storage and similarity retrieval."""
        cm = CaseMemory(capacity=1000)
        for i in range(n_cases):
            cm.store_case(PrecedentCase(
                case_id=f"case_{i}",
                situation=f"Patient {i} presents with fever and cough",
                verdict=f"Administer treatment plan {i % 5}",
                rationale="Standard clinical protocol",
            ))

        t0 = time.perf_counter()
        n_queries = 50
        for _ in range(n_queries):
            cm.find_precedents("Patient with severe fever and chest cough", top_k=3)
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_queries) * 1000.0
        ops_sec = n_queries / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Case-Based Precedent Retrieval",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_thinking(self, cortex: CortexGraph, n_queries: int = 3):
        """Benchmark System 2 executive thinking loop."""
        t0 = time.perf_counter()
        for _ in range(n_queries):
            cortex.think("Analyze trade-offs between renewable energy and grid reliability", mode="analytical")
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_queries) * 1000.0
        ops_sec = n_queries / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="System 2 Thinking Engine (Multi-Step)",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_problem_solver(self, cortex: CortexGraph, n_problems: int = 5):
        """Benchmark problem solver parsing and mathematical deduction."""
        problems = [
            "Given distance = 100 km and time = 2 hours, what is speed?",
            "Given mass = 50 kg and acceleration = 9.8 m/s^2, find force.",
            "If price = $80 and discount = 20%, calculate final price.",
        ]
        t0 = time.perf_counter()
        for i in range(n_problems):
            cortex.solve(problems[i % len(problems)])
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_problems) * 1000.0
        ops_sec = n_problems / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Problem Solver & Constraint Engine",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))

    def benchmark_graph_reasoning(self, cortex: CortexGraph, n_queries: int = 5):
        """Benchmark standard cortical query and proof path ranking."""
        t0 = time.perf_counter()
        for _ in range(n_queries):
            cortex.query("What treats hypertension?")
        dt = time.perf_counter() - t0

        avg_ms = (dt / n_queries) * 1000.0
        ops_sec = n_queries / max(dt, 1e-6)
        self.results.append(BenchmarkResult(
            name="Wavefront Activation & Path Ranking",
            operations_per_second=ops_sec,
            avg_latency_ms=avg_ms,
        ))


if __name__ == "__main__":
    from mayon_cortex.learning.bootstrap import bootstrap_all_sectors
    brain = CortexGraph()
    bootstrap_all_sectors(brain)
    suite = CortexBenchmarkSuite()
    suite.run_all(brain)
