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
Mayon-Cortex Large-Scale Vector Indexing Benchmark
==================================================
Evaluates indexing throughput, query latency (p50, p95, p99), and recall
when node count scales from 1,000 to 100,000+ concept nodes on pure CPU.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mayon_cortex.core.vector_index import (
    VectorIndexFactory,
    _USEARCH_AVAILABLE,
    _FAISS_AVAILABLE,
)


def generate_dataset(num_vectors: int, dim: int = 384, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates normalized concept vectors and test queries."""
    np.random.seed(seed)
    # Generate vectors in chunks to keep memory usage low
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-8)

    queries = np.random.randn(100, dim).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= np.maximum(q_norms, 1e-8)

    return vectors, queries


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Computes exact top-k nearest neighbors via brute-force dot product."""
    sims = np.dot(queries, vectors.T)
    top_k_indices = np.argsort(-sims, axis=1)[:, :k]
    return top_k_indices


def benchmark_backend(
    backend: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    dim: int = 384,
) -> Dict:
    """Benchmarks a single vector index backend."""
    gc.collect()

    # 1. Measure Build / Insertion Time
    index = VectorIndexFactory.create_index(dim=dim, backend=backend)
    t0 = time.perf_counter()
    index.add_batch(vectors)
    build_time = time.perf_counter() - t0
    throughput_nodes_sec = len(vectors) / max(build_time, 1e-6)

    # 2. Measure Single-Query Latency
    latencies = []
    retrieved_indices = []

    # Warmup
    for i in range(min(5, len(queries))):
        index.search(queries[i], k=k)

    for i in range(len(queries)):
        t_start = time.perf_counter()
        scores, idxs = index.search(queries[i], k=k)
        latencies.append((time.perf_counter() - t_start) * 1000.0)  # ms
        retrieved_indices.append(idxs[0])

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    qps = float(1000.0 / np.mean(latencies))

    # 3. Calculate Recall@k against exact ground truth
    recalls = []
    for i in range(len(queries)):
        gt_set = set(ground_truth[i])
        ret_set = set(retrieved_indices[i][:k])
        # Overlap fraction
        overlap = len(gt_set.intersection(ret_set))
        recalls.append(overlap / k)
    mean_recall = float(np.mean(recalls))

    return {
        "backend": index.backend_name,
        "nodes": len(vectors),
        "build_time_s": build_time,
        "build_rate_nodes_s": throughput_nodes_sec,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "qps": qps,
        "recall_at_10": mean_recall,
    }


def run_scaling_benchmark(scales: List[int] = [1000, 10000, 50000, 100000], dim: int = 384):
    print("=" * 80)
    print(" [BENCHMARK] MAYON-CORTEX HIGH-PERFORMANCE VECTOR SEARCH SCALABILITY")
    print(f" Target Dimension: {dim}D (MiniLM / Concept Embedding Space)")
    print(f" Hardware Runtime: Pure CPU (Zero GPU)")
    print(f" Backends Available: USearch={_USEARCH_AVAILABLE}, FAISS={_FAISS_AVAILABLE}, NumPy=True")
    print("=" * 80)

    backends_to_test = []
    if _USEARCH_AVAILABLE:
        backends_to_test.append("usearch")
    if _FAISS_AVAILABLE:
        backends_to_test.append("faiss")
    backends_to_test.append("numpy")

    all_results = []

    for N in scales:
        print(f"\n--- Testing Scale: {N:,} Nodes ---")
        vectors, queries = generate_dataset(num_vectors=N, dim=dim)

        print("  Computing exact ground truth (brute force)...")
        gt = compute_ground_truth(vectors, queries, k=10)

        for backend in backends_to_test:
            # Skip numpy for scales > 50k to save test execution time if requested
            if backend == "numpy" and N > 50000:
                continue

            print(f"  Benchmarking backend: {backend.upper()} on {N:,} nodes...")
            res = benchmark_backend(backend, vectors, queries, gt, k=10, dim=dim)
            all_results.append(res)
            print(
                f"    * {res['backend']:<18} | Build: {res['build_rate_nodes_s']:>10.0f} nodes/s | "
                f"p50: {res['p50_ms']:>6.3f} ms | p99: {res['p99_ms']:>6.3f} ms | "
                f"QPS: {res['qps']:>7.0f} | Recall@10: {res['recall_at_10']*100:>5.1f}%"
            )

    print("\n" + "=" * 92)
    print(" [SUMMARY] FINAL SCALABILITY BENCHMARK TABLE")
    print("=" * 92)
    print(f"{'Scale (Nodes)':<14} | {'Backend':<18} | {'Build Rate':<14} | {'p50 (ms)':<9} | {'p99 (ms)':<9} | {'QPS':<8} | {'Recall@10':<9}")
    print("-" * 92)
    for r in all_results:
        print(
            f"{r['nodes']:<14,d} | {r['backend']:<18} | {r['build_rate_nodes_s']:>10.0f} n/s | "
            f"{r['p50_ms']:>7.3f} ms | {r['p99_ms']:>7.3f} ms | {r['qps']:>8.0f} | {r['recall_at_10']*100:>7.1f}%"
        )
    print("=" * 92)


if __name__ == "__main__":
    test_scales = [1000, 10000, 50000, 100000]
    if len(sys.argv) > 1:
        test_scales = [int(x) for x in sys.argv[1:]]
    run_scaling_benchmark(test_scales)
