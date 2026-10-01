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
Official ARC-AGI 400 Evaluation Dataset Benchmark Harness
==========================================================
Executes Mayon-Cortex Neuro-Symbolic DSL Program Synthesis across
the complete 400 official evaluation challenges from François Chollet's ARC-AGI benchmark.

Usage:
    python benchmarks/benchmark_arc_400_eval.py                   # Run default 50 tasks
    python benchmarks/benchmark_arc_400_eval.py --num-tasks 400    # Run all 400 evaluation tasks
    python benchmarks/benchmark_arc_400_eval.py --task-id 00576224 # Run specific task
"""

import argparse
import glob
import os
import sys
import time
from typing import List, Tuple

# Set UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCProgramSynthesizer,
    IntelligentSynthesizer,
    ARCTask,
    ARCSolution,
)
from mayon_cortex.reasoning.cortical_arc_reasoner import CorticalARCReasoner


def load_evaluation_tasks(data_dir: str, limit: int = 400, task_id_filter: str = None) -> List[ARCTask]:
    """Loads ARCTasks from official JSON files in data_dir."""
    pattern = os.path.join(data_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if task_id_filter:
        files = [f for f in files if task_id_filter in os.path.basename(f)]

    if limit and limit > 0:
        files = files[:limit]

    tasks = []
    for filepath in files:
        try:
            task = ARCTask.from_json_file(filepath, category="ARC-400-Eval")
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    return tasks


def run_400_eval_benchmark(
    num_tasks: int = 50,
    max_depth: int = 2,
    max_beam_width: int = 15,
    task_id: str = None,
    data_dir: str = None,
    engine: str = "cortical",
):
    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, "data", "arc_evaluation")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run: python scripts/download_arc_dataset.py first.")
        return

    print("\n" + "=" * 90)
    print("  MAYON-CORTEX: ARC-AGI 400 EVALUATION SET BENCHMARK HARNESS")
    print(f"  Target: {num_tasks} Tasks | Engine: {engine.upper()} | Pure CPU")
    print("=" * 90 + "\n")

    tasks = load_evaluation_tasks(data_dir, limit=num_tasks, task_id_filter=task_id)
    print(f"Loaded {len(tasks)} evaluation tasks from {data_dir}\n")

    if engine == "cortical":
        solver = CorticalARCReasoner()
    else:
        solver = IntelligentSynthesizer()

    train_solved_count = 0
    test_solved_count = 0
    total_time_ms = 0.0
    solved_tasks = []
    submission_predictions = {}

    print(f"{'Task ID':<14} | {'Train':<6} | {'Test':<6} | {'Latency':<9} | {'Reasoned Cognitive Program / Rule'}")
    print("-" * 105)

    for idx, task in enumerate(tasks, start=1):
        t0 = time.perf_counter()
        if engine == "cortical":
            solution: ARCSolution = solver.solve(task)
        else:
            solution: ARCSolution = solver.synthesize(
                task,
                max_depth=max_depth,
                max_beam_width=max_beam_width,
                use_intelligent_pruning=True,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_time_ms += dt_ms

        train_ok = solution.is_solved

        # Check ground-truth test accuracy
        test_ok = False
        if train_ok and solution.test_predictions and len(task.test_pairs) > 0:
            all_test_matches = True
            for pred_grid, (test_in, expected_out) in zip(solution.test_predictions, task.test_pairs):
                if expected_out is None or not pred_grid.equals(expected_out):
                    all_test_matches = False
                    break
            test_ok = all_test_matches

        if train_ok:
            train_solved_count += 1
        if test_ok:
            test_solved_count += 1
            solved_tasks.append(task.task_id)

        # Record predictions in submission format
        preds_for_task = []
        for p in solution.test_predictions:
            mat = p.matrix.tolist() if hasattr(p, "matrix") else p
            preds_for_task.append({"attempt_1": mat, "attempt_2": mat})
        submission_predictions[task.task_id] = preds_for_task

        train_str = "PASS" if train_ok else "FAIL"
        test_str = "PASS" if test_ok else "FAIL"
        prog_str = " -> ".join(solution.synthesized_program) if solution.synthesized_program else "None"
        if len(prog_str) > 55:
            prog_str = prog_str[:52] + "..."

        print(f"[{idx:03d}] {task.task_id:<8} | {train_str:<6} | {test_str:<6} | {dt_ms:>6.1f} ms | {prog_str}", flush=True)

    print("=" * 105)
    n = max(1, len(tasks))
    train_pct = (train_solved_count / n) * 100.0
    test_pct = (test_solved_count / n) * 100.0
    avg_lat = total_time_ms / n

    print(f"  EVALUATION SUMMARY ({len(tasks)} tasks evaluated):")
    print(f"  - Train Demonstrated Solved : {train_solved_count}/{n} ({train_pct:.1f}%)")
    print(f"  - Test Ground-Truth Solved  : {test_solved_count}/{n} ({test_pct:.1f}%)")
    print(f"  - Avg Latency per Task      : {avg_lat:.2f} ms (Pure CPU Execution)")
    print(f"  - Solved Tasks ({len(solved_tasks)}) : {', '.join(solved_tasks)}")
    print("=" * 105 + "\n")

    return submission_predictions


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Mayon-Cortex ARC-AGI 400 Evaluation Benchmark")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks to evaluate (1-400)")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum DSL search depth")
    parser.add_argument("--max-beam-width", type=int, default=15, help="Beam width for search")
    parser.add_argument("--task-id", type=str, default=None, help="Filter for specific task ID")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to evaluation JSON files")
    parser.add_argument("--engine", type=str, default="cortical", choices=["cortical", "synthesizer"], help="Reasoning engine to evaluate")
    parser.add_argument("--save-predictions", type=str, default=None, help="Optional path to save submission JSON")
    args = parser.parse_args()

    preds = run_400_eval_benchmark(
        num_tasks=args.num_tasks,
        max_depth=args.max_depth,
        max_beam_width=args.max_beam_width,
        task_id=args.task_id,
        data_dir=args.data_dir,
        engine=args.engine,
    )

    if args.save_predictions and preds:
        with open(args.save_predictions, "w") as f:
            json.dump(preds, f)
        print(f"Saved evaluation predictions to {args.save_predictions}")
