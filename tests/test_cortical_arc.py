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
Unit & Integration Tests for Mayon-Cortex Graph-Native ARC Reasoning
====================================================================
Tests:
1. ARCSpatialSceneGraph: Object segmentation, topological holes, spatial relations.
2. CorticalARCReasoner: Analogical rule induction and demonstration verification.
3. MayonCortex.solve_arc: End-to-end cognitive problem solving on master brain.
"""

import os
import json
import pytest
import numpy as np

from mayon_cortex.perception import ARCSpatialSceneGraph
from mayon_cortex.reasoning import CorticalARCReasoner, ARCTask, ARCSolution
from mayon_cortex.engine import MayonCortex


class TestARCSpatialSceneGraph:
    """Verifies that 2D grids are converted to cognitive spatial relation graphs."""

    def test_object_and_cavity_graph_extraction(self):
        # 5x5 grid with a green ring (color 3) enclosing a blue center dot (color 1)
        # and a separate red square (color 2)
        grid = np.array([
            [3, 3, 3, 0, 2],
            [3, 0, 3, 0, 2],
            [3, 3, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=int)

        sg = ARCSpatialSceneGraph(scene_id="ring_scene", grid=grid)
        assert sg.grid_shape == (5, 5)
        assert sg.background_color == 0
        assert len(sg.objects) == 2  # green ring and red square

        # Find the green ring object
        ring_objs = [o for o in sg.objects.values() if o.attributes["color"] == 3]
        assert len(ring_objs) == 1
        ring = ring_objs[0]
        assert ring.attributes["num_holes"] == 1  # 1 enclosed cavity!
        assert ring.attributes["area"] == 8
        assert not ring.attributes["is_solid"]

        # Red block has 0 holes and is solid
        red_objs = [o for o in sg.objects.values() if o.attributes["color"] == 2]
        assert len(red_objs) == 1
        red = red_objs[0]
        assert red.attributes["num_holes"] == 0
        assert red.attributes["is_solid"]

        # Spatial relations: ring is left of red block
        rel_types = [r.relation for r in sg.relations]
        assert "left_of" in rel_types or "right_of" in rel_types

    def test_delimiter_and_symmetry_detection(self):
        # 3x5 grid with vertical delimiter of 5s down the center
        grid = np.array([
            [1, 0, 5, 2, 0],
            [0, 1, 5, 0, 2],
            [1, 1, 5, 2, 2],
        ], dtype=int)

        sg = ARCSpatialSceneGraph(scene_id="delim_scene", grid=grid)
        assert 2 in sg.delimiters["vertical"]


class TestCorticalARCReasoner:
    """Verifies that CorticalARCReasoner discovers and proves rules via graph analogy."""

    def test_standalone_reasoner_affine_scale(self):
        # 60c09cac: Affine 2D 2x scaling
        task_path = "data/arc_evaluation/60c09cac.json"
        if not os.path.exists(task_path):
            pytest.skip("Evaluation dataset not found")

        reasoner = CorticalARCReasoner()
        solution = reasoner.solve(task_path)
        assert solution.is_solved is True
        assert solution.confidence == 1.0

        # Verify ground truth
        with open(task_path, "r") as f:
            data = json.load(f)
        gt = data["test"][0]["output"]
        assert np.array_equal(solution.predicted_test_grids[0], gt)

    def test_standalone_reasoner_delimiter_algebra(self):
        # 195ba7dc: Vertical delimiter OR logic
        task_path = "data/arc_evaluation/195ba7dc.json"
        if not os.path.exists(task_path):
            pytest.skip("Evaluation dataset not found")

        reasoner = CorticalARCReasoner()
        solution = reasoner.solve(task_path)
        assert solution.is_solved is True
        assert solution.confidence == 1.0

        with open(task_path, "r") as f:
            data = json.load(f)
        gt = data["test"][0]["output"]
        assert np.array_equal(solution.predicted_test_grids[0], gt)

    def test_standalone_reasoner_quadrant_reflection(self):
        # 0c786b71: 4-quadrant reflection
        task_path = "data/arc_evaluation/0c786b71.json"
        if not os.path.exists(task_path):
            pytest.skip("Evaluation dataset not found")

        reasoner = CorticalARCReasoner()
        solution = reasoner.solve(task_path)
        assert solution.is_solved is True
        assert solution.confidence == 1.0

        with open(task_path, "r") as f:
            data = json.load(f)
        gt = data["test"][0]["output"]
        assert np.array_equal(solution.predicted_test_grids[0], gt)


class TestMayonCortexMasterEngineIntegration:
    """Verifies that MayonCortex itself solves ARC tasks natively through solve_arc."""

    def test_cortex_master_engine_solve_arc(self):
        task_path = "data/arc_evaluation/60c09cac.json"
        if not os.path.exists(task_path):
            pytest.skip("Evaluation dataset not found")

        # Create master cortical brain
        cortex = MayonCortex()
        assert hasattr(cortex, "solve_arc")
        assert hasattr(cortex, "arc_reasoner")

        # Solve task through cortex master API
        solution = cortex.solve_arc(task_path)
        assert isinstance(solution, ARCSolution)
        assert solution.is_solved is True

        with open(task_path, "r") as f:
            data = json.load(f)
        gt = data["test"][0]["output"]
        assert np.array_equal(solution.predicted_test_grids[0], gt)
