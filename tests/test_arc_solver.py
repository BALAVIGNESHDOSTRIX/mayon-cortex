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

import pytest
import numpy as np
from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCObject,
    ARCPerception,
    ARCDSL,
    ARCProgramSynthesizer,
    ARCTask,
    ARCSolution,
)
from mayon_cortex.engine import MayonCortex


class TestARCPerceptionAndDSL:
    def test_grid_creation_and_ascii(self):
        mat = [[0, 1, 2], [3, 4, 5]]
        grid = ARCGrid(mat)
        assert grid.shape == (2, 3)
        assert grid.colors == {0, 1, 2, 3, 4, 5}
        ascii_out = grid.to_ascii("Test Grid")
        assert "2x3" in ascii_out
        assert "┌" in ascii_out

    def test_object_extraction_connected_components(self):
        # Grid with a 2x2 red square (color 2) and a 1x1 blue dot (color 1)
        mat = [
            [0, 2, 2, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        grid = ARCGrid(mat)
        objs = ARCPerception.extract_objects(grid, connectivity=4)
        assert len(objs) == 2

        # Sort by area descending
        objs.sort(key=lambda o: o.area, reverse=True)
        assert objs[0].color == 2
        assert objs[0].area == 4
        assert objs[0].bbox == (0, 1, 1, 2)
        assert objs[0].is_solid_rectangle() is True

        assert objs[1].color == 1
        assert objs[1].area == 1

    def test_enclosed_holes_detection(self):
        # 5x5 grid with a hollow red ring enclosing coordinate (2,2)
        mat = [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        grid = ARCGrid(mat)
        holes = ARCPerception.detect_enclosed_holes(grid, background_color=0)
        assert len(holes) == 1
        assert (2, 2) in holes

    def test_geometric_rotations_and_flips(self):
        mat = [[1, 2], [3, 4]]
        grid = ARCGrid(mat)

        rot90 = ARCDSL.rot90(grid)
        assert rot90.to_list() == [[3, 1], [4, 2]]

        flip_h = ARCDSL.flip_h(grid)
        assert flip_h.to_list() == [[2, 1], [4, 3]]

        flip_v = ARCDSL.flip_v(grid)
        assert flip_v.to_list() == [[3, 4], [1, 2]]

    def test_gravity_down(self):
        mat = [
            [0, 1, 0],
            [2, 0, 3],
            [0, 0, 0],
            [0, 0, 0],
        ]
        grid = ARCGrid(mat)
        grav = ARCDSL.gravity(grid, direction="down")
        expected = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [2, 1, 3],
        ]
        assert grav.to_list() == expected

    def test_hole_filling(self):
        mat = [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        grid = ARCGrid(mat)
        filled = ARCDSL.fill_enclosed_holes(grid, fill_color=4)
        assert filled.matrix[2, 2] == 4

    def test_line_connecting(self):
        # Two blue dots on same row, two green dots on same col
        mat = [
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ]
        grid = ARCGrid(mat)
        connected = ARCDSL.connect_same_colors_with_lines(grid)
        # Check row 0 line of blue (1)
        assert np.all(connected.matrix[0, 1:5] == 1)
        # Check col 0 line of green (3) from row 2 to 4
        assert np.all(connected.matrix[2:5, 0] == 3)


class TestARCProgramSynthesis:
    def test_synthesize_rotation_task(self):
        synth = ARCProgramSynthesizer()
        task = ARCTask(
            task_id="rotate_demo",
            category="geometry",
            train_pairs=[
                (ARCGrid([[1, 2], [3, 0]]), ARCGrid([[3, 1], [0, 2]])),
                (ARCGrid([[5, 0], [0, 7]]), ARCGrid([[0, 5], [7, 0]])),
            ],
            test_pairs=[
                (ARCGrid([[2, 4], [6, 8]]), ARCGrid([[6, 2], [8, 4]])),
            ]
        )
        sol = synth.synthesize(task, max_depth=2)
        assert sol.is_solved is True
        assert "rot90" in sol.synthesized_program
        assert sol.test_predictions[0].equals(task.test_pairs[0][1])

    def test_synthesize_hole_fill_task(self):
        synth = ARCProgramSynthesizer()
        task = ARCTask(
            task_id="hole_fill_demo",
            category="topology",
            train_pairs=[
                (
                    ARCGrid([[2, 2, 2], [2, 0, 2], [2, 2, 2]]),
                    ARCGrid([[2, 2, 2], [2, 4, 2], [2, 2, 2]]),
                ),
            ],
            test_pairs=[
                (
                    ARCGrid([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 0, 3, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]]),
                    ARCGrid([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 4, 3, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]]),
                )
            ]
        )
        sol = synth.synthesize(task, max_depth=2)
        assert sol.is_solved is True
        assert "fill_enclosed_holes_4" in sol.synthesized_program
        assert sol.test_predictions[0].equals(task.test_pairs[0][1])

    def test_engine_integration_solve_arc(self):
        cortex = MayonCortex()
        sol = cortex.solve_arc_grid(
            train_pairs=[
                ([[1, 0], [0, 0]], [[0, 0], [1, 0]]),
                ([[0, 2], [0, 0]], [[0, 0], [0, 2]]),
            ],
            test_input=[[3, 4], [0, 0]],
        )
        assert sol.is_solved is True
        assert sol.synthesized_program[0] in ["flip_v", "gravity_down"]
        assert sol.test_predictions[0].to_list() == [[0, 0], [3, 4]]
