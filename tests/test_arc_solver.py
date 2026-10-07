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

import pytest
import numpy as np
from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCObject,
    ARCPerception,
    ARCDSL,
    ARCProgramSynthesizer,
    IntelligentSynthesizer,
    ObjectCentricSynthesizer,
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

    def test_perception_background_and_symmetry(self):
        mat = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 3, 4, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        grid = ARCGrid(mat)
        bg = ARCPerception.detect_background_color(grid)
        assert bg == 0

        syms = ARCPerception.detect_symmetry(grid)
        assert syms["vertical"] is True
        assert syms["horizontal"] is False

        hist = ARCPerception.color_histogram(grid)
        assert hist[0] == 14
        assert hist[1] == 2
        assert hist[4] == 1

        obj_count = ARCPerception.count_objects(grid)
        assert obj_count >= 1

    def test_perception_io_analysis(self):
        g1 = ARCGrid([[1, 2], [3, 4]])
        g2 = ARCGrid([[2, 1], [4, 3]])
        analysis = ARCPerception.analyze_io_relationship([(g1, g2)])
        assert analysis["same_shape"] is True
        assert analysis["colors_preserved"] is True
        assert analysis["output_is_smaller"] is False
        assert analysis["output_is_larger"] is False

    def test_dsl_morphological_operations(self):
        # 3x3 with single center pixel
        mat = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        grid = ARCGrid(mat)

        dilated = ARCDSL.dilate(grid)
        assert dilated.matrix[1, 1] == 2
        assert dilated.matrix[0, 1] == 2
        assert dilated.matrix[2, 1] == 2
        assert dilated.matrix[1, 0] == 2
        assert dilated.matrix[1, 2] == 2

        # Solid 3x3 eroding
        solid = ARCGrid(np.full((3, 3), 2))
        eroded = ARCDSL.erode(solid)
        # All border pixels touch edge, so only interior (or none) survives
        assert eroded.matrix[0, 0] == 0

    def test_dsl_symmetry_operations(self):
        mat = [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
        ]
        grid = ARCGrid(mat)
        mirrored = ARCDSL.mirror_left_to_right(grid)
        assert mirrored.matrix[0, 3] == 1
        assert mirrored.matrix[0, 2] == 2
        assert mirrored.matrix[1, 3] == 3
        assert mirrored.matrix[1, 2] == 4

        # Anti-diagonal flip
        diag = ARCDSL.diagonal_flip_anti(grid)
        assert diag.shape == (4, 2)

    def test_dsl_object_operations(self):
        # Two objects: a 2x2 square (area 4) and a 1x1 dot (area 1)
        mat = [
            [2, 2, 0, 0],
            [2, 2, 0, 0],
            [0, 0, 0, 1],
        ]
        grid = ARCGrid(mat)

        # Remove smallest should remove the dot
        no_smallest = ARCDSL.remove_smallest_object(grid)
        assert no_smallest.matrix[2, 3] == 0
        assert no_smallest.matrix[0, 0] == 2

        # Keep largest retains square
        only_largest = ARCDSL.keep_largest_object(grid)
        assert only_largest.matrix[0, 0] == 2
        assert only_largest.matrix[2, 3] == 0

    def test_dsl_grid_composition(self):
        g1 = ARCGrid([[1, 0], [0, 2]])
        g2 = ARCGrid([[0, 3], [0, 0]])

        overlay = ARCDSL.overlay_nonzero(g1, g2)
        assert overlay.matrix[0, 1] == 3
        assert overlay.matrix[0, 0] == 1

        xor_g = ARCDSL.xor_grids(g1, g2)
        assert xor_g.matrix[0, 0] == 1
        assert xor_g.matrix[0, 1] == 3

    def test_dsl_border_and_patterns(self):
        base = ARCGrid([[1, 2], [3, 4]])
        bordered = ARCDSL.add_border(base, color=5, width=1)
        assert bordered.shape == (4, 4)
        assert bordered.matrix[0, 0] == 5
        assert bordered.matrix[1, 1] == 1

        unbordered = ARCDSL.remove_border(bordered, width=1)
        assert unbordered.shape == (2, 2)
        assert unbordered.equals(base)

        tiled = ARCDSL.repeat_horizontal(base, n=2)
        assert tiled.shape == (2, 4)


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

    def test_intelligent_candidate_selection_pruning(self):
        synth = IntelligentSynthesizer()
        task = ARCTask(
            task_id="same_size_demo",
            category="symmetry",
            train_pairs=[
                (ARCGrid([[1, 2], [3, 4]]), ARCGrid([[2, 1], [4, 3]])),
            ],
            test_pairs=[
                (ARCGrid([[5, 6], [7, 8]]), ARCGrid([[6, 5], [8, 7]])),
            ]
        )
        candidates = synth.select_candidate_ops(task)
        candidate_names = [name for name, _ in candidates]
        # Should prune expansion ops like tile, scale, add_border
        assert "scale_2x" not in candidate_names
        assert "tile_2x2" not in candidate_names
        assert "add_border_1" not in candidate_names

    def test_synthesize_mirror_task(self):
        synth = IntelligentSynthesizer()
        task = ARCTask(
            task_id="mirror_demo",
            category="symmetry",
            train_pairs=[
                (ARCGrid([[1, 0], [2, 0]]), ARCGrid([[1, 1], [2, 2]])),
            ],
            test_pairs=[
                (ARCGrid([[3, 0], [4, 0]]), ARCGrid([[3, 3], [4, 4]])),
            ]
        )
        sol = synth.synthesize(task, max_depth=2)
        assert sol.is_solved is True
        assert "mirror_left_to_right" in sol.synthesized_program
        assert sol.test_predictions[0].equals(task.test_pairs[0][1])

    def test_arc_task_from_dict_official_format(self):
        official_json_payload = {
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[1, 0], [0, 0]]}
            ],
            "test": [
                {"input": [[0, 2], [0, 0]], "output": [[2, 0], [0, 0]]}
            ]
        }
        task = ARCTask.from_dict(official_json_payload, task_id="official_demo")
        assert task.task_id == "official_demo"
        assert len(task.train_pairs) == 1
        assert len(task.test_pairs) == 1
        assert task.train_pairs[0][0].shape == (2, 2)
        assert task.test_pairs[0][1] is not None

    def test_object_centric_multi_program_synthesis(self):
        # Blue dot (1) should be deleted, Red square/dot (2) should be recolored to yellow (4)
        synth = ARCProgramSynthesizer()
        task = ARCTask(
            task_id="obj_centric_demo",
            category="object_manipulation",
            train_pairs=[
                (
                    ARCGrid([[1, 0, 0], [0, 2, 2], [0, 2, 2]]),
                    ARCGrid([[0, 0, 0], [0, 4, 4], [0, 4, 4]]),
                ),
                (
                    ARCGrid([[0, 1, 0], [0, 0, 0], [2, 0, 0]]),
                    ARCGrid([[0, 0, 0], [0, 0, 0], [4, 0, 0]]),
                ),
            ],
            test_pairs=[
                (
                    ARCGrid([[0, 0, 1], [0, 2, 0], [0, 0, 0]]),
                    ARCGrid([[0, 0, 0], [0, 4, 0], [0, 0, 0]]),
                ),
            ]
        )
        sol = synth.synthesize(task, max_depth=1)
        assert sol.is_solved is True
        assert sol.test_predictions[0].equals(task.test_pairs[0][1])

    def test_dsl_connect_lines_priority(self):
        # Two horizontal points of color 2, two vertical points of color 3 crossing at (1, 1)
        mat = np.zeros((3, 3), dtype=int)
        mat[1, 0] = 2
        mat[1, 2] = 2
        mat[0, 1] = 3
        mat[2, 1] = 3
        grid = ARCGrid(mat)

        h_then_v = ARCDSL.connect_lines_h_then_v(grid)
        # Vertical is drawn second, so intersection (1, 1) must be color 3
        assert h_then_v.matrix[1, 1] == 3
        assert h_then_v.matrix[1, 0] == 2
        assert h_then_v.matrix[1, 2] == 2

        v_then_h = ARCDSL.connect_lines_v_then_h(grid)
        # Horizontal is drawn second, so intersection (1, 1) must be color 2
        assert v_then_h.matrix[1, 1] == 2

    def test_dsl_fractal_complement_expand(self):
        # 2x2 motif: color 1 at diagonal, background 0 elsewhere
        mat = np.array([[1, 0], [0, 1]], dtype=int)
        grid = ARCGrid(mat)
        expanded = ARCDSL.fractal_complement_expand(grid)
        assert expanded.shape == (4, 4)
        # In complement motif, (0, 0) was 1 -> motif has 0 at (0, 0) and 1 at (0, 1), (1, 0)
        assert expanded.matrix[0, 0] == 0
        assert expanded.matrix[0, 1] == 1
        assert expanded.matrix[1, 0] == 1
        assert expanded.matrix[1, 1] == 0

    def test_dsl_extend_diagonal_arithmetic(self):
        # Diagonal arithmetic sequence: (1, 1) and (2, 2) of color 8
        mat = np.zeros((5, 5), dtype=int)
        mat[1, 1] = 8
        mat[2, 2] = 8
        grid = ARCGrid(mat)
        extended = ARCDSL.extend_diagonal_arithmetic(grid, ext_color=2)
        # Ray should extend to (0, 0), (3, 3), (4, 4) with color 2
        assert extended.matrix[0, 0] == 2
        assert extended.matrix[3, 3] == 2
        assert extended.matrix[4, 4] == 2
        # Seeds remain 8
        assert extended.matrix[1, 1] == 8
        assert extended.matrix[2, 2] == 8

    def test_perception_detect_enclosed_chambers(self):
        # A 5x5 box with hollow 3x3 chamber inside
        mat = np.zeros((7, 7), dtype=int)
        mat[1:6, 1:6] = 2
        mat[2:5, 2:5] = 0
        grid = ARCGrid(mat)
        chambers = ARCPerception.detect_enclosed_chambers(grid)
        assert len(chambers) == 1
        assert len(chambers[0]) == 9

    def test_official_arc_evaluation_solvers(self):
        import os
        import json
        eval_tasks = [
            "60c09cac.json",
            "85b81ff1.json",
            "070dd51e.json",
            "0692e18c.json",
            "0b17323b.json",
            "00dbd492.json",
            "15696249.json",
            "12422b43.json",
            "140c817e.json",
            "17b80ad2.json",
            "00576224.json",
            "0c786b71.json",
            "0a2355a6.json",
            "833dafe3.json",
            "37d3e8b2.json",
            "ae58858e.json",
            "358ba94e.json",
            "0a1d4ef5.json",
            "195ba7dc.json",
            "34b99a2b.json",
            "506d28a5.json",
            "5d2a5c43.json",
            "e133d23d.json",
            "21f83797.json",
            "45bbe264.json",
            "1a2e2828.json",
        ]
        synth = ARCProgramSynthesizer()
        for fname in eval_tasks:
            fpath = os.path.join("data", "arc_evaluation", fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                data = json.load(f)
            task = ARCTask.from_dict(data, task_id=fname.replace(".json", ""))
            sol = synth.synthesize(task)
            assert sol.is_solved is True, f"Failed on {fname}"
            test_in, test_expected = task.test_pairs[0]
            if test_expected is not None:
                assert sol.test_predictions[0].equals(test_expected), f"Mismatch on {fname}"



