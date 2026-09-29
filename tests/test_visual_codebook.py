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

"""Unit tests for Visual Feature Extractor, Visual Codebook, and Spatial Graph."""

import numpy as np
import pytest

from mayon_cortex.perception.visual_features import VisualFeatureExtractor
from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph
from mayon_cortex.perception.visual_generator import VisualTokenPredictor, GraphImageGenerator
from mayon_cortex.perception.procedural_textures import ProceduralTextureEngine
from mayon_cortex.perception.raymarcher import RaymarchRenderer, SDFObject, Material


def test_visual_feature_extractor():
    extractor = VisualFeatureExtractor()
    patch = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    feat = extractor.extract_patch_features(patch)
    assert feat.shape == (256,)
    assert np.isclose(np.linalg.norm(feat), 1.0, atol=1e-3)

    # Test individual components
    hog = extractor.hog_features(patch)
    assert hog.shape == (36,)

    color = extractor.color_histogram(patch)
    assert color.shape == (48,)

    lbp = extractor.lbp_features(patch)
    assert lbp.shape == (28,)

    gabor = extractor.gabor_features(patch)
    assert gabor.shape == (16,)

    edge_dense = extractor.spatial_edge_density(patch)
    assert edge_dense.shape == (16,)

    palette = extractor.dominant_color_palette(patch)
    assert palette.shape == (24,)

    dct_feat = extractor.dct_frequency_energy(patch)
    assert dct_feat.shape == (16,)

    grad_dist = extractor.gradient_magnitude_distribution(patch)
    assert grad_dist.shape == (16,)

    coherence = extractor.edge_orientation_coherence(patch)
    assert coherence.shape == (16,)

    fractal = extractor.fractal_dimension_estimate(patch)
    assert fractal.shape == (16,)


def test_image_decomposition():
    extractor = VisualFeatureExtractor()
    img = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)

    patches = extractor.decompose_image(img, patch_size=(32, 32))
    assert len(patches) == 9  # 3x3 grid
    assert patches[0].patch.shape == (32, 32, 3)
    assert patches[0].row_idx == 0 and patches[0].col_idx == 0
    assert patches[8].row_idx == 2 and patches[8].col_idx == 2


def test_visual_codebook_learning_and_encoding():
    codebook = VisualCodebook(codebook_size=16, feature_dim=128, patch_size=(32, 32))

    # Create dummy images with labels
    img1 = np.full((64, 64, 3), 200, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 50, dtype=np.uint8)

    samples = [
        (img1, "bright", {"color": "white"}),
        (img2, "dark", {"color": "black"}),
    ]

    num_learned = codebook.learn(samples, k=8)
    assert num_learned > 0
    assert len(codebook.entries) == num_learned

    # Test encoding
    indices, grid_shape = codebook.encode_image(img1)
    assert len(indices) == 4
    assert grid_shape == (2, 2)

    # Test decoding
    reconstructed = codebook.decode_indices(indices, grid_shape)
    assert reconstructed.shape == (64, 64, 3)
    assert reconstructed.dtype == np.uint8


def test_visual_spatial_graph():
    spatial_graph = VisualSpatialGraph(num_entries=10)
    grid = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    spatial_graph.learn_from_grid(grid)

    prob_right = spatial_graph.get_transition_probability(1, 2, "right")
    assert prob_right > 0.0

    prob_below = spatial_graph.get_transition_probability(1, 4, "below")
    assert prob_below > 0.0


def test_procedural_textures():
    engine = ProceduralTextureEngine()

    perlin = engine.perlin_noise_2d(64, 64)
    assert perlin.shape == (64, 64)
    assert 0.0 <= perlin.min() and perlin.max() <= 1.0

    fbm = engine.fbm_noise_2d(64, 64)
    assert fbm.shape == (64, 64)

    marble = engine.marble_texture(64, 64)
    assert marble.shape == (64, 64, 3)
    assert marble.dtype == np.uint8

    wood = engine.wood_texture(64, 64)
    assert wood.shape == (64, 64, 3)

    clouds = engine.clouds_sky(64, 64)
    assert clouds.shape == (64, 64, 3)


def test_raymarcher_renderer():
    renderer = RaymarchRenderer(max_steps=16, max_dist=10.0)

    objects = [
        SDFObject(
            name="sphere",
            sdf_fn=lambda p: RaymarchRenderer.sdf_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0),
            material=Material(albedo=(0.9, 0.2, 0.2)),
        )
    ]

    img = renderer.render_scene(objects, width=32, height=32)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8
