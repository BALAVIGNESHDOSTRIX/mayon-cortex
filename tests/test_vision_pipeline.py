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
Comprehensive Integration Tests for Mayon-Cortex Vision Pipeline
=================================================================
Validates 256-dim feature extraction, 3-level hierarchical codebook,
universal saliency, knowledge boundary refusal, MRF refinement,
and 7-stage visual imagination engine.
"""

import numpy as np
import pytest

from mayon_cortex.perception.visual_features import VisualFeatureExtractor
from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph
from mayon_cortex.perception.visual_generator import GraphImageGenerator, GenerationConfig
from mayon_cortex.perception.knowledge_boundary import KnowledgeBoundary, ConceptKnowledge
from mayon_cortex.perception.visual_imagination import (
    VisualImagination,
    ImaginationConfig,
    ImaginationResult,
)
from mayon_cortex.engine import MayonCortex


def test_visual_feature_extractor_256dim():
    """Verify 256-dim visual feature extraction and L2 unit normalization."""
    extractor = VisualFeatureExtractor()
    patch = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    feat = extractor.extract_patch_features(patch)
    assert feat.shape == (256,)
    assert np.isclose(np.linalg.norm(feat), 1.0, atol=1e-3)

    # Verify each sub-component
    assert extractor.hog_features(patch).shape == (36,)
    assert extractor.color_histogram(patch).shape == (48,)
    assert extractor.lbp_features(patch).shape == (28,)
    assert extractor.gabor_features(patch).shape == (16,)
    assert extractor.spatial_edge_density(patch).shape == (16,)
    assert extractor.dominant_color_palette(patch).shape == (24,)
    assert extractor.dct_frequency_energy(patch).shape == (16,)
    assert extractor.gradient_magnitude_distribution(patch).shape == (16,)
    assert extractor.edge_orientation_coherence(patch).shape == (16,)
    assert extractor.fractal_dimension_estimate(patch).shape == (16,)
    assert extractor.multi_scale_saliency(patch).shape == (24,)


def test_codebook_learn_and_encode():
    """Verify codebook learning, patch quantization, and decode round-trip."""
    codebook = VisualCodebook(codebook_size=16, feature_dim=256, patch_size=(32, 32))

    img1 = np.full((64, 64, 3), 220, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 40, dtype=np.uint8)

    samples = [
        (img1, "bright", {"color": "white"}),
        (img2, "dark", {"color": "black"}),
    ]

    num_learned = codebook.learn(samples, k=8)
    assert num_learned > 0
    assert len(codebook.entries) == num_learned

    indices, grid_shape = codebook.encode_image(img1)
    assert len(indices) == 4
    assert grid_shape == (2, 2)

    reconstructed = codebook.decode_indices(indices, grid_shape)
    assert reconstructed.shape == (64, 64, 3)
    assert reconstructed.dtype == np.uint8


def test_hierarchical_codebook():
    """Verify 3-level hierarchical codebook multi-scale encoding."""
    codebook = VisualCodebook(codebook_size=32, feature_dim=256, patch_size=(32, 32))
    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

    samples = [(img, "pattern", {})]
    codebook.learn(samples, k=16)

    # Hierarchical encoding across coarse, medium, fine
    h_encoded = codebook.encode_hierarchical(img)
    assert "coarse" in h_encoded
    assert "medium" in h_encoded
    assert "fine" in h_encoded

    for lvl in ["coarse", "medium", "fine"]:
        indices, shape = h_encoded[lvl]
        assert len(indices) == 4
        assert shape == (2, 2)


def test_universal_saliency():
    """Verify Itti-Koch inspired universal visual saliency extraction."""
    img = np.full((64, 64, 3), 50, dtype=np.uint8)
    # Bright center square
    img[20:44, 20:44] = 230

    mask = VisualCodebook.extract_universal_saliency(img)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    # Center should have higher saliency than corners
    assert mask[32, 32] > mask[5, 5]


def test_spatial_graph_learn():
    """Verify spatial graph learns 8-directional transitions and positional priors."""
    spatial_graph = VisualSpatialGraph(num_entries=10)
    grid = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    spatial_graph.learn_from_grid(grid, label="test_object")

    prob_right = spatial_graph.get_transition_probability(1, 2, "right")
    assert prob_right > 0.0

    prob_below = spatial_graph.get_transition_probability(1, 4, "below")
    assert prob_below > 0.0

    # Positional prior
    fitness = spatial_graph.get_position_fitness(1, norm_r=0.0, norm_c=0.0, label="test_object")
    assert fitness > 0.5


def test_spatial_graph_reservoir():
    """Verify reservoir sampling caps positional samples to prevent memory explosion."""
    spatial_graph = VisualSpatialGraph(num_entries=10, max_samples_per_token=20)
    # Feed 100 grids containing token 1
    for _ in range(100):
        spatial_graph.learn_from_grid([[1, 2], [2, 1]])

    assert len(spatial_graph.position_samples[1]) <= 20
    assert len(spatial_graph.position_samples[2]) <= 20


def test_knowledge_boundary_known_concept():
    """Verify KnowledgeBoundary approves known concepts with high confidence."""
    kb = KnowledgeBoundary()
    kb.register_concept(
        label="cat",
        images_count=3,
        patches_count=48,
        avg_vector=np.ones(256) / 16.0,
    )

    report = kb.assess_prompt("a fluffy orange cat")
    assert report.can_generate is True
    assert "cat" in report.known_concepts
    assert report.confidence > 0.5


def test_knowledge_boundary_unknown_concept():
    """Verify KnowledgeBoundary honestly refuses unlearned concepts with explanation."""
    kb = KnowledgeBoundary()
    kb.register_concept(
        label="cat",
        images_count=2,
        patches_count=32,
        avg_vector=np.ones(256) / 16.0,
    )

    report = kb.assess_prompt("a supersonic spaceship flying through a wormhole")
    assert report.can_generate is False
    assert report.confidence == 0.0
    assert "spaceship" in report.reason or "supersonic" in report.reason
    assert "cat" in report.reason  # Suggests known concept


def test_knowledge_boundary_partial_match():
    """Verify KnowledgeBoundary handles partial prompt knowledge gracefully."""
    kb = KnowledgeBoundary()
    kb.register_concept("cat", 2, 32, np.ones(256) / 16.0)

    report = kb.assess_prompt("a cat wearing an astronaut suit")
    assert report.can_generate is True
    assert "cat" in report.known_concepts
    assert len(report.unknown_concepts) > 0
    assert "partially know" in report.reason


def test_mrf_refinement():
    """Verify MRF energy minimization optimizes local grid harmony."""
    codebook = VisualCodebook(codebook_size=8, feature_dim=256)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    codebook.learn([(img, "nature", {})], k=4)

    spatial_graph = VisualSpatialGraph(num_entries=8)
    spatial_graph.learn_from_grid([[0, 1], [2, 3]])

    gen = GraphImageGenerator(codebook=codebook, spatial_graph=spatial_graph)

    # Initial grid
    grid = [[0, 3], [3, 0]]
    refined = gen._mrf_refine(grid, available_tokens=[0, 1, 2, 3], passes=2)
    assert len(refined) == 2
    assert len(refined[0]) == 2


def test_imagination_full_pipeline():
    """End-to-end test of 7-stage visual imagination pipeline."""
    codebook = VisualCodebook(codebook_size=16, feature_dim=256)
    dummy_cat = np.full((64, 64, 3), 160, dtype=np.uint8)
    dummy_cat[20:44, 20:44] = [210, 140, 50]  # Orange patch
    codebook.learn([(dummy_cat, "cat", {"color": "orange"})], k=8)

    kb = KnowledgeBoundary()
    kb.register_concept("cat", 1, 16, codebook.concept_embeddings["cat"])

    imagination = VisualImagination(codebook=codebook, knowledge_boundary=kb)
    config = ImaginationConfig(
        output_width=128,
        output_height=128,
        coarse_grid=(4, 4),
        medium_grid=(4, 4),
        fine_grid=(4, 4),
        mrf_passes=1,
    )

    result = imagination.imagine("a fluffy orange cat on a wooden table", config=config)
    assert result.can_generate is True
    assert result.image.shape == (128, 128, 3)
    assert result.image.dtype == np.uint8
    assert "stage3_coarse" in result.generation_stages
    assert "stage7_final" in result.generation_stages


def test_imagination_refuses_unknown():
    """Verify visual imagination returns empty image and honest refusal for unknown concept."""
    codebook = VisualCodebook(codebook_size=8, feature_dim=256)
    kb = KnowledgeBoundary()
    imagination = VisualImagination(codebook=codebook, knowledge_boundary=kb)

    result = imagination.imagine("alien starship orbiting Saturn")
    assert result.can_generate is False
    assert result.image.shape == (512, 512, 3)
    assert np.all(result.image == 0)
    assert "haven't learned" in result.capability_report.reason


def test_cortex_engine_imagine_image():
    """Test CortexGraph.imagine_image() and learn_image() integrated API."""
    cortex = MayonCortex()
    cat_img = np.full((64, 64, 3), 150, dtype=np.uint8)
    cat_img[20:40, 20:40] = [220, 130, 40]

    # Learn image
    learn_res = cortex.learn_image(cat_img, label="cat", patch_size=(32, 32))
    assert learn_res["label"] == "cat"
    assert "cat" in cortex.knowledge_boundary.concepts

    # Generate known
    res_known = cortex.imagine_image("a cute orange cat", width=64, height=64)
    assert res_known.can_generate is True
    assert res_known.image.shape == (64, 64, 3)

    # Refuse unknown
    res_unknown = cortex.imagine_image("quantum computer mainframe", width=64, height=64)
    assert res_unknown.can_generate is False
    assert "haven't learned" in res_unknown.capability_report.reason
