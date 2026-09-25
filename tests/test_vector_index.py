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
Unit Tests for Mayon-Cortex High-Performance Vector Index Engine
================================================================
Tests USearch, FAISS, and NumPy backends for:
  - Vector creation & batch insertion
  - Nearest neighbor retrieval & similarity scoring
  - Cross-backend result consistency
  - Graph integration & traverse functionality
  - Index serialization / deserialization
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from mayon_cortex.core.vector_index import (
    BaseVectorIndex,
    VectorIndexFactory,
    USearchVectorIndex,
    FaissVectorIndex,
    NumpyVectorIndex,
    _USEARCH_AVAILABLE,
    _FAISS_AVAILABLE,
)
from mayon_cortex.core.graph import MayonGraph, GraphLevel


@pytest.fixture
def sample_vectors():
    np.random.seed(42)
    dim = 64
    N = 100
    vecs = np.random.randn(N, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def available_backends():
    backends = ["numpy"]
    if _USEARCH_AVAILABLE:
        backends.append("usearch")
    if _FAISS_AVAILABLE:
        backends.append("faiss")
    return backends


class TestVectorIndexBackends:
    """Test standalone index backends."""

    def test_factory_auto_selection(self):
        index = VectorIndexFactory.create_index(dim=64, backend="auto")
        assert isinstance(index, BaseVectorIndex)
        assert index.dim == 64
        if _USEARCH_AVAILABLE:
            assert "usearch" in index.backend_name
        elif _FAISS_AVAILABLE:
            assert "faiss" in index.backend_name
        else:
            assert "numpy" in index.backend_name

    def test_numpy_backend_correctness(self, sample_vectors):
        dim = sample_vectors.shape[1]
        index = NumpyVectorIndex(dim=dim, metric="cos")
        index.add_batch(sample_vectors)
        assert index.size == len(sample_vectors)

        # Query with exact first vector -> score should be 1.0, index 0
        q = sample_vectors[0]
        scores, indices = index.search(q, k=5)
        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert indices[0, 0] == 0
        assert np.isclose(scores[0, 0], 1.0, atol=1e-4)

    @pytest.mark.skipif(not _USEARCH_AVAILABLE, reason="USearch not installed")
    def test_usearch_backend(self, sample_vectors):
        dim = sample_vectors.shape[1]
        index = USearchVectorIndex(dim=dim, metric="cos")
        index.add_batch(sample_vectors)
        assert index.size == len(sample_vectors)

        # Exact match test
        q = sample_vectors[5]
        scores, indices = index.search(q, k=5)
        assert indices[0, 0] == 5
        assert np.isclose(scores[0, 0], 1.0, atol=1e-3)

    @pytest.mark.skipif(not _FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_backend(self, sample_vectors):
        dim = sample_vectors.shape[1]
        index = FaissVectorIndex(dim=dim, metric="cos")
        index.add_batch(sample_vectors)
        assert index.size == len(sample_vectors)

        q = sample_vectors[10]
        scores, indices = index.search(q, k=5)
        assert indices[0, 0] == 10
        assert np.isclose(scores[0, 0], 1.0, atol=1e-4)

    def test_cross_backend_consistency(self, sample_vectors, available_backends):
        """Verify that USearch, FAISS, and NumPy return consistent top-1 nearest neighbors."""
        dim = sample_vectors.shape[1]
        indices_instances = {}

        for backend in available_backends:
            idx = VectorIndexFactory.create_index(dim=dim, backend=backend)
            idx.add_batch(sample_vectors)
            indices_instances[backend] = idx

        # Generate 10 random query vectors
        np.random.seed(123)
        queries = np.random.randn(10, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        for q in queries:
            ref_scores, ref_indices = indices_instances["numpy"].search(q, k=3)
            ref_top1 = ref_indices[0, 0]

            for b in available_backends:
                if b == "numpy":
                    continue
                b_scores, b_indices = indices_instances[b].search(q, k=3)
                b_top1 = b_indices[0, 0]
                # High recall check: top-1 index matches
                assert b_top1 == ref_top1, f"Backend {b} returned {b_top1} but expected {ref_top1}"

    def test_save_and_load(self, sample_vectors, available_backends):
        dim = sample_vectors.shape[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            for backend in available_backends:
                idx = VectorIndexFactory.create_index(dim=dim, backend=backend)
                idx.add_batch(sample_vectors)

                save_path = Path(tmpdir) / f"index_{backend}"
                idx.save(save_path)

                loaded_idx = VectorIndexFactory.create_index(dim=dim, backend=backend)
                loaded_idx.load(save_path)
                assert loaded_idx.size == len(sample_vectors)

                # Search on loaded index
                q = sample_vectors[2]
                scores, indices = loaded_idx.search(q, k=3)
                assert indices[0, 0] == 2


class TestMayonGraphVectorIntegration:
    """Test MayonGraph integration with polymorphic vector backends."""

    def test_graph_with_auto_backend(self):
        graph = MayonGraph(concept_dim=32, ann_index_type="auto")
        assert graph.vector_index is not None

        # Add single node
        v1 = np.random.randn(32).astype(np.float32)
        n1 = graph.add_node(vector=v1, source_text="Concept Alpha", sector="medical")

        # Add bulk nodes
        v_bulk = np.random.randn(5, 32).astype(np.float32)
        texts = [f"Bulk Node {i}" for i in range(5)]
        bulk_nodes = graph.add_nodes_bulk(vectors=v_bulk, source_texts=texts, sectors=["tech"] * 5)

        assert graph.num_nodes == 6
        assert graph.vector_index.size == 6

        # Direct search
        sim_results = graph.search_similar(v1, top_k=3)
        assert len(sim_results) > 0
        assert sim_results[0][0].id == n1.id
        assert np.isclose(sim_results[0][1], 1.0, atol=1e-3)

        # Traverse query
        subgraph = graph.traverse(v1, top_k=2)
        assert len(subgraph.nodes) > 0
        assert subgraph.nodes[0].id == n1.id

    def test_graph_backward_compatibility_faiss_index_proxy(self):
        graph = MayonGraph(concept_dim=16, ann_index_type="auto")
        v = np.random.randn(16).astype(np.float32)
        graph.add_node(vector=v, source_text="Legacy Test")

        # Calling graph.faiss_index.search directly (legacy code path)
        scores, indices = graph.faiss_index.search(v.reshape(1, -1), k=1)
        assert scores.shape == (1, 1)
        assert indices.shape == (1, 1)
        assert indices[0, 0] == 0

    def test_graph_save_and_load_with_vector_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MayonGraph(concept_dim=32, ann_index_type="auto")
            v1 = np.random.randn(32).astype(np.float32)
            v2 = np.random.randn(32).astype(np.float32)

            n1 = graph.add_node(vector=v1, source_text="Source Node", sector="science")
            n2 = graph.add_node(vector=v2, source_text="Target Node", sector="science")
            graph.add_edge(n1.id, n2.id, relation_type="causes", confidence=0.95)

            # Save
            graph.save(tmpdir)

            # Load
            loaded_graph = MayonGraph.load(tmpdir)
            assert loaded_graph.num_nodes == 2
            assert loaded_graph.num_edges == 1
            assert loaded_graph.vector_index.size == 2

            # Query loaded graph
            results = loaded_graph.search_similar(v1, top_k=1)
            assert results[0][0].id == n1.id
