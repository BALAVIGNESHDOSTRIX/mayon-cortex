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
Mayon-Cortex Unified High-Performance Vector Index Engine
=========================================================
Polymorphic vector indexing architecture supporting:
  1. USearchIndex    — Ultra-fast, SIMD-accelerated (AVX-512/AVX2/NEON) HNSW index with mmap support.
  2. FaissIndex      — Industry-standard FAISS Flat/IVF inner-product index.
  3. NumpyIndex      — Zero-dependency pure CPU BLAS vectorized matrix engine with exact retrieval.

All backends expose an identical interface with (scores, indices) output format.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Backend availability probes
_USEARCH_AVAILABLE = False
_FAISS_AVAILABLE = False

try:
    import usearch.index
    _USEARCH_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    pass


class BaseVectorIndex(ABC):
    """Abstract Base Class for all Mayon-Cortex Vector Index Backends."""

    def __init__(self, dim: int, metric: str = "cos"):
        self.dim = dim
        self.metric = metric

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the underlying index backend."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently indexed."""
        pass

    @abstractmethod
    def add(self, vector: np.ndarray, key: Optional[int] = None) -> None:
        """Add a single vector with an integer identifier."""
        pass

    @abstractmethod
    def add_batch(self, vectors: np.ndarray, keys: Optional[Sequence[int]] = None) -> None:
        """Add a batch of vectors with optional integer identifiers."""
        pass

    @abstractmethod
    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query: Query vector of shape (D,) or (B, D).
            k: Number of neighbors to return.
            
        Returns:
            Tuple[scores, indices]:
                scores: Similarity scores of shape (B, k) or (1, k).
                indices: Integer keys/indices of shape (B, k) or (1, k).
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Serialize index to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Deserialize index from disk."""
        pass

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# 1. USearch High-Performance SIMD HNSW Index
# ─────────────────────────────────────────────────────────────────────────────
class USearchVectorIndex(BaseVectorIndex):
    """
    USearch Vector Index:
    - Zero-copy memory mapped files (mmap)
    - Hardware SIMD optimization (AVX2, AVX-512, ARM NEON, SVE)
    - Scalable HNSW graph structure with sub-millisecond latency at > 100k nodes.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 32,
        expansion_add: int = 256,
        expansion_search: int = 128,
    ):
        super().__init__(dim=dim, metric=metric)
        if not _USEARCH_AVAILABLE:
            raise ImportError(
                "usearch is not installed. Run `pip install usearch` or use backend='faiss'/'numpy'."
            )

        self.dtype = dtype
        self._next_key = 0
        self._index = usearch.index.Index(
            ndim=dim,
            metric=metric,
            dtype=dtype,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )

    @property
    def backend_name(self) -> str:
        return f"usearch-hnsw-{self.dtype}"

    @property
    def size(self) -> int:
        return len(self._index)

    def add(self, vector: np.ndarray, key: Optional[int] = None) -> None:
        if key is None:
            key = self._next_key
            self._next_key += 1
        else:
            self._next_key = max(self._next_key, key + 1)

        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        self._index.add(key, vec)

    def add_batch(self, vectors: np.ndarray, keys: Optional[Sequence[int]] = None) -> None:
        N = len(vectors)
        if N == 0:
            return

        if keys is None:
            keys = np.arange(self._next_key, self._next_key + N, dtype=np.int64)
            self._next_key += N
        else:
            keys = np.asarray(keys, dtype=np.int64)
            self._next_key = max(self._next_key, int(np.max(keys)) + 1)

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(keys, vecs)

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.full((1, 0), -1, dtype=np.int64)

        q = np.ascontiguousarray(query, dtype=np.float32)
        is_single = q.ndim == 1
        if is_single:
            q = q.reshape(1, -1)

        B = q.shape[0]
        actual_k = min(k, self.size)

        all_scores = []
        all_indices = []

        # Execute search for each query vector in batch
        for b in range(B):
            matches = self._index.search(q[b], actual_k)
            # USearch matches contain keys and distances
            keys = np.asarray(matches.keys, dtype=np.int64)
            dists = np.asarray(matches.distances, dtype=np.float32)

            # Convert distance to similarity score
            if self.metric == "cos":
                scores = 1.0 - dists
            elif self.metric == "ip":
                scores = dists
            else:
                scores = -dists

            # Pad with -1 if fewer than k returned
            if len(keys) < k:
                pad_len = k - len(keys)
                keys = np.pad(keys, (0, pad_len), constant_values=-1)
                scores = np.pad(scores, (0, pad_len), constant_values=0.0)

            all_scores.append(scores[:k])
            all_indices.append(keys[:k])

        return np.vstack(all_scores), np.vstack(all_indices)

    def save(self, path: Union[str, Path]) -> None:
        p = str(path)
        self._index.save(p)

    def load(self, path: Union[str, Path]) -> None:
        p = str(path)
        self._index.load(p)
        if len(self._index) > 0:
            self._next_key = len(self._index)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAISS Vector Index (Flat / IVF)
# ─────────────────────────────────────────────────────────────────────────────
class FaissVectorIndex(BaseVectorIndex):
    """
    FAISS-backed Vector Index wrapping IndexFlatIP or IndexIVFFlat.
    """

    def __init__(self, dim: int, metric: str = "cos", index_type: str = "flat"):
        super().__init__(dim=dim, metric=metric)
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is not installed. Run `pip install faiss-cpu` or use backend='usearch'/'numpy'."
            )

        self.index_type = index_type
        self._size = 0
        self._build_index()

    def _build_index(self):
        if self.index_type == "flat" or self.index_type == "ip":
            self._index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.dim, 16)
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = faiss.IndexFlatIP(self.dim)

    @property
    def backend_name(self) -> str:
        return f"faiss-{self.index_type}"

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, vector: np.ndarray, key: Optional[int] = None) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        self._index.add(vec)
        self._size += 1

    def add_batch(self, vectors: np.ndarray, keys: Optional[Sequence[int]] = None) -> None:
        if len(vectors) == 0:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vecs)
        self._size += len(vectors)

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.full((1, 0), -1, dtype=np.int64)

        q = np.ascontiguousarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        actual_k = min(k, self.size)
        scores, indices = self._index.search(q, actual_k)
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, path: Union[str, Path]) -> None:
        faiss.write_index(self._index, str(path))

    def load(self, path: Union[str, Path]) -> None:
        self._index = faiss.read_index(str(path))
        self._size = self._index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# 3. Native NumPy / BLAS Matrix Index (Zero C++ Dependencies)
# ─────────────────────────────────────────────────────────────────────────────
class NumpyVectorIndex(BaseVectorIndex):
    """
    Pure Python & NumPy BLAS Vector Index:
    - 100% deterministic, zero external C++ extensions
    - Vectorized matrix multiplication with hardware-accelerated BLAS (OpenBLAS/MKL)
    - Sub-millisecond exact cosine similarity for up to 50k vectors.
    """

    def __init__(self, dim: int, metric: str = "cos"):
        super().__init__(dim=dim, metric=metric)
        self._vectors: Optional[np.ndarray] = None
        self._keys: List[int] = []

    @property
    def backend_name(self) -> str:
        return "numpy-blas"

    @property
    def size(self) -> int:
        return len(self._keys)

    def add(self, vector: np.ndarray, key: Optional[int] = None) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        k = len(self._keys) if key is None else key

        if self._vectors is None:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])
        self._keys.append(k)

    def add_batch(self, vectors: np.ndarray, keys: Optional[Sequence[int]] = None) -> None:
        N = len(vectors)
        if N == 0:
            return

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if keys is None:
            new_keys = list(range(len(self._keys), len(self._keys) + N))
        else:
            new_keys = list(keys)

        if self._vectors is None:
            self._vectors = vecs
        else:
            self._vectors = np.vstack([self._vectors, vecs])
        self._keys.extend(new_keys)

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.size == 0 or self._vectors is None:
            return np.zeros((1, 0), dtype=np.float32), np.full((1, 0), -1, dtype=np.int64)

        q = np.ascontiguousarray(query, dtype=np.float32)
        is_single = q.ndim == 1
        if is_single:
            q = q.reshape(1, -1)

        actual_k = min(k, self.size)
        # Compute cosine similarity via matrix multiply (assuming vectors are normalized)
        # sim matrix of shape (B, N)
        sims = np.dot(q, self._vectors.T)

        B = q.shape[0]
        batch_scores = []
        batch_indices = []

        keys_arr = np.array(self._keys, dtype=np.int64)

        for b in range(B):
            row_sims = sims[b]
            if actual_k < self.size:
                # Fast top-k using partition
                top_part = np.argpartition(-row_sims, actual_k - 1)[:actual_k]
                sorted_top = top_part[np.argsort(-row_sims[top_part])]
            else:
                sorted_top = np.argsort(-row_sims)

            scores = row_sims[sorted_top]
            indices = keys_arr[sorted_top]

            if len(indices) < k:
                pad_len = k - len(indices)
                indices = np.pad(indices, (0, pad_len), constant_values=-1)
                scores = np.pad(scores, (0, pad_len), constant_values=0.0)

            batch_scores.append(scores[:k])
            batch_indices.append(indices[:k])

        return np.vstack(batch_scores).astype(np.float32), np.vstack(batch_indices).astype(np.int64)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not str(p).endswith(".npz"):
            p = Path(f"{p}.npz")
        np.savez_compressed(
            p,
            vectors=self._vectors if self._vectors is not None else np.zeros((0, self.dim), dtype=np.float32),
            keys=np.array(self._keys, dtype=np.int64),
            dim=self.dim,
        )

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not p.exists() and Path(f"{p}.npz").exists():
            p = Path(f"{p}.npz")
        data = np.load(p)
        self._vectors = data["vectors"]
        self._keys = data["keys"].tolist()
        self.dim = int(data.get("dim", self._vectors.shape[1] if len(self._vectors) > 0 else self.dim))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Factory and Compatibility Helpers
# ─────────────────────────────────────────────────────────────────────────────
class VectorIndexFactory:
    """
    Factory to instantiate the best available high-performance vector index.
    Priority order in 'auto' mode:
      1. USearch (SIMD HNSW with sub-ms scaling)
      2. FAISS (Flat/IVF)
      3. NumPy BLAS (Deterministic pure CPU fallback)
    """

    @staticmethod
    def create_index(
        dim: int,
        backend: str = "auto",
        metric: str = "cos",
        dtype: str = "f32",
        **kwargs,
    ) -> BaseVectorIndex:
        backend_lower = (backend or "auto").lower()

        # Handle explicit backend choices
        if backend_lower in ("usearch", "hnsw", "usearch-hnsw"):
            if _USEARCH_AVAILABLE:
                return USearchVectorIndex(dim=dim, metric=metric, dtype=dtype, **kwargs)
            logger.warning("USearch requested but not available. Falling back to FAISS/NumPy.")

        if backend_lower in ("faiss", "faiss-flat", "faiss-ivf", "flat", "ivf"):
            if _FAISS_AVAILABLE:
                index_type = "ivf" if "ivf" in backend_lower else "flat"
                return FaissVectorIndex(dim=dim, metric=metric, index_type=index_type)
            logger.warning("FAISS requested but not available. Falling back to NumPy.")

        if backend_lower in ("numpy", "numpy-blas", "native"):
            return NumpyVectorIndex(dim=dim, metric=metric)

        # 'auto' mode: Probe best available
        if _USEARCH_AVAILABLE:
            return USearchVectorIndex(dim=dim, metric=metric, dtype=dtype, **kwargs)
        elif _FAISS_AVAILABLE:
            return FaissVectorIndex(dim=dim, metric=metric, index_type="flat")
        else:
            return NumpyVectorIndex(dim=dim, metric=metric)
