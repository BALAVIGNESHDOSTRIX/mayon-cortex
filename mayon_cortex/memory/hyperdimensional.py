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
Hyperdimensional Computing Layer — The Algebra of Thought
============================================================
Phase Ω of the Brain-Like Intelligence upgrade.

Binary/bipolar vectors of ~10,000 dimensions with THREE operations:
  BIND:    A ⊕ B (XOR)    — combine two concepts
  BUNDLE:  majority(A,B,C) — create a set of concepts
  PERMUTE: shift(A, k)     — encode position/sequence

NO BACKPROPAGATION. CPU-native. Bit operations only.
450x faster than graph neural networks for classification tasks.

Reference: VS-Graph (2024), Kanerva (1988), TorchHD
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HyperVector:
    """
    A high-dimensional binary vector for holographic computation.
    
    Default dimension: 10,000 bits
    Operations are all element-wise and CPU-native (XOR, majority vote).
    """

    def __init__(self, bits_or_dim: Any = 10000, bits: Optional[np.ndarray] = None):
        if bits is not None:
            self.bits = bits
        elif isinstance(bits_or_dim, int):
            self.bits = np.random.randint(0, 2, bits_or_dim, dtype=np.uint8)
        elif isinstance(bits_or_dim, np.ndarray):
            self.bits = bits_or_dim
        else:
            self.bits = np.random.randint(0, 2, 10000, dtype=np.uint8)

    @classmethod
    def random(cls, dim: int = 10000) -> "HyperVector":
        """Create a random hypervector (iid Bernoulli(0.5))."""
        return cls(bits=np.random.randint(0, 2, dim, dtype=np.uint8))

    @classmethod
    def zeros(cls, dim: int = 10000) -> "HyperVector":
        """Create a zero hypervector."""
        return cls(bits=np.zeros(dim, dtype=np.uint8))


    @classmethod
    def from_text(cls, text: str, dim: int = 10000) -> "HyperVector":
        """
        Create a deterministic hypervector from text.
        Uses text as seed for reproducible random generation.
        """
        seed = hash(text.lower().strip()) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        return cls(bits=rng.randint(0, 2, dim, dtype=np.uint8))

    @classmethod
    def from_float_vector(cls, vec: np.ndarray, dim: int = 10000) -> "HyperVector":
        """
        Convert a float vector (e.g., ℝ³⁸⁴) to a binary hypervector.
        Uses random projection: each bit = sign(random_projection · vec).
        """
        seed = int(abs(vec[:4].sum()) * 1e6) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        proj = rng.randn(len(vec), dim).astype(np.float32)
        proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
        projected = vec @ proj
        return cls(bits=(projected > 0).astype(np.uint8))

    @property
    def dim(self) -> int:
        return len(self.bits)

    # ── The Three Operations ──

    @staticmethod
    def bind(a: "HyperVector", b: "HyperVector") -> "HyperVector":
        """
        BIND: Combine two concepts. A ⊕ B (element-wise XOR).
        
        Properties:
          - Self-inverse: bind(bind(A, B), B) ≈ A
          - Dissimilar to both inputs
          - Associative and commutative
          
        Use for: encoding relationships ("red" BIND "car" = "red_car")
        """
        return HyperVector(bits=np.bitwise_xor(a.bits, b.bits))

    @staticmethod
    def bundle(*vectors: "HyperVector") -> "HyperVector":
        """
        BUNDLE: Create a set/superposition. Majority vote over inputs.
        
        Properties:
          - Similar to all inputs
          - Preserves information from all components
          
        Use for: creating category vectors, accumulating evidence
        """
        if not vectors:
            raise ValueError("Need at least one vector to bundle")
        if len(vectors) == 1:
            return HyperVector(bits=vectors[0].bits.copy())

        stacked = np.stack([v.bits for v in vectors])
        threshold = len(vectors) / 2.0

        # Majority vote with random tie-breaking
        sums = stacked.sum(axis=0).astype(np.float32)
        result = np.zeros(vectors[0].dim, dtype=np.uint8)
        result[sums > threshold] = 1
        # Random tie-breaking for exact ties
        ties = sums == threshold
        if np.any(ties):
            result[ties] = np.random.randint(0, 2, np.sum(ties), dtype=np.uint8)

        return HyperVector(bits=result)

    @staticmethod
    def permute(v: "HyperVector", k: int = 1) -> "HyperVector":
        """
        PERMUTE: Encode position/sequence. Circular shift by k positions.
        
        Properties:
          - Dissimilar to the original (for k > ~10)
          - Invertible: permute(permute(v, k), -k) = v
          
        Use for: encoding word order in sentences
        """
        return HyperVector(bits=np.roll(v.bits, k))

    # ── Similarity ──

    def similarity(self, other: "HyperVector") -> float:
        """
        Hamming similarity: fraction of matching bits.
        Returns value in [0, 1] where 1 = identical, 0.5 = random, 0 = complement.
        """
        return 1.0 - np.count_nonzero(self.bits != other.bits) / len(self.bits)

    def cosine_approx(self, other: "HyperVector") -> float:
        """
        Approximate cosine similarity from Hamming distance.
        Maps [0, 1] Hamming similarity to [-1, 1] cosine-like range.
        """
        h_sim = self.similarity(other)
        return 2.0 * h_sim - 1.0

    # ── Utility ──

    def __repr__(self) -> str:
        ones = np.sum(self.bits)
        return f"HyperVector(dim={self.dim}, ones={ones}/{self.dim})"

    def __len__(self) -> int:
        return self.dim


class HyperdimensionalMemory:
    """
    Associative memory using hypervectors.
    
    Store: bundle concept hypervectors into class prototypes
    Retrieve: find nearest match by Hamming distance
    Learn: single-shot, no iteration, no backprop
    
    This is an alternative to FAISS for graph-native search.
    Works purely with bit operations on CPU.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self._prototypes: Dict[str, HyperVector] = {}
        self._item_store: Dict[str, List[HyperVector]] = {}

    def store(self, key: str, value: HyperVector) -> None:
        """
        Store a hypervector under a key.
        If key exists, the new value is bundled with existing prototype.
        """
        if key in self._prototypes:
            self._item_store[key].append(value)
            self._prototypes[key] = HyperVector.bundle(
                *self._item_store[key]
            )
        else:
            self._item_store[key] = [value]
            self._prototypes[key] = HyperVector(bits=value.bits.copy())

    def retrieve(self, query: HyperVector, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar stored prototypes to a query.
        Returns list of (key, similarity) pairs sorted by similarity.
        """
        if not self._prototypes:
            return []

        similarities = []
        for key, proto in self._prototypes.items():
            sim = query.similarity(proto)
            similarities.append((key, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def learn_one_shot(self, key: str, example: HyperVector) -> None:
        """
        One-shot learning: store a single example.
        No iteration. No backprop. Immediate.
        """
        self.store(key, example)

    def forget(self, key: str) -> None:
        """Remove a stored prototype."""
        self._prototypes.pop(key, None)
        self._item_store.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._prototypes)


class HyperdimensionalEncoder:
    """
    Encode structured data as hypervectors for graph-native computation.
    
    Encodes:
      - Concepts (text → deterministic hypervector)
      - Relations (concept BIND concept = relation vector)
      - Sequences (concept at position = concept PERMUTE position)
      - Sets (bundle of multiple concepts)
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self._codebook: Dict[str, HyperVector] = {}

    def encode_concept(self, text: str) -> HyperVector:
        """Encode a concept as a deterministic hypervector."""
        key = text.lower().strip()
        if key not in self._codebook:
            self._codebook[key] = HyperVector.from_text(key, self.dim)
        return self._codebook[key]

    def encode_relation(self, subject: str, relation: str, obj: str) -> HyperVector:
        """
        Encode a relation triple as a single hypervector.
        triple = subject BIND relation BIND object
        """
        s = self.encode_concept(subject)
        r = self.encode_concept(relation)
        o = self.encode_concept(obj)
        return HyperVector.bind(HyperVector.bind(s, r), o)

    def encode_sequence(self, tokens: List[str]) -> HyperVector:
        """
        Encode an ordered sequence as a single hypervector.
        sequence = BUNDLE(token₁ PERMUTE 1, token₂ PERMUTE 2, ...)
        """
        if not tokens:
            return HyperVector.zeros(self.dim)

        permuted = []
        for i, token in enumerate(tokens):
            tv = self.encode_concept(token)
            pv = HyperVector.permute(tv, i + 1)
            permuted.append(pv)

        return HyperVector.bundle(*permuted)

    def encode_set(self, items: List[str]) -> HyperVector:
        """
        Encode an unordered set as a single hypervector.
        set = BUNDLE(item₁, item₂, ...)
        """
        if not items:
            return HyperVector.zeros(self.dim)

        vectors = [self.encode_concept(item) for item in items]
        return HyperVector.bundle(*vectors)

    def decode_nearest(self, query: HyperVector, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the nearest concepts to a query hypervector."""
        if not self._codebook:
            return []

        sims = []
        for key, vec in self._codebook.items():
            sim = query.similarity(vec)
            sims.append((key, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]
