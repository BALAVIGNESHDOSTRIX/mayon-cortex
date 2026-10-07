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
Working Memory — The Brain's Prefrontal Cortex
================================================
Short-term conversational buffer that maintains context across
multiple turns of interaction.

Without working memory, each query is processed in isolation —
like a goldfish. With it, the system can follow up on previous
questions, refer to earlier context, and maintain coherent dialogue.

Adapted from mayon-net's working memory module for use with Cortex-Graph.

Usage:
    wm = WorkingMemory(config)
    wm.push(query="What treats hypertension?", query_vec=vec, answer="Lisinopril")
    augmented_vec = wm.augment_query(new_query_vec)  # blends with context
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mayon_cortex.core.config import CortexConfig


@dataclass
class MemoryItem:
    """A single item in working memory (one exchange)."""
    query_text: str
    query_vec: np.ndarray
    answer_text: str
    confidence: float
    sector: str
    timestamp: float
    retrieved_node_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class WorkingMemory:
    """
    Short-term conversation buffer — the brain's prefrontal cortex.

    Maintains the last N exchanges and provides:
    1. Context vector computation (recency-weighted average)
    2. Query augmentation (blend current query with conversation history)
    3. Topic continuity detection
    4. Conversation summarization

    Architecture:
    - Fixed-size circular buffer (default 10 items)
    - Recency-weighted: latest = weight 1.0, oldest = weight 0.1
    - No persistence — working memory is session-scoped
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.max_items = self.config.wm_max_items
        self.concept_dim = self.config.concept_dim
        self.blend_alpha = self.config.wm_context_blend
        self.topic_threshold = self.config.wm_topic_threshold
        self.buffer: List[MemoryItem] = []

    def push(
        self,
        query_text: str,
        query_vec: np.ndarray,
        answer_text: str,
        confidence: float = 0.5,
        sector: str = "general",
        retrieved_node_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a new exchange to working memory."""
        item = MemoryItem(
            query_text=query_text,
            query_vec=query_vec.copy(),
            answer_text=answer_text,
            confidence=confidence,
            sector=sector,
            timestamp=time.time(),
            retrieved_node_ids=retrieved_node_ids or [],
            metadata=metadata or {},
        )

        self.buffer.append(item)

        # Evict oldest if full
        if len(self.buffer) > self.max_items:
            self.buffer = self.buffer[-self.max_items:]

    def get_context_vector(self) -> Optional[np.ndarray]:
        """
        Compute recency-weighted context vector from conversation history.
        Latest items have weight 1.0, oldest have weight 0.1.
        """
        if not self.buffer:
            return None

        n = len(self.buffer)
        # Recency weights: linear from 0.1 (oldest) to 1.0 (newest)
        weights = np.linspace(0.1, 1.0, n)
        weights /= weights.sum()

        context = np.zeros(self.concept_dim, dtype=np.float32)
        for i, item in enumerate(self.buffer):
            vec = item.query_vec
            if len(vec) == self.concept_dim:
                context += weights[i] * vec

        norm = np.linalg.norm(context)
        if norm > 0:
            context /= norm

        return context

    def augment_query(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Blend current query with conversation context.
        Returns: (1-α) * query + α * context
        """
        context = self.get_context_vector()
        if context is None:
            return query_vec

        blended = (1 - self.blend_alpha) * query_vec + self.blend_alpha * context
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm

        return blended.astype(np.float32)

    def is_same_topic(self, query_vec: np.ndarray) -> bool:
        """Check if the new query is on the same topic as recent conversation."""
        if not self.buffer:
            return False

        last_vec = self.buffer[-1].query_vec
        q_norm = np.linalg.norm(query_vec)
        l_norm = np.linalg.norm(last_vec)

        if q_norm > 0 and l_norm > 0:
            sim = float(np.dot(query_vec, last_vec) / (q_norm * l_norm))
            return sim >= self.topic_threshold

        return False

    def get_recent_sectors(self) -> List[str]:
        """Get sectors from recent conversation for context-aware sector filtering."""
        if not self.buffer:
            return []
        return list(set(item.sector for item in self.buffer[-3:]))

    def get_recent_node_ids(self) -> List[str]:
        """Get recently retrieved node IDs for continuity."""
        ids = []
        for item in reversed(self.buffer[-3:]):
            ids.extend(item.retrieved_node_ids)
        return ids

    def summarize(self) -> str:
        """Summarize conversation history as text for context injection."""
        if not self.buffer:
            return ""

        parts = []
        for item in self.buffer[-5:]:  # Last 5 exchanges
            parts.append(f"Q: {item.query_text}")
            if item.answer_text:
                parts.append(f"A: {item.answer_text[:200]}")

        return "\n".join(parts)

    def clear(self):
        """Clear all working memory (new conversation)."""
        self.buffer.clear()

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
