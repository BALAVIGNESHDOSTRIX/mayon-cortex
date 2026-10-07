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
Visual Spatial Graph — Positional Encodings & Spatial Relation Topology
=======================================================================
Stores 8-directional spatial transitions, positional priors, and
co-occurrence statistics between visual codebook entries.

Equivalent to 2D relative positional embeddings in Vision Transformers (ViT),
derived directly from empirical image spatial structures.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


DIRECTION_OFFSETS: Dict[str, Tuple[int, int]] = {
    "right": (0, 1),
    "below": (1, 0),
    "left": (0, -1),
    "above": (-1, 0),
    "right_below": (1, 1),
    "left_below": (1, -1),
    "right_above": (-1, 1),
    "left_above": (-1, -1),
}


@dataclass
class SpatialEdge:
    """Directed spatial relationship between two visual codebook tokens."""
    source_id: int
    target_id: int
    relation: str  # e.g. "right", "below"
    weight: float = 1.0
    count: int = 1


class VisualSpatialGraph:
    """
    Learns 2D spatial arrangement grammar of visual tokens.
    """

    def __init__(self, num_entries: int = 1024, max_samples_per_token: int = 500):
        self.num_entries = num_entries
        self.max_samples_per_token = max_samples_per_token
        self.reset()

    def reset(self):
        """Clear all transitions, positional distributions, and co-occurrences."""
        self.transitions: Dict[str, Dict[int, Dict[int, float]]] = {
            d: defaultdict(lambda: defaultdict(float)) for d in DIRECTION_OFFSETS
        }
        self.position_samples: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.label_position_samples: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        self.co_occurrences: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def _add_position_sample(self, samples_list: List[Tuple[float, float]], sample: Tuple[float, float]):
        """Reservoir sampling to cap positional sample storage at max_samples_per_token."""
        if len(samples_list) < self.max_samples_per_token:
            samples_list.append(sample)
        else:
            idx = int(np.random.randint(0, len(samples_list) + 1))
            if idx < self.max_samples_per_token:
                samples_list[idx] = sample

    def learn_from_grid(
        self,
        grid_tokens: List[List[int]],
        label: Optional[str] = None,
        learning_rate: float = 0.1,
    ):
        """
        Extract spatial transitions and positional priors from a 2D token grid.
        
        Args:
            grid_tokens: 2D matrix of codebook token IDs [rows][cols]
            label: Optional concept label for label-conditioned spatial priors
            learning_rate: Hebbian reinforcement factor
        """
        rows = len(grid_tokens)
        if rows == 0:
            return
        cols = len(grid_tokens[0])
        if cols == 0:
            return

        unique_in_image: Set[int] = set()
        norm_label = label.lower().strip() if label else None

        for r in range(rows):
            for c in range(cols):
                curr_token = grid_tokens[r][c]
                unique_in_image.add(curr_token)

                # Record normalized spatial position [0, 1] via reservoir sampling
                norm_r = r / max(1, rows - 1)
                norm_c = c / max(1, cols - 1)
                self._add_position_sample(self.position_samples[curr_token], (norm_r, norm_c))
                if norm_label:
                    self._add_position_sample(self.label_position_samples[norm_label][curr_token], (norm_r, norm_c))

                # Check 8 spatial directions
                for rel_name, (dr, dc) in DIRECTION_OFFSETS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_token = grid_tokens[nr][nc]
                        current_prob = self.transitions[rel_name][curr_token][neighbor_token]
                        # Hebbian reinforcement: P_new = P_old + lr * (1 - P_old)
                        self.transitions[rel_name][curr_token][neighbor_token] = (
                            current_prob + learning_rate * (1.0 - current_prob)
                        )

        # Update pairwise co-occurrences
        token_list = list(unique_in_image)
        for i in range(len(token_list)):
            for j in range(i, len(token_list)):
                t1, t2 = token_list[i], token_list[j]
                self.co_occurrences[t1][t2] += 1
                if t1 != t2:
                    self.co_occurrences[t2][t1] += 1

    def get_transition_probability(
        self,
        source_token: int,
        target_token: int,
        direction: str,
    ) -> float:
        """Return learned probability of target appearing at specified direction from source."""
        if direction not in self.transitions:
            return 0.0
        return self.transitions[direction][source_token].get(target_token, 0.0)

    def get_position_fitness(
        self,
        token: int,
        norm_r: float,
        norm_c: float,
        label: Optional[str] = None,
        sigma: float = 0.08,
    ) -> float:
        """
        Compute Gaussian likelihood score of a token occurring at normalized grid coordinate (norm_r, norm_c).
        Conditioned on label for sharp anatomical/scene localization.
        """
        norm_label = label.lower().strip() if label else None
        samples = None
        if norm_label and norm_label in self.label_position_samples:
            samples = self.label_position_samples[norm_label].get(token)

        if not samples:
            samples = self.position_samples.get(token)

        if not samples:
            return 0.2  # Low prior for unseen positions

        # Gaussian kernel density estimate with focused sigma
        coords = np.array(samples)
        dists_sq = (coords[:, 0] - norm_r) ** 2 + (coords[:, 1] - norm_c) ** 2
        densities = np.exp(-dists_sq / (2.0 * sigma * sigma))
        return float(np.max(densities))

    def get_co_occurrence_score(self, token1: int, token2: int) -> float:
        """Return normalized co-occurrence score between two visual tokens."""
        count = self.co_occurrences[token1].get(token2, 0)
        c1 = sum(self.co_occurrences[token1].values()) or 1
        c2 = sum(self.co_occurrences[token2].values()) or 1
        # Jaccard-like co-occurrence index
        return count / (c1 + c2 - count + 1e-6)
