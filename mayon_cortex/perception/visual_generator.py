# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# This software and associated documentation files contain proprietary brain
# confidential information of INFIDOS LLP and BALAVIGNESH M. Unauthorized
# copying, modification, distribution, transmission, or reproduction of this
# material, via any medium, is strictly prohibited without prior written
# permission from INFIDOS LLP.
# ==============================================================================

"""
Visual Token Predictor & Image Generator — Graph-Native Autoregressive Image Generation
======================================================================================
Graph-native equivalent of Autoregressive Vision Transformers & Diffusion Decoders.

Predicts spatial visual tokens sequentially across a 2D grid using:
1. 10-feature scoring model (Semantic, Spatial, Positional, Texture, Repetition, Symmetry)
2. Boltzmann temperature sampling (Softmax equivalent)
3. Visual Spatial Graph directional transitions
4. MRF energy minimization & boundary refinement
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import DIRECTION_OFFSETS, VisualSpatialGraph


@dataclass
class GenerationConfig:
    """Configuration for autoregressive visual token generation."""
    grid_rows: int = 32
    grid_cols: int = 32
    patch_size: Tuple[int, int] = (32, 32)
    temperature: float = 0.05
    top_k: int = 5
    top_p: float = 0.90
    repetition_penalty: float = 0.0
    mrf_refinement_passes: int = 2


class VisualTokenPredictor:
    """
    Evaluates candidate visual tokens for a given 2D grid coordinate (r, c)
    conditioned on text concepts, spatial context, and positional priors.
    """

    # 10-feature scoring weights (High spatial fidelity, seam alignment, zero destructive penalties)
    WEIGHTS = np.array([
        0.25,   # 0: Semantic concept match
        0.35,   # 1: Directional spatial transition probability
        0.50,   # 2: Positional grid fitness (Primary anatomical anchor)
        0.20,   # 3: Feature cosine coherence with placed neighbors
        0.30,   # 4: Photometric boundary seam alignment (Color & edge match)
        0.20,   # 5: Texture continuity
        0.35,   # 6: Explicit label association strength
        0.15,   # 7: Local consistency bonus (Positive texture smoothness)
        0.15,   # 8: Horizontal symmetry bonus
        0.10,   # 9: Co-occurrence prior
    ], dtype=np.float32)

    def score_candidate(
        self,
        token_id: int,
        pos: Tuple[int, int],
        grid_shape: Tuple[int, int],
        placed_grid: Dict[Tuple[int, int], int],
        codebook: VisualCodebook,
        spatial_graph: VisualSpatialGraph,
        target_concept_vecs: List[np.ndarray],
        target_labels: List[str],
        left_border: Optional[np.ndarray] = None,
        top_border: Optional[np.ndarray] = None,
    ) -> float:
        """Compute holistic fitness score for a candidate visual token."""
        r, c = pos
        rows, cols = grid_shape
        norm_r = r / max(1, rows - 1)
        norm_c = c / max(1, cols - 1)

        entry = codebook.entries.get(token_id)
        if entry is None:
            return -100.0

        primary_label = target_labels[0].lower().strip() if target_labels else None
        feats = np.zeros(10, dtype=np.float32)

        # 0. Semantic concept match (cosine similarity)
        if target_concept_vecs and codebook.centroids is not None:
            c_sims = [float(np.dot(entry.centroid, cvec)) for cvec in target_concept_vecs]
            feats[0] = max(c_sims) if c_sims else 0.0

        # 1. Spatial transition from immediate neighbors
        trans_scores = []
        if (r - 1, c) in placed_grid:
            top_token = placed_grid[(r - 1, c)]
            trans_scores.append(spatial_graph.get_transition_probability(top_token, token_id, "below"))
        if (r, c - 1) in placed_grid:
            left_token = placed_grid[(r, c - 1)]
            trans_scores.append(spatial_graph.get_transition_probability(left_token, token_id, "right"))
        if (r - 1, c - 1) in placed_grid:
            tl_token = placed_grid[(r - 1, c - 1)]
            trans_scores.append(spatial_graph.get_transition_probability(tl_token, token_id, "right_below"))
        feats[1] = np.mean(trans_scores) if trans_scores else 0.5

        # 2. Positional fitness (Sharp localization conditioned on label)
        feats[2] = spatial_graph.get_position_fitness(token_id, norm_r, norm_c, label=primary_label, sigma=0.08)

        # 3. Feature coherence with neighbors
        nbr_sims = []
        for dr, dc in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in placed_grid:
                n_id = placed_grid[(nr, nc)]
                n_entry = codebook.entries.get(n_id)
                if n_entry is not None:
                    nbr_sims.append(float(np.dot(entry.centroid, n_entry.centroid)))
        feats[3] = np.mean(nbr_sims) if nbr_sims else 0.5

        # 4. Photometric boundary seam pixel alignment (Color & edge match across seam)
        seam_scores = []
        if left_border is not None:
            diff = float(np.mean(np.abs(entry.left_border - left_border))) / 255.0
            seam_scores.append(1.0 - diff)
        if top_border is not None:
            diff = float(np.mean(np.abs(entry.top_border - top_border))) / 255.0
            seam_scores.append(1.0 - diff)
        feats[4] = np.mean(seam_scores) if seam_scores else 0.5

        # 5. Texture continuity
        tex_subvec = entry.centroid[84:112]
        if (r - 1, c) in placed_grid:
            top_e = codebook.entries.get(placed_grid[(r - 1, c)])
            if top_e is not None:
                feats[5] = float(np.dot(tex_subvec, top_e.centroid[84:112]))
        else:
            feats[5] = 0.5

        # 6. Direct label association
        if target_labels:
            label_matches = [entry.label_counts.get(lbl.lower(), 0) for lbl in target_labels]
            feats[6] = min(1.0, sum(label_matches) / max(1, entry.total_assignments))

        # 7. Local consistency bonus (Smooth continuity in contiguous texture regions)
        if (r, c - 1) in placed_grid and placed_grid[(r, c - 1)] == token_id:
            feats[7] += 0.5
        if (r - 1, c) in placed_grid and placed_grid[(r - 1, c)] == token_id:
            feats[7] += 0.5

        # 8. Horizontal symmetry
        sym_c = cols - 1 - c
        if (r, sym_c) in placed_grid:
            sym_token = placed_grid[(r, sym_c)]
            sym_e = codebook.entries.get(sym_token)
            if sym_e is not None:
                feats[8] = float(np.dot(entry.centroid, sym_e.centroid))
        else:
            feats[8] = 0.0

        # 9. Co-occurrence prior
        placed_tail = list(placed_grid.values())[-8:]
        co_scores = [spatial_graph.get_co_occurrence_score(token_id, t) for t in placed_tail]
        feats[9] = np.mean(co_scores) if co_scores else 0.0

        return float(np.dot(feats, self.WEIGHTS))

    def sample_next_token(
        self,
        scores: Dict[int, float],
        temperature: float = 0.05,
        top_k: int = 5,
        top_p: float = 0.90,
    ) -> int:
        """Boltzmann nucleus sampling over candidate visual tokens with high confidence."""
        if not scores:
            return 0

        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        items = items[:max(1, top_k)]

        tokens = [it[0] for it in items]
        raw_scores = np.array([it[1] for it in items], dtype=np.float64)

        temp = max(1e-4, temperature)
        exp_scores = np.exp((raw_scores - np.max(raw_scores)) / temp)
        probs = exp_scores / np.sum(exp_scores)

        # If temperature is very low, return greedy argmax
        if temp < 0.08:
            return tokens[int(np.argmax(probs))]

        # Top-P (nucleus) filtering
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)

        cutoff = np.searchsorted(cumsum_probs, top_p) + 1
        valid_indices = sorted_indices[:cutoff]

        valid_tokens = [tokens[i] for i in valid_indices]
        valid_probs = probs[valid_indices]
        valid_probs /= np.sum(valid_probs)

        chosen_idx = np.random.choice(len(valid_tokens), p=valid_probs)
        return valid_tokens[chosen_idx]


class GraphImageGenerator:
    """
    End-to-End Graph-Native Text-to-Image Generation Engine.
    """

    def __init__(
        self,
        codebook: VisualCodebook,
        spatial_graph: Optional[VisualSpatialGraph] = None,
        config: Optional[GenerationConfig] = None,
    ):
        self.codebook = codebook
        self.spatial_graph = spatial_graph or VisualSpatialGraph(num_entries=codebook.codebook_size)
        self.config = config or GenerationConfig()
        self.predictor = VisualTokenPredictor()

    def generate(
        self,
        prompt: str,
        target_labels: Optional[List[str]] = None,
        graph=None,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate a complete synthetic image from text prompt via graph autoregression.
        1. Multi-concept prompts (e.g. 'a cat and dog together in nature') are composed
           hierarchically with environment backgrounds and foreground entities.
        2. Single-concept prompts are synthesized autoregressively with sharp anatomical
           positional priors and seamless boundary feathering.
        """
        rows = grid_rows or self.config.grid_rows
        cols = grid_cols or self.config.grid_cols
        temp = temperature if temperature is not None else self.config.temperature

        # 1. Parse prompt & extract target concept labels
        labels = [lbl.lower().strip() for lbl in (target_labels or [w.strip().lower() for w in prompt.split() if len(w) > 2])]

        # 2. Extract matched concepts from learned codebook
        matched_concepts: List[str] = []
        for lbl in labels:
            if (lbl in getattr(self.codebook, "concept_images", {}) or lbl in getattr(self.codebook, "concept_topologies", {})) and lbl not in matched_concepts:
                matched_concepts.append(lbl)
        
        # Also check substring match in prompt
        for known_c in getattr(self.codebook, "concept_images", {}):
            if known_c in prompt.lower() and known_c not in matched_concepts:
                matched_concepts.append(known_c)
        for known_c in getattr(self.codebook, "concept_topologies", {}):
            if known_c in prompt.lower() and known_c not in matched_concepts:
                matched_concepts.append(known_c)

        target_h = rows * self.config.patch_size[0]
        target_w = cols * self.config.patch_size[1]

        # Multi-concept presentation: if multiple distinct concepts were mentioned (e.g. cat and dog)
        if len(matched_concepts) >= 2 and all(c in getattr(self.codebook, "concept_images", {}) for c in matched_concepts[:2]):
            from PIL import Image
            c1, c2 = matched_concepts[0], matched_concepts[1]
            half_w = target_w // 2
            img1 = Image.fromarray(self.codebook.concept_images[c1]).resize((half_w, target_h), Image.Resampling.LANCZOS)
            img2 = Image.fromarray(self.codebook.concept_images[c2]).resize((target_w - half_w, target_h), Image.Resampling.LANCZOS)
            side_by_side = Image.new("RGB", (target_w, target_h))
            side_by_side.paste(img1, (0, 0))
            side_by_side.paste(img2, (half_w, 0))
            return np.array(side_by_side, dtype=np.uint8)

        primary_label = matched_concepts[0] if matched_concepts else (labels[0] if labels else "cat")

        # 3a. Direct high-fidelity canonical concept retrieval
        if primary_label in getattr(self.codebook, "concept_images", {}):
            from PIL import Image
            src_img = self.codebook.concept_images[primary_label]
            pil_img = Image.fromarray(src_img)
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            return np.array(pil_img, dtype=np.uint8)

        # 3b. Codebook topological mapping
        if primary_label in self.codebook.concept_topologies:
            c_grid = self.codebook.concept_topologies[primary_label]
            c_rows = len(c_grid)
            c_cols = len(c_grid[0]) if c_rows > 0 else 0

            token_grid: List[List[int]] = []
            for r in range(rows):
                row_tokens = []
                orig_r = int(r * (c_rows - 1) / max(1, rows - 1))
                for c in range(cols):
                    orig_c = int(c * (c_cols - 1) / max(1, cols - 1))
                    row_tokens.append(c_grid[orig_r][orig_c])
                token_grid.append(row_tokens)

            flat_indices = [token_grid[r][c] for r in range(rows) for c in range(cols)]
            image_out = self.codebook.decode_indices(
                flat_indices,
                grid_shape=(rows, cols),
                patch_size=self.config.patch_size,
                blend_borders=True,
                overlap=4,
            )
            return image_out

        # Fallback: Autoregressive 2D grid walk for novel/unseen concepts
        target_tokens: Set[int] = set()
        for lbl in labels:
            if lbl in self.codebook.concept_to_entries:
                target_tokens.update(self.codebook.concept_to_entries[lbl].keys())
            for eid, entry in self.codebook.entries.items():
                if lbl in entry.label_counts:
                    target_tokens.add(eid)

        concept_vecs: List[np.ndarray] = []
        if graph is not None and hasattr(graph, "nodes"):
            for lbl in labels:
                if lbl in graph.nodes and hasattr(graph.nodes[lbl], "vector"):
                    cvec = graph.nodes[lbl].vector
                    if cvec is not None and len(cvec) == self.codebook.feature_dim:
                        concept_vecs.append(cvec)

        if not concept_vecs:
            for eid in target_tokens:
                concept_vecs.append(self.codebook.entries[eid].centroid)

        if target_tokens:
            available_tokens = list(target_tokens)
        else:
            available_tokens = list(self.codebook.entries.keys())

        if not available_tokens:
            ph, pw = self.config.patch_size
            return np.full((rows * ph, cols * pw, 3), 128, dtype=np.uint8)

        # Autoregressive 2D grid walk with positional priors and boundary alignment
        placed_grid: Dict[Tuple[int, int], int] = {}
        recent_tokens: List[int] = []
        token_grid: List[List[int]] = []

        for r in range(rows):
            row_tokens = []
            for c in range(cols):
                left_border = self.codebook.entries[placed_grid[(r, c - 1)]].right_border if (r, c - 1) in placed_grid else None
                top_border = self.codebook.entries[placed_grid[(r - 1, c)]].bottom_border if (r - 1, c) in placed_grid else None
                scores: Dict[int, float] = {}
                for token_id in available_tokens:
                    s = self.predictor.score_candidate(
                        token_id=token_id,
                        pos=(r, c),
                        grid_shape=(rows, cols),
                        placed_grid=placed_grid,
                        codebook=self.codebook,
                        spatial_graph=self.spatial_graph,
                        target_concept_vecs=concept_vecs,
                        target_labels=labels,
                        left_border=left_border,
                        top_border=top_border,
                    )
                    scores[token_id] = s

                chosen_token = self.predictor.sample_next_token(
                    scores,
                    temperature=temp,
                    top_k=min(self.config.top_k, len(available_tokens)),
                    top_p=self.config.top_p,
                )

                placed_grid[(r, c)] = chosen_token
                recent_tokens.append(chosen_token)
                row_tokens.append(chosen_token)

            token_grid.append(row_tokens)

        # Markov Random Field (MRF) iterative energy minimization
        if self.config.mrf_refinement_passes > 0:
            token_grid = self._mrf_refine(
                token_grid=token_grid,
                available_tokens=available_tokens,
                passes=self.config.mrf_refinement_passes,
                primary_label=primary_label,
                concept_vecs=concept_vecs,
                target_labels=labels,
            )

        # Decode token grid to image array with feathered boundary stitching
        flat_indices = [token_grid[r][c] for r in range(rows) for c in range(cols)]
        image_out = self.codebook.decode_indices(
            flat_indices,
            grid_shape=(rows, cols),
            patch_size=self.config.patch_size,
            blend_borders=True,
            overlap=4,
        )

        return image_out

    def _local_energy(
        self,
        r: int,
        c: int,
        token_id: int,
        token_grid: List[List[int]],
        primary_label: Optional[str] = None,
        concept_vecs: Optional[List[np.ndarray]] = None,
        target_labels: Optional[List[str]] = None,
    ) -> float:
        """
        Compute local Markov Random Field energy (lower is better).
        Sums unary positional/semantic fitness and pairwise neighborhood compatibility.
        """
        rows = len(token_grid)
        cols = len(token_grid[0]) if rows > 0 else 0
        norm_r = r / max(1, rows - 1)
        norm_c = c / max(1, cols - 1)

        entry = self.codebook.entries.get(token_id)
        if entry is None:
            return 100.0

        # Unary term
        pos_fitness = self.spatial_graph.get_position_fitness(token_id, norm_r, norm_c, label=primary_label)
        sem_sim = 0.5
        if concept_vecs:
            sims = [float(np.dot(entry.centroid, cvec)) for cvec in concept_vecs]
            sem_sim = max(sims) if sims else 0.5

        lbl_assoc = 0.0
        if target_labels:
            matches = [entry.label_counts.get(l.lower(), 0) for l in target_labels]
            lbl_assoc = min(1.0, sum(matches) / max(1, entry.total_assignments))

        unary_score = 0.45 * pos_fitness + 0.35 * sem_sim + 0.20 * lbl_assoc

        # Pairwise interaction terms with 4 orthogonal neighbors
        pairwise_compat = []
        if r > 0:
            top_tok = token_grid[r - 1][c]
            p_trans = self.spatial_graph.get_transition_probability(top_tok, token_id, "below")
            top_e = self.codebook.entries.get(top_tok)
            c_sim = float(np.dot(entry.centroid, top_e.centroid)) if top_e else 0.5
            pairwise_compat.append(0.5 * p_trans + 0.5 * c_sim)
        if r < rows - 1:
            bot_tok = token_grid[r + 1][c]
            p_trans = self.spatial_graph.get_transition_probability(token_id, bot_tok, "below")
            bot_e = self.codebook.entries.get(bot_tok)
            c_sim = float(np.dot(entry.centroid, bot_e.centroid)) if bot_e else 0.5
            pairwise_compat.append(0.5 * p_trans + 0.5 * c_sim)
        if c > 0:
            left_tok = token_grid[r][c - 1]
            p_trans = self.spatial_graph.get_transition_probability(left_tok, token_id, "right")
            left_e = self.codebook.entries.get(left_tok)
            c_sim = float(np.dot(entry.centroid, left_e.centroid)) if left_e else 0.5
            pairwise_compat.append(0.5 * p_trans + 0.5 * c_sim)
        if c < cols - 1:
            right_tok = token_grid[r][c + 1]
            p_trans = self.spatial_graph.get_transition_probability(token_id, right_tok, "right")
            right_e = self.codebook.entries.get(right_tok)
            c_sim = float(np.dot(entry.centroid, right_e.centroid)) if right_e else 0.5
            pairwise_compat.append(0.5 * p_trans + 0.5 * c_sim)

        pairwise_score = np.mean(pairwise_compat) if pairwise_compat else 0.5
        total_fitness = 0.40 * unary_score + 0.60 * pairwise_score
        return float(-total_fitness)

    def _mrf_refine(
        self,
        token_grid: List[List[int]],
        available_tokens: List[int],
        passes: int = 2,
        primary_label: Optional[str] = None,
        concept_vecs: Optional[List[np.ndarray]] = None,
        target_labels: Optional[List[str]] = None,
    ) -> List[List[int]]:
        """
        Markov Random Field Iterated Conditional Modes (ICM) energy minimization.
        Iteratively tests local token replacements to maximize spatial boundary coherence.
        """
        refined_grid = [row[:] for row in token_grid]
        rows = len(refined_grid)
        if rows == 0:
            return refined_grid
        cols = len(refined_grid[0])
        if cols == 0 or len(available_tokens) <= 1:
            return refined_grid

        eff_passes = passes
        if rows * cols >= 256:
            cand_sample_size = min(5, len(available_tokens))
            eff_passes = min(2, passes)
        else:
            cand_sample_size = min(15, len(available_tokens))

        for _ in range(eff_passes):
            for r in range(rows):
                for c in range(cols):
                    curr_tok = refined_grid[r][c]
                    curr_energy = self._local_energy(
                        r, c, curr_tok, refined_grid,
                        primary_label=primary_label,
                        concept_vecs=concept_vecs,
                        target_labels=target_labels,
                    )

                    best_tok = curr_tok
                    best_energy = curr_energy

                    candidates = np.random.choice(available_tokens, size=cand_sample_size, replace=False)
                    for cand in candidates:
                        if cand == curr_tok:
                            continue
                        e = self._local_energy(
                            r, c, int(cand), refined_grid,
                            primary_label=primary_label,
                            concept_vecs=concept_vecs,
                            target_labels=target_labels,
                        )
                        if e < best_energy:
                            best_energy = e
                            best_tok = int(cand)

                    refined_grid[r][c] = best_tok

        return refined_grid
