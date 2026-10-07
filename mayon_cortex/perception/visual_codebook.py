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
Visual Codebook — Graph-Native Visual Vocabulary & Hierarchical Vector Quantization
===================================================================================
Graph-native equivalent of VQ-VAE visual codebook with multi-scale hierarchies.

Features:
1. 3-Level Hierarchical Codebook (Coarse: 64, Medium: 256, Fine: 1024)
2. Incremental Learning via EMA centroid updates
3. Universal Saliency Extraction (Itti-Koch center-surround mechanism)
4. Concept Embeddings Cache & Gram-matrix style statistics
5. Boundary-feathered seamless index decoding
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from mayon_cortex.perception.visual_features import PatchInfo, VisualFeatureExtractor, rgb_to_gray


@dataclass
class CodebookEntry:
    """A discrete visual prototype representing a cluster of visual patterns."""
    entry_id: int
    centroid: np.ndarray                          # 256-dim prototype vector
    medoid_patch: Optional[np.ndarray] = None     # Highest-fidelity patch closest to centroid
    exemplar_patches: List[np.ndarray] = field(default_factory=list)  # Top real patches (RGB uint8)
    label_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_assignments: int = 0
    level: str = "medium"                         # coarse, medium, fine
    spatial_transitions: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {
            "right": defaultdict(float),
            "below": defaultdict(float),
            "left": defaultdict(float),
            "above": defaultdict(float),
        }
    )

    def get_representative_patch(self) -> np.ndarray:
        """Return the best exemplar patch or average patch."""
        if self.medoid_patch is not None:
            return self.medoid_patch
        if self.exemplar_patches:
            return self.exemplar_patches[0]
        return np.full((32, 32, 3), 128, dtype=np.uint8)

    def _ensure_borders(self):
        if not hasattr(self, "_left_border") or self._left_border is None:
            p = self.get_representative_patch().astype(np.float32)
            self._left_border = p[:, 0, :]
            self._right_border = p[:, -1, :]
            self._top_border = p[0, :, :]
            self._bottom_border = p[-1, :, :]

    @property
    def left_border(self) -> np.ndarray:
        self._ensure_borders()
        return self._left_border

    @property
    def right_border(self) -> np.ndarray:
        self._ensure_borders()
        return self._right_border

    @property
    def top_border(self) -> np.ndarray:
        self._ensure_borders()
        return self._top_border

    @property
    def bottom_border(self) -> np.ndarray:
        self._ensure_borders()
        return self._bottom_border


class VisualCodebook:
    """
    Learned hierarchical visual dictionary for encoding and decoding images
    without neural networks or backpropagation.
    """

    HIERARCHICAL_CONFIG: Dict[str, int] = {
        "coarse": 64,
        "medium": 256,
        "fine": 1024,
    }

    def __init__(
        self,
        codebook_size: int = 256,
        feature_dim: int = 256,
        patch_size: Tuple[int, int] = (32, 32),
        max_exemplars_per_entry: int = 5,
        ema_decay: float = 0.85,
    ):
        self.codebook_size = codebook_size
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.max_exemplars = max_exemplars_per_entry
        self.ema_decay = ema_decay
        self.extractor = VisualFeatureExtractor()

        # Primary (medium level) codebook state
        self.entries: Dict[int, CodebookEntry] = {}
        self.centroids: Optional[np.ndarray] = None  # (K, 256)
        self.concept_to_entries: Dict[str, Counter] = defaultdict(Counter)
        self.concept_topologies: Dict[str, List[List[int]]] = {}
        self.concept_images: Dict[str, np.ndarray] = {}
        self.concept_masks: Dict[str, np.ndarray] = {}

        # Hierarchical levels (coarse, medium, fine)
        self.hierarchical_centroids: Dict[str, Optional[np.ndarray]] = {
            "coarse": None,
            "medium": None,
            "fine": None,
        }
        self.hierarchical_entries: Dict[str, Dict[int, CodebookEntry]] = {
            "coarse": {},
            "medium": {},
            "fine": {},
        }

        # Concept embeddings & style caches
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        self.concept_styles: Dict[str, Dict[str, Any]] = {}
        self.concept_variances: Dict[str, float] = {}

        # Training patch cache
        self.history_samples: List[Tuple[np.ndarray, str, Optional[Dict[str, Any]]]] = []
        self._cached_patches: List[np.ndarray] = []
        self._cached_features: List[np.ndarray] = []
        self._cached_patch_meta: List[Tuple[int, str, int, int]] = []
        self.total_images_learned: int = 0

    def _kmeans(
        self,
        features: np.ndarray,
        k: int,
        max_iters: int = 25,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy k-means++ clustering on unit-normalized feature vectors.
        Returns: (centroids, cluster_assignments)
        """
        n_samples, dim = features.shape
        rng = np.random.RandomState(seed)

        if n_samples <= k:
            centroids = features.copy()
            if n_samples < k:
                pad_needed = k - n_samples
                pad_indices = rng.choice(n_samples, pad_needed, replace=True)
                centroids = np.vstack([centroids, features[pad_indices]])
            assignments = np.arange(n_samples)
            return centroids, assignments

        # K-means++ initialization
        centroids = np.zeros((k, dim), dtype=np.float32)
        first_idx = rng.randint(n_samples)
        centroids[0] = features[first_idx]

        closest_dist_sq = np.sum((features - centroids[0]) ** 2, axis=1)

        for c_idx in range(1, k):
            dist_sum = np.sum(closest_dist_sq)
            if dist_sum > 1e-9:
                probs = closest_dist_sq / dist_sum
                chosen_idx = rng.choice(n_samples, p=probs)
            else:
                chosen_idx = rng.randint(n_samples)
            centroids[c_idx] = features[chosen_idx]
            new_dist_sq = np.sum((features - centroids[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        assignments = np.zeros(n_samples, dtype=np.int32)

        # Iterative optimization (cosine distance on unit sphere)
        for _ in range(max_iters):
            sims = np.dot(features, centroids.T)  # (n_samples, k)
            new_assignments = np.argmax(sims, axis=1)

            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments

            for j in range(k):
                mask = (assignments == j)
                if np.any(mask):
                    mean_vec = np.mean(features[mask], axis=0)
                    norm = np.linalg.norm(mean_vec) + 1e-7
                    centroids[j] = mean_vec / norm
                else:
                    rand_pt = features[rng.randint(n_samples)]
                    centroids[j] = rand_pt

        return centroids, assignments

    @staticmethod
    def extract_universal_saliency(img_arr: np.ndarray) -> np.ndarray:
        """
        Itti-Koch inspired universal visual saliency extraction.
        Zero neural networks — operates purely on multi-scale center-surround
        intensity, color-opponent, and gradient energy channels.
        """
        h, w = img_arr.shape[:2]
        if img_arr.ndim == 2:
            img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1).astype(np.float32)
        else:
            img_rgb = img_arr[..., :3].astype(np.float32)

        r = img_rgb[..., 0]
        g = img_rgb[..., 1]
        b = img_rgb[..., 2]

        # 1. Perceptual channels
        intensity = (r + g + b) / 3.0
        rg = np.abs(r - g)
        by = np.abs(b - 0.5 * (r + g))

        try:
            import scipy.ndimage as ndi

            # 2. Multi-scale center-surround feature maps
            # Scales: fine (sigma 2, 4) vs coarse (sigma 8, 16)
            saliency_map = np.zeros((h, w), dtype=np.float32)

            for channel in [intensity, rg, by]:
                c_fine = ndi.gaussian_filter(channel, sigma=2.5)
                c_coarse = ndi.gaussian_filter(channel, sigma=9.0)
                diff = np.abs(c_fine - c_coarse)
                norm_d = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
                saliency_map += norm_d

            # 3. Gradient edge saliency
            gx = ndi.sobel(intensity, axis=1)
            gy = ndi.sobel(intensity, axis=0)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-6)
            saliency_map += 0.5 * grad_norm

            # 4. Central bias prior (natural photography center concentration)
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h * 0.5, w * 0.5
            dist_sq = ((y - center_y) / (h * 0.45)) ** 2 + ((x - center_x) / (w * 0.45)) ** 2
            center_prior = np.exp(-0.5 * dist_sq).astype(np.float32)
            saliency_map *= (0.4 + 0.6 * center_prior)

            # 5. Thresholding & morphological refinement
            saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
            binary = saliency_norm > np.percentile(saliency_norm, 35)

            # Fill interior holes and close gaps
            binary_filled = ndi.binary_fill_holes(binary)
            binary_closed = ndi.binary_closing(binary_filled, structure=np.ones((9, 9)))
            mask_float = binary_closed.astype(np.float32)

            # Feather edges with Gaussian blur
            feathered = ndi.gaussian_filter(mask_float, sigma=3.0)
            core = ndi.binary_erosion(binary_closed, structure=np.ones((5, 5)))
            feathered[core] = 1.0
            return np.clip(feathered, 0.0, 1.0).astype(np.float32)

        except Exception:
            # Fallback when scipy is unavailable
            gray = rgb_to_gray(img_rgb)
            g_mean = np.mean(gray)
            diff = np.abs(gray - g_mean)
            norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
            return np.clip(norm, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _extract_solid_subject_mask(cls, img_arr: np.ndarray, label: str = "cat") -> np.ndarray:
        """Universal subject saliency mask with backward-compatible signature."""
        return cls.extract_universal_saliency(img_arr)

    def learn(
        self,
        images_with_labels: List[Tuple[np.ndarray, str, Optional[Dict[str, Any]]]],
        patch_size: Optional[Tuple[int, int]] = None,
        k: Optional[int] = None,
        incremental: bool = True,
    ) -> int:
        """
        Train the visual codebook from a collection of labeled images.
        Supports both hierarchical clustering and incremental EMA updates.
        """
        if patch_size is not None:
            self.patch_size = patch_size
        k_clusters = k or self.codebook_size

        new_features: List[np.ndarray] = []
        new_patches: List[np.ndarray] = []
        new_meta: List[Tuple[int, str, int, int]] = []

        concept_patch_feats: Dict[str, List[np.ndarray]] = defaultdict(list)

        for item in images_with_labels:
            img_idx = len(self.history_samples)
            self.history_samples.append(item)
            img_arr = item[0]
            label = item[1].lower().strip()
            patches = self.extractor.decompose_image(img_arr, patch_size=self.patch_size)

            for p_info in patches:
                feat = self.extractor.extract_patch_features(p_info.patch)
                self._cached_patches.append(p_info.patch)
                self._cached_features.append(feat)
                self._cached_patch_meta.append((img_idx, label, p_info.row_idx, p_info.col_idx))

                new_features.append(feat)
                new_patches.append(p_info.patch)
                new_meta.append((img_idx, label, p_info.row_idx, p_info.col_idx))
                concept_patch_feats[label].append(feat)

        all_patches = self._cached_patches
        all_features = self._cached_features
        patch_meta = self._cached_patch_meta

        if not all_features:
            return 0

        feat_matrix = np.array(all_features, dtype=np.float32)

        # 1. Primary K-means Clustering (Incremental EMA or Full)
        actual_k = min(k_clusters, len(feat_matrix))

        if incremental and self.centroids is not None and len(self.centroids) == actual_k:
            # Incremental EMA update
            sims = np.dot(feat_matrix, self.centroids.T)
            assignments = np.argmax(sims, axis=1)
            for j in range(actual_k):
                mask = (assignments == j)
                if np.any(mask):
                    batch_mean = np.mean(feat_matrix[mask], axis=0)
                    norm_mean = batch_mean / (np.linalg.norm(batch_mean) + 1e-7)
                    updated = self.ema_decay * self.centroids[j] + (1.0 - self.ema_decay) * norm_mean
                    self.centroids[j] = updated / (np.linalg.norm(updated) + 1e-7)
        else:
            centroids, assignments = self._kmeans(feat_matrix, actual_k)
            self.centroids = centroids

        # 2. Hierarchical Multi-Scale Codebook Levels
        for lvl_name, lvl_size in self.HIERARCHICAL_CONFIG.items():
            eff_k = min(lvl_size, len(feat_matrix))
            lvl_centroids, lvl_assigns = self._kmeans(feat_matrix, eff_k, seed=101 + eff_k)
            self.hierarchical_centroids[lvl_name] = lvl_centroids
            self.hierarchical_entries[lvl_name] = {
                j: CodebookEntry(entry_id=j, centroid=lvl_centroids[j], level=lvl_name)
                for j in range(eff_k)
            }
            # Assign medoids and exemplar patches for hierarchy
            for p_idx, c_id in enumerate(lvl_assigns):
                h_entry = self.hierarchical_entries[lvl_name][c_id]
                h_entry.total_assignments += 1
                if len(h_entry.exemplar_patches) < self.max_exemplars:
                    h_entry.exemplar_patches.append(all_patches[p_idx])

        # 3. Populate primary entries and concept mapping
        self.entries = {j: CodebookEntry(entry_id=j, centroid=self.centroids[j], level="medium") for j in range(actual_k)}
        self.concept_to_entries = defaultdict(Counter)

        cluster_patch_indices = defaultdict(list)
        image_grid_maps: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(dict)

        for p_idx, entry_id in enumerate(assignments):
            img_idx, label, r_idx, c_idx = patch_meta[p_idx]
            entry = self.entries[entry_id]
            entry.total_assignments += 1
            entry.label_counts[label] += 1
            self.concept_to_entries[label][entry_id] += 1
            cluster_patch_indices[entry_id].append(p_idx)

            if len(entry.exemplar_patches) < self.max_exemplars:
                entry.exemplar_patches.append(all_patches[p_idx])

            image_grid_maps[img_idx][(r_idx, c_idx)] = entry_id

        # Determine highest-fidelity medoid patch for each cluster
        for entry_id, p_indices in cluster_patch_indices.items():
            if p_indices:
                cluster_feats = feat_matrix[p_indices]
                c_vec = self.centroids[entry_id]
                sims = np.dot(cluster_feats, c_vec)
                best_idx = p_indices[int(np.argmax(sims))]
                self.entries[entry_id].medoid_patch = all_patches[best_idx]

        # 4. Build spatial transitions between adjacent patches
        for img_idx, grid in image_grid_maps.items():
            for (r, c), entry_id in grid.items():
                curr_entry = self.entries[entry_id]
                if (r, c + 1) in grid:
                    right_id = grid[(r, c + 1)]
                    curr_entry.spatial_transitions["right"][right_id] += 1.0
                    self.entries[right_id].spatial_transitions["left"][entry_id] += 1.0
                if (r + 1, c) in grid:
                    below_id = grid[(r + 1, c)]
                    curr_entry.spatial_transitions["below"][below_id] += 1.0
                    self.entries[below_id].spatial_transitions["above"][entry_id] += 1.0

        for entry in self.entries.values():
            for direction, targets in entry.spatial_transitions.items():
                total = sum(targets.values())
                if total > 0:
                    for tid in list(targets.keys()):
                        targets[tid] /= total

        # 5. Store concept topologies, universal saliency masks, embeddings, and style statistics
        for img_idx, item in enumerate(self.history_samples):
            lbl = item[1].lower().strip()
            img_arr = item[0]
            self.concept_images[lbl] = img_arr
            grid_map = image_grid_maps[img_idx]
            if grid_map:
                max_r = max(k[0] for k in grid_map.keys()) + 1
                max_c = max(k[1] for k in grid_map.keys()) + 1
                grid_2d = [[grid_map.get((r, c), 0) for c in range(max_c)] for r in range(max_r)]
                self.concept_topologies[lbl] = grid_2d
                # Universal saliency mask for all concepts
                self.concept_masks[lbl] = self.extract_universal_saliency(img_arr)

        # 6. Compute concept embeddings and Gram-matrix style representations
        for lbl in set(m[1] for m in patch_meta):
            lbl_indices = [idx for idx, m in enumerate(patch_meta) if m[1] == lbl]
            if lbl_indices:
                lbl_feats = feat_matrix[lbl_indices]
                avg_vec = np.mean(lbl_feats, axis=0)
                norm_avg = avg_vec / (np.linalg.norm(avg_vec) + 1e-7)
                self.concept_embeddings[lbl] = norm_avg

                # Variance / consistency
                variances = float(np.mean(np.var(lbl_feats, axis=0)))
                self.concept_variances[lbl] = variances

                # Gram matrix of feature correlations for style transfer
                # Downsample feature dimension to 32 for fast style alignment
                step = max(1, self.feature_dim // 32)
                sub_feats = lbl_feats[:, ::step]
                gram = np.dot(sub_feats.T, sub_feats) / float(len(lbl_indices))
                self.concept_styles[lbl] = {
                    "gram": gram,
                    "mean_feat": norm_avg[::step],
                    "mean_color": np.mean(self.concept_images[lbl], axis=(0, 1)) if lbl in self.concept_images else np.array([128, 128, 128]),
                }

        self.total_images_learned = len(self.history_samples)
        return len(self.entries)

    def encode_patch(self, patch: np.ndarray, level: str = "medium") -> int:
        """Find the nearest codebook entry index for an image patch at specified level."""
        centroids = self.hierarchical_centroids.get(level) if level != "medium" else self.centroids
        if centroids is None or len(centroids) == 0:
            centroids = self.centroids
        if centroids is None or len(self.entries) == 0:
            return 0
        feat = self.extractor.extract_patch_features(patch)
        sims = np.dot(centroids, feat)
        return int(np.argmax(sims))

    def encode_image(
        self,
        image: np.ndarray,
        patch_size: Optional[Tuple[int, int]] = None,
        level: str = "medium",
    ) -> Tuple[List[int], Tuple[int, int]]:
        """Encode image into a grid of codebook token IDs at specified hierarchy level."""
        ps = patch_size or self.patch_size
        patches = self.extractor.decompose_image(image, patch_size=ps)
        if not patches:
            return [], (0, 0)

        max_row = max(p.row_idx for p in patches) + 1
        max_col = max(p.col_idx for p in patches) + 1

        indices = [self.encode_patch(p.patch, level=level) for p in patches]
        return indices, (max_row, max_col)

    def encode_hierarchical(
        self,
        image: np.ndarray,
        patch_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tuple[List[int], Tuple[int, int]]]:
        """Encode image across all 3 hierarchical levels (coarse, medium, fine)."""
        result = {}
        for lvl in ["coarse", "medium", "fine"]:
            result[lvl] = self.encode_image(image, patch_size=patch_size, level=lvl)
        return result

    def decode_indices(
        self,
        indices: List[int],
        grid_shape: Tuple[int, int],
        patch_size: Optional[Tuple[int, int]] = None,
        blend_borders: bool = True,
        overlap: int = 4,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Decode a sequence of visual token indices back into an RGB image.
        Seamless boundary-feathered stitching with exact output dimension matching.
        """
        ps = patch_size or self.patch_size
        ph, pw = ps
        rows, cols = grid_shape
        nominal_h = rows * ph
        nominal_w = cols * pw

        if rows <= 0 or cols <= 0 or len(indices) != rows * cols:
            out_h, out_w = target_size if target_size else (nominal_h, nominal_w)
            return np.zeros((max(1, out_h), max(1, out_w), 3), dtype=np.uint8)

        if not blend_borders or overlap <= 0:
            canvas = np.zeros((nominal_h, nominal_w, 3), dtype=np.uint8)
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    entry = self.entries.get(indices[idx])
                    patch = entry.get_representative_patch() if entry else np.full((ph, pw, 3), 128, dtype=np.uint8)
                    canvas[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = patch
                    idx += 1
            if target_size and (target_size[0] != nominal_h or target_size[1] != nominal_w):
                from PIL import Image
                return np.array(Image.fromarray(canvas).resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS))
            return canvas

        ov = min(overlap, min(ph, pw) // 4)
        stride_y = ph - ov
        stride_x = pw - ov
        out_h = (rows - 1) * stride_y + ph
        out_w = (cols - 1) * stride_x + pw

        canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)

        wy = np.ones(ph, dtype=np.float32)
        t = np.arange(ov, dtype=np.float32) / max(1, ov)
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)
        wy[:ov] = ramp
        wy[-ov:] = ramp[::-1]

        wx = np.ones(pw, dtype=np.float32)
        wx[:ov] = ramp
        wx[-ov:] = ramp[::-1]

        w2d = (wy[:, None] * wx[None, :])[:, :, None]
        w2d = np.maximum(w2d, 1e-4)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                entry = self.entries.get(indices[idx])
                if entry is not None:
                    patch = entry.get_representative_patch().astype(np.float32)
                else:
                    patch = np.full((ph, pw, 3), 128.0, dtype=np.float32)

                y_start = r * stride_y
                x_start = c * stride_x

                canvas[y_start:y_start + ph, x_start:x_start + pw] += patch * w2d
                weight_map[y_start:y_start + ph, x_start:x_start + pw] += w2d
                idx += 1

        reconstructed = canvas / (weight_map + 1e-6)
        stitched = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # Guarantee exact target size or nominal grid dimensions (rows * ph, cols * pw)
        final_h = target_size[0] if target_size else nominal_h
        final_w = target_size[1] if target_size else nominal_w

        if stitched.shape[0] != final_h or stitched.shape[1] != final_w:
            from PIL import Image
            stitched = np.array(
                Image.fromarray(stitched).resize((final_w, final_h), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )

        return stitched

    def integrate_with_graph(self, graph, node_prefix: str = "vcode::") -> int:
        """
        Publish visual codebook entries and ensure concept nodes are wired with visual pattern edges.
        """
        if graph is None:
            return 0

        added_edges = 0
        for entry_id, entry in self.entries.items():
            node_id = f"{node_prefix}{entry_id}"
            if node_id not in graph.nodes:
                try:
                    graph.add_node(
                        vector=entry.centroid,
                        source_text=f"VisualPrototype_{entry_id}",
                        level=1,
                        sector="vision",
                        node_type="visual_codebook_entry",
                        provenance="visual_codebook",
                        metadata={
                            "entry_id": entry_id,
                            "total_assignments": entry.total_assignments,
                            "top_labels": dict(entry.label_counts),
                        },
                    )
                except Exception:
                    pass

            if hasattr(graph, "add_edge"):
                for label, count in entry.label_counts.items():
                    concept_id = label.lower().strip()
                    if concept_id not in graph.nodes and hasattr(graph, "add_node"):
                        try:
                            graph.add_node(
                                vector=entry.centroid,
                                source_text=concept_id,
                                level=1,
                                sector="vision",
                                node_type="concept",
                                provenance="visual_learning",
                            )
                        except Exception:
                            pass

                    if concept_id in graph.nodes:
                        confidence = min(1.0, count / max(1, entry.total_assignments))
                        try:
                            graph.add_edge(
                                source_id=concept_id,
                                target_id=node_id,
                                relation_type="visual_pattern",
                                confidence=float(confidence),
                                sector="vision",
                                provenance="visual_codebook",
                            )
                            added_edges += 1
                        except Exception:
                            pass

        return added_edges
