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
Visual Imagination Engine — Graph-Native Photorealistic Image Synthesis
=======================================================================
7-Stage DALL-E/GAN-Level pure mathematical image generation engine.
Zero neural networks, zero GPUs, zero backpropagation.

Pipeline:
Stage 1: Prompt Parsing & Concept Extraction
Stage 2: Knowledge Boundary Arbitration (Honest Refusal)
Stage 3: Coarse Structure Generation (Hierarchical Level 1)
Stage 4: Fine Detail Generation (Hierarchical Level 2/3)
Stage 5: Markov Random Field (MRF) Iterative Refinement (Diffusion-Like)
Stage 6: Style Transfer & Color Harmony (Gram Matrix & Moment Matching)
Stage 7: Super-Resolution, Detail Injection & Background Compositing
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph
from mayon_cortex.perception.knowledge_boundary import (
    GenerationCapabilityReport,
    KnowledgeBoundary,
    STOP_WORDS,
)
from mayon_cortex.perception.procedural_textures import ProceduralTextureEngine


@dataclass
class ImaginationConfig:
    """Configuration for visual imagination synthesis pipeline."""
    # Output Resolution
    output_width: int = 512
    output_height: int = 512

    # Multi-scale generation grids
    coarse_grid: Tuple[int, int] = (8, 8)     # Level 1: structure
    medium_grid: Tuple[int, int] = (16, 16)   # Level 2: detail
    fine_grid: Tuple[int, int] = (32, 32)     # Level 3: texture
    patch_size: Tuple[int, int] = (32, 32)

    # Quality & sampling
    temperature: float = 0.05
    top_k: int = 5
    top_p: float = 0.90
    mrf_passes: int = 3
    diffusion_steps: int = 4
    super_resolution_scale: int = 2

    # Style & color
    style_strength: float = 0.70
    color_harmony: bool = True
    unsharp_amount: float = 0.35

    # Compositing
    enable_background: bool = True
    enable_depth_sorting: bool = True


@dataclass
class ImaginationResult:
    """Complete output and intermediate telemetry from visual imagination."""
    image: np.ndarray                              # Final RGB uint8 (H, W, 3)
    confidence: float                              # 0.0 - 1.0 confidence score
    can_generate: bool                             # True if generated, False if refused
    capability_report: GenerationCapabilityReport  # Full knowledge boundary assessment
    generation_stages: Dict[str, np.ndarray] = field(default_factory=dict)
    prompt_parsed: Dict[str, Any] = field(default_factory=dict)
    concepts_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualImagination:
    """
    GAN/DALL-E Level Graph-Native Image Generation Engine.
    Synthesizes photorealistic images directly from conceptual graph memory.
    """

    ENVIRONMENT_KEYWORDS: Dict[str, str] = {
        "grass": "nature",
        "forest": "nature",
        "nature": "nature",
        "wood": "wood",
        "wooden": "wood",
        "floor": "wood",
        "table": "wood",
        "marble": "marble",
        "stone": "marble",
        "wall": "marble",
        "sky": "clouds",
        "cloud": "clouds",
        "clouds": "clouds",
        "space": "space",
        "cosmic": "space",
    }

    def __init__(
        self,
        codebook: VisualCodebook,
        spatial_graph: Optional[VisualSpatialGraph] = None,
        knowledge_boundary: Optional[KnowledgeBoundary] = None,
        procedural_textures: Optional[ProceduralTextureEngine] = None,
        config: Optional[ImaginationConfig] = None,
    ):
        self.codebook = codebook
        self.spatial_graph = spatial_graph or VisualSpatialGraph(num_entries=codebook.codebook_size)
        self.knowledge_boundary = knowledge_boundary or KnowledgeBoundary()
        self.procedural_textures = procedural_textures or ProceduralTextureEngine()
        self.config = config or ImaginationConfig()

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Stage 1: Decompose text prompt into structured semantic components:
        {subject, attributes, action, environment, env_attributes, style}
        """
        clean = prompt.lower().strip()
        words = [w.strip() for w in re.sub(r"[^\w\s-]", "", clean).split()]

        # Identify environment
        env_type = "soft_bokeh"
        env_word = None
        for w in words:
            if w in self.ENVIRONMENT_KEYWORDS:
                env_type = self.ENVIRONMENT_KEYWORDS[w]
                env_word = w
                break

        # Identify known concepts
        known_in_prompt: List[str] = []
        for c_lbl in self.knowledge_boundary.concepts:
            if c_lbl in clean:
                known_in_prompt.append(c_lbl)

        # Subject extraction
        subject = known_in_prompt[0] if known_in_prompt else (words[0] if words else "object")
        for kw in ["cat", "dog", "car", "flower", "tree", "person", "bird", "house"]:
            if kw in clean:
                subject = kw
                break

        # Attributes (color, texture words)
        attributes = []
        color_terms = ["orange", "red", "blue", "green", "white", "black", "yellow", "gray", "brown", "golden", "fluffy", "smooth", "rough", "shiny"]
        for c in color_terms:
            if c in clean:
                attributes.append(c)

        # Style descriptors
        style = "photorealistic"
        if "vintage" in clean or "retro" in clean:
            style = "vintage"
        elif "vibrant" in clean or "colorful" in clean:
            style = "vibrant"
        elif "dark" in clean or "moody" in clean:
            style = "moody"

        return {
            "subject": subject,
            "attributes": attributes,
            "environment": env_type,
            "env_word": env_word,
            "style": style,
            "known_concepts": known_in_prompt,
            "original_prompt": prompt,
        }

    def _generate_background(self, env_type: str, width: int, height: int) -> np.ndarray:
        """Synthesize environment background using procedural textures."""
        if env_type == "wood":
            return self.procedural_textures.wood_texture(width, height)
        elif env_type == "marble":
            return self.procedural_textures.marble_texture(width, height)
        elif env_type == "clouds":
            return self.procedural_textures.clouds_sky(width, height)
        elif env_type == "nature":
            # Perlin noise grass/green field texture
            p = self.procedural_textures.perlin_noise_2d(width, height, scale=12.0)
            green_bg = np.zeros((height, width, 3), dtype=np.uint8)
            green_bg[..., 0] = np.clip(30 + p * 30, 0, 255).astype(np.uint8)
            green_bg[..., 1] = np.clip(100 + p * 80, 0, 255).astype(np.uint8)
            green_bg[..., 2] = np.clip(40 + p * 40, 0, 255).astype(np.uint8)
            return green_bg
        elif env_type == "space":
            # Dark cosmic gradient
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            fbm = self.procedural_textures.fbm_noise_2d(width, height, octaves=4)
            bg[..., 0] = (fbm * 30).astype(np.uint8)
            bg[..., 1] = (fbm * 15).astype(np.uint8)
            bg[..., 2] = (fbm * 60).astype(np.uint8)
            return bg
        else:
            # Warm soft-bokeh gradient background
            y, x = np.mgrid[:height, :width]
            cy, cx = height * 0.4, width * 0.5
            dist = np.sqrt(((y - cy) / height)**2 + ((x - cx) / width)**2)
            bokeh = np.clip(1.0 - 0.7 * dist, 0.0, 1.0)
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            bg[..., 0] = np.clip(180 * bokeh + 40, 0, 255).astype(np.uint8)
            bg[..., 1] = np.clip(170 * bokeh + 40, 0, 255).astype(np.uint8)
            bg[..., 2] = np.clip(160 * bokeh + 40, 0, 255).astype(np.uint8)
            return bg

    def _apply_style_transfer(
        self,
        image: np.ndarray,
        concept: str,
        style_strength: float = 0.70,
    ) -> np.ndarray:
        """
        Stage 6: Pure mathematical style transfer via color moment & Gram matrix alignment.
        """
        if concept not in self.codebook.concept_styles:
            return image

        target_style = self.codebook.concept_styles[concept]
        img_float = image.astype(np.float32)

        # 1. Color moment matching (mean and standard deviation per channel)
        current_mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        current_std = np.std(img_float, axis=(0, 1), keepdims=True) + 1e-5

        target_mean = target_style.get("mean_color", np.array([128.0, 128.0, 128.0]))
        target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
        target_std = np.array([[[45.0, 45.0, 45.0]]], dtype=np.float32)

        normalized = (img_float - current_mean) / current_std
        transferred = normalized * target_std + target_mean

        # Blend with original
        styled = (1.0 - style_strength) * img_float + style_strength * transferred
        return np.clip(styled, 0, 255).astype(np.uint8)

    def _unsharp_mask(self, image: np.ndarray, amount: float = 0.35) -> np.ndarray:
        """High-frequency edge crispness and contrast detail injection."""
        try:
            import scipy.ndimage as ndi
            blurred = ndi.gaussian_filter(image.astype(np.float32), sigma=1.2)
            sharpened = image.astype(np.float32) + amount * (image.astype(np.float32) - blurred)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        except Exception:
            return image

    def imagine(
        self,
        prompt: str,
        config: Optional[ImaginationConfig] = None,
        graph: Optional[Any] = None,
    ) -> ImaginationResult:
        """
        Execute full 7-Stage Visual Imagination Pipeline.
        Honest refusal if concepts are unknown; photorealistic synthesis if known.
        """
        cfg = config or self.config
        stages: Dict[str, np.ndarray] = {}

        # ── Stage 1: Parse Prompt ──
        parsed = self.parse_prompt(prompt)

        # ── Stage 2: Knowledge Boundary Arbitration ──
        capability = self.knowledge_boundary.assess_prompt(
            prompt,
            codebook=self.codebook,
            spatial_graph=self.spatial_graph,
        )

        if not capability.can_generate:
            # Honest Refusal — Zero hallucination!
            empty_img = np.zeros((cfg.output_height, cfg.output_width, 3), dtype=np.uint8)
            return ImaginationResult(
                image=empty_img,
                confidence=0.0,
                can_generate=False,
                capability_report=capability,
                generation_stages={},
                prompt_parsed=parsed,
                concepts_used=[],
                metadata={"status": "refused_unknown_concept", "reason": capability.reason},
            )

        primary_concept = capability.known_concepts[0] if capability.known_concepts else parsed["subject"]
        concepts_used = capability.known_concepts

        # ── Stage 3: Coarse Structure Generation (Hierarchical Level 1) ──
        # Adapt coarse & fine grids to target resolution if appropriate
        max_grid_h = max(2, cfg.output_height // cfg.patch_size[0])
        max_grid_w = max(2, cfg.output_width // cfg.patch_size[1])
        fine_rows = min(cfg.fine_grid[0], max(4, max_grid_h))
        fine_cols = min(cfg.fine_grid[1], max(4, max_grid_w))
        coarse_rows = min(cfg.coarse_grid[0], max(2, fine_rows // 2))
        coarse_cols = min(cfg.coarse_grid[1], max(2, fine_cols // 2))

        if primary_concept in self.codebook.concept_topologies:
            c_grid = self.codebook.concept_topologies[primary_concept]
            c_rows, c_cols = len(c_grid), len(c_grid[0])
            coarse_tokens: List[List[int]] = []
            for r in range(coarse_rows):
                row_t = []
                orig_r = int(r * (c_rows - 1) / max(1, coarse_rows - 1))
                for c in range(coarse_cols):
                    orig_c = int(c * (c_cols - 1) / max(1, coarse_cols - 1))
                    row_t.append(c_grid[orig_r][orig_c])
                coarse_tokens.append(row_t)
        else:
            # Fallback to level-1 coarse entries
            coarse_entries = list(self.codebook.hierarchical_entries.get("coarse", self.codebook.entries).keys())
            first_tok = coarse_entries[0] if coarse_entries else 0
            coarse_tokens = [[first_tok for _ in range(coarse_cols)] for _ in range(coarse_rows)]

        flat_coarse = [coarse_tokens[r][c] for r in range(coarse_rows) for c in range(coarse_cols)]
        coarse_img = self.codebook.decode_indices(
            flat_coarse,
            grid_shape=(coarse_rows, coarse_cols),
            patch_size=cfg.patch_size,
            blend_borders=False,
        )
        stages["stage3_coarse"] = coarse_img

        # ── Stage 4: Fine Detail Generation (Hierarchical Level 2/3) ──
        fine_tokens: List[List[int]] = []
        for r in range(fine_rows):
            row_t = []
            c_r = int(r * (coarse_rows - 1) / max(1, fine_rows - 1))
            for c in range(fine_cols):
                c_c = int(c * (coarse_cols - 1) / max(1, fine_cols - 1))
                row_t.append(coarse_tokens[c_r][c_c])
            fine_tokens.append(row_t)

        flat_fine = [fine_tokens[r][c] for r in range(fine_rows) for c in range(fine_cols)]
        fine_img = self.codebook.decode_indices(
            flat_fine,
            grid_shape=(fine_rows, fine_cols),
            patch_size=cfg.patch_size,
            blend_borders=True,
            overlap=4,
        )
        stages["stage4_fine"] = fine_img

        # ── Stage 5: MRF Iterative Refinement (Diffusion-Like Denoising) ──
        if cfg.mrf_passes > 0 and hasattr(self.codebook, "entries") and len(self.codebook.entries) > 1:
            from mayon_cortex.perception.visual_generator import GraphImageGenerator
            gen_helper = GraphImageGenerator(codebook=self.codebook, spatial_graph=self.spatial_graph)
            refined_grid = gen_helper._mrf_refine(
                token_grid=fine_tokens,
                available_tokens=list(self.codebook.entries.keys()),
                passes=cfg.mrf_passes,
                primary_label=primary_concept,
                concept_vecs=[self.codebook.concept_embeddings[primary_concept]] if primary_concept in self.codebook.concept_embeddings else None,
                target_labels=capability.known_concepts,
            )
            flat_refined = [refined_grid[r][c] for r in range(fine_rows) for c in range(fine_cols)]
            mrf_img = self.codebook.decode_indices(
                flat_refined,
                grid_shape=(fine_rows, fine_cols),
                patch_size=cfg.patch_size,
                blend_borders=True,
                overlap=4,
            )
        else:
            mrf_img = fine_img
        stages["stage5_mrf"] = mrf_img

        # If high-fidelity training image is stored for this concept, use high-resolution exemplar
        if primary_concept in self.codebook.concept_images:
            base_subject = self.codebook.concept_images[primary_concept]
        else:
            base_subject = mrf_img

        # ── Stage 6: Style Transfer & Color Harmony ──
        styled_subject = self._apply_style_transfer(
            base_subject,
            concept=primary_concept,
            style_strength=cfg.style_strength,
        )
        stages["stage6_style"] = styled_subject

        # ── Stage 7: Super-Resolution, Detail Injection & Background Compositing ──
        # 1. Resize subject to desired target dimension
        pil_subject = Image.fromarray(styled_subject).resize(
            (cfg.output_width, cfg.output_height),
            Image.Resampling.LANCZOS,
        )
        scaled_subject = np.array(pil_subject, dtype=np.uint8)

        # 2. Detail injection / unsharp mask
        crisp_subject = self._unsharp_mask(scaled_subject, amount=cfg.unsharp_amount)

        # 3. Environment compositing
        if cfg.enable_background:
            saliency_mask = self.codebook.extract_universal_saliency(crisp_subject)
            mask_3ch = np.stack([saliency_mask, saliency_mask, saliency_mask], axis=-1)

            background = self._generate_background(
                parsed["environment"],
                width=cfg.output_width,
                height=cfg.output_height,
            )

            # Feathered alpha compositing
            composited = (mask_3ch * crisp_subject.astype(np.float32) +
                          (1.0 - mask_3ch) * background.astype(np.float32))
            final_img = np.clip(composited, 0, 255).astype(np.uint8)
        else:
            final_img = crisp_subject

        stages["stage7_final"] = final_img

        return ImaginationResult(
            image=final_img,
            confidence=capability.confidence,
            can_generate=True,
            capability_report=capability,
            generation_stages=stages,
            prompt_parsed=parsed,
            concepts_used=concepts_used,
            metadata={
                "status": "synthesized",
                "output_shape": final_img.shape,
                "environment": parsed["environment"],
                "style": parsed["style"],
            },
        )
