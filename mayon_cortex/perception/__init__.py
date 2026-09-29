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
Mayon-Cortex Perception — Vision, Codebooks, Multimodal Graphs & Rendering
=========================================================================
Visual feature extraction, VQ-codebooks, spatial graphs, autoregressive visual token
generation, knowledge boundary arbitration, visual imagination engine, procedural
textures, raymarching, and spatial scene graphs.
"""

from mayon_cortex.perception.vision import VisionCortex, ImageNode, VisualEncoder
from mayon_cortex.perception.scene_graph import SceneGraph, VisualObject, SpatialRelation
from mayon_cortex.perception.visual_features import VisualFeatureExtractor, PatchInfo
from mayon_cortex.perception.visual_codebook import VisualCodebook, CodebookEntry
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph, SpatialEdge
from mayon_cortex.perception.visual_generator import VisualTokenPredictor, GraphImageGenerator, GenerationConfig
from mayon_cortex.perception.procedural_textures import ProceduralTextureEngine
from mayon_cortex.perception.raymarcher import RaymarchRenderer, SDFObject, Material
from mayon_cortex.perception.knowledge_boundary import (
    KnowledgeBoundary,
    ConceptKnowledge,
    GenerationCapabilityReport,
)
from mayon_cortex.perception.visual_imagination import (
    VisualImagination,
    ImaginationConfig,
    ImaginationResult,
)

__all__ = [
    "VisionCortex",
    "ImageNode",
    "VisualEncoder",
    "SceneGraph",
    "VisualObject",
    "SpatialRelation",
    "VisualFeatureExtractor",
    "PatchInfo",
    "VisualCodebook",
    "CodebookEntry",
    "VisualSpatialGraph",
    "SpatialEdge",
    "VisualTokenPredictor",
    "GraphImageGenerator",
    "GenerationConfig",
    "ProceduralTextureEngine",
    "RaymarchRenderer",
    "SDFObject",
    "Material",
    "KnowledgeBoundary",
    "ConceptKnowledge",
    "GenerationCapabilityReport",
    "VisualImagination",
    "ImaginationConfig",
    "ImaginationResult",
]
