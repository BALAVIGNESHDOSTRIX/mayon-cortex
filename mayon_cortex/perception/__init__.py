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
Mayon-Cortex Perception — Multimodal Input Understanding & Spatial Graphs
=========================================================================
Visual feature extraction, VQ-codebooks, spatial scene graphs, and knowledge
boundary arbitration for multimodal graph ingestion.
"""

from mayon_cortex.perception.vision import VisionCortex, ImageNode, VisualEncoder
from mayon_cortex.perception.scene_graph import SceneGraph, ARCSpatialSceneGraph, VisualObject, SpatialRelation
from mayon_cortex.perception.visual_features import VisualFeatureExtractor, PatchInfo
from mayon_cortex.perception.visual_codebook import VisualCodebook, CodebookEntry
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph, SpatialEdge
from mayon_cortex.perception.knowledge_boundary import (
    KnowledgeBoundary,
    ConceptKnowledge,
    GenerationCapabilityReport,
)

__all__ = [
    "VisionCortex",
    "ImageNode",
    "VisualEncoder",
    "SceneGraph",
    "ARCSpatialSceneGraph",
    "VisualObject",
    "SpatialRelation",
    "VisualFeatureExtractor",
    "PatchInfo",
    "VisualCodebook",
    "CodebookEntry",
    "VisualSpatialGraph",
    "SpatialEdge",
    "KnowledgeBoundary",
    "ConceptKnowledge",
    "GenerationCapabilityReport",
]
