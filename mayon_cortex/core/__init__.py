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
Mayon-Cortex Core — Graph Tissue & Infrastructure
==================================================
Houses the fundamental graph data structures, node decision units,
configurations, loaders, storage managers, and deduplication.
"""

from mayon_cortex.core.graph import (
    MayonGraph,
    ConceptNode,
    RelationEdge,
    GraphLevel,
    Subgraph,
    LightweightGraphView,
)
from mayon_cortex.core.loader import (
    EmbeddingModel,
    PythonDocsLoader,
    CodebaseLoader,
    MultiSectorKnowledgeLoader,
)
from mayon_cortex.core.config import (
    CortexConfig,
    DecisionPriority,
    RELATION_PRIORITY_MAP,
    OPPOSING_RELATIONS,
    CONFLICTING_RELATIONS,
)
from mayon_cortex.core.mayon_config import MayonConfig
from mayon_cortex.core.ndu import NodeDecisionUnit
from mayon_cortex.core.storage import GraphStorageManager, StorageConfig
from mayon_cortex.core.deduplication import GraphDeduplicator, DeduplicationReport
from mayon_cortex.core.vector_index import (
    BaseVectorIndex,
    VectorIndexFactory,
    USearchVectorIndex,
    FaissVectorIndex,
    NumpyVectorIndex,
)

# Compatibility Aliases
Node = ConceptNode
Edge = RelationEdge

__all__ = [
    "MayonGraph",
    "ConceptNode",
    "RelationEdge",
    "GraphLevel",
    "Subgraph",
    "LightweightGraphView",
    "Node",
    "Edge",
    "EmbeddingModel",
    "PythonDocsLoader",
    "CodebaseLoader",
    "MultiSectorKnowledgeLoader",
    "CortexConfig",
    "MayonConfig",
    "DecisionPriority",
    "RELATION_PRIORITY_MAP",
    "OPPOSING_RELATIONS",
    "CONFLICTING_RELATIONS",
    "NodeDecisionUnit",
    "GraphStorageManager",
    "StorageConfig",
    "GraphDeduplicator",
    "DeduplicationReport",
    "BaseVectorIndex",
    "VectorIndexFactory",
    "USearchVectorIndex",
    "FaissVectorIndex",
    "NumpyVectorIndex",
]
