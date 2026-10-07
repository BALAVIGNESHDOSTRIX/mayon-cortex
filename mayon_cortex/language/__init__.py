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
Mayon-Cortex Language — Ingestion, Generation, Word Graphs & Autoregressive Engine
=================================================================================
Universal data loading, language parsing, graph-native autoregression,
hierarchical prose, compositional word graphs, sentence assembly, and Broca decoders.
"""

from mayon_cortex.language.universal_loader import UniversalDataLoader, IngestionReport
from mayon_cortex.language.language_ingestion import LanguageIngester, IngestionStats
from mayon_cortex.language.prose_engine import HierarchicalProseEngine, ProseConfig
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator, WordNode, GraphPoetEngine
from mayon_cortex.language.autoregressive_engine import GraphAutoRegressiveEngine, TextGenerationConfig
from mayon_cortex.language.sentence_assembler import SentenceAssembler
from mayon_cortex.language.sentence_templates import SentenceTemplateEngine
from mayon_cortex.language.broca import BrocaDecoder, BrocaConfig
from mayon_cortex.language.broca_graph import BrocaGraphDecoder, BrocaGraphConfig
from mayon_cortex.language.ngram_model import NGramModel
from mayon_cortex.language.schema import (
    StandardKnowledgeItem,
    StandardDocumentItem,
    KnowledgePackage,
)
from mayon_cortex.language.global_loader import (
    GlobalDataLoader,
    GlobalIngestionReport,
)
from mayon_cortex.language.exporter import (
    BrainKnowledgeExporter,
    ExportReport,
)

__all__ = [
    "StandardKnowledgeItem",
    "StandardDocumentItem",
    "KnowledgePackage",
    "GlobalDataLoader",
    "GlobalIngestionReport",
    "BrainKnowledgeExporter",
    "ExportReport",
    "UniversalDataLoader",
    "IngestionReport",
    "LanguageIngester",
    "IngestionStats",
    "HierarchicalProseEngine",
    "ProseConfig",
    "WordGraph",
    "WordGraphGenerator",
    "WordNode",
    "GraphPoetEngine",
    "GraphAutoRegressiveEngine",
    "TextGenerationConfig",
    "SentenceAssembler",
    "SentenceTemplateEngine",
    "BrocaDecoder",
    "BrocaConfig",
    "BrocaGraphDecoder",
    "BrocaGraphConfig",
    "NGramModel",
]
