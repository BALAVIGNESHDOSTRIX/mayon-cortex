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
Mayon-Cortex Language — Ingestion, Generation, Word Graphs & Prose
===================================================================
Universal data loading (PDF, CSV, JSON, SQL), language syntactic parsing,
hierarchical prose narrative engine, compositional word graphs,
sentence assembly, Broca decoders, and N-gram models.
"""

from mayon_cortex.language.universal_loader import UniversalDataLoader, IngestionReport
from mayon_cortex.language.language_ingestion import LanguageIngester, IngestionStats
from mayon_cortex.language.prose_engine import HierarchicalProseEngine, ProseConfig
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator, WordNode, GraphPoetEngine
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
    "SentenceAssembler",
    "SentenceTemplateEngine",
    "BrocaDecoder",
    "BrocaConfig",
    "BrocaGraphDecoder",
    "BrocaGraphConfig",
    "NGramModel",
    "SEED_CORPORA",
]
