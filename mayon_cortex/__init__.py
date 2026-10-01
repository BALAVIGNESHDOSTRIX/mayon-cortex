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
Mayon-Cortex: The Unified Cognitive Architecture
=================================================
A biologically-inspired, neuro-symbolic brain-like growing intelligence
operating on pure CPU with zero GPU, zero backprop, and 100% deterministic proofs.

10 Modular Subsystems:
  • core/        - Graph tissue, NDU neuron firing rules, configuration & storage
  • dynamics/    - Spreading activation waves, salience decay & energy physics
  • memory/      - Working memory, case memory & 10k-dim HDC vector memory
  • reasoning/   - PathRanker, formal logic, System 2 thinking & problem solving
  • executive/   - Prefrontal strategy, uncertainty gating & self-correction
  • learning/    - Hebbian synaptic plasticity, forward-forward & predictive coding
  • affect/      - VAD emotional resonance & homeostasis
  • language/    - Universal ingestion, prose engine, word graphs & Broca articulation
  • perception/  - Vision cortex & multimodal spatial scene graphs
  • imagination/ - Mental sandboxes, concept blending & dream consolidation
"""

__version__ = "1.0.0"

# Main Orchestrators
from mayon_cortex.engine import MayonCortex, CortexGraph, CortexResponse, DecisionResponse
from mayon_cortex.pipeline import CortexPipeline, PipelineStats, PipelineMode

# 1. Core
from mayon_cortex.core import (
    MayonGraph,
    ConceptNode,
    RelationEdge,
    GraphLevel,
    Subgraph,
    LightweightGraphView,
    Node,
    Edge,
    EmbeddingModel,
    PythonDocsLoader,
    CodebaseLoader,
    MultiSectorKnowledgeLoader,
    CortexConfig,
    MayonConfig,
    DecisionPriority,
    RELATION_PRIORITY_MAP,
    OPPOSING_RELATIONS,
    CONFLICTING_RELATIONS,
    NodeDecisionUnit,
    GraphStorageManager,
    StorageConfig,
    GraphDeduplicator,
    DeduplicationReport,
    BaseVectorIndex,
    VectorIndexFactory,
    USearchVectorIndex,
    FaissVectorIndex,
    NumpyVectorIndex,
)

# 2. Dynamics
from mayon_cortex.dynamics.activation_wave import ActivationWave, ActivatedPath
from mayon_cortex.dynamics.salience import SalienceWeighter
from mayon_cortex.dynamics.feedforward_cascade import FeedforwardCascade, CascadeLayer, CascadeActivation
from mayon_cortex.dynamics.energy_canvas import EnergyCanvas, MRFConfig
from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator, MorphogeneticParams

# 3. Memory
from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.memory.case_memory import CaseMemory, PrecedentCase, CaseMatch
from mayon_cortex.memory.hyperdimensional import HyperVector, HyperdimensionalMemory
from mayon_cortex.memory.thought_scratchpad import ThoughtScratchpad, ScratchpadStep

# 4. Reasoning
from mayon_cortex.reasoning.path_ranker import PathRanker, ScoredPath
from mayon_cortex.reasoning.logic import LogicEngine, ProofStep, FormalProof
from mayon_cortex.reasoning.thinking import ThinkingEngine, ThoughtResult
from mayon_cortex.reasoning.problem_solver import ProblemSolver, ProblemDecomposition, SolutionStep, SolvedProblem
from mayon_cortex.reasoning.judgment_engine import JudgmentEngine, JudgmentVerdict
from mayon_cortex.reasoning.analogy import AnalogyEngine, AnalogyResult
from mayon_cortex.reasoning.cross_domain import CrossDomainResolver, CrossDomainEvidence
from mayon_cortex.reasoning.abstraction import FractalAbstractionEngine, AbstractConcept
from mayon_cortex.reasoning.knowledge_correlation import KnowledgeCorrelator, CorrelationPath, SynthesisResult
from mayon_cortex.reasoning.cortical_arc_reasoner import CorticalARCReasoner, CorticalHypothesis
from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCObject,
    ARCPerception,
    ARCDSL,
    ARCProgramSynthesizer,
    IntelligentSynthesizer,
    ARCTask,
    ARCSolution,
)

# 5. Executive
from mayon_cortex.executive.executive import (
    ExecutiveController,
    ExecutivePlan,
    Complexity,
    ThinkingStrategy,
)
from mayon_cortex.executive.metacognition import MetacognitionMonitor, EpistemicState
from mayon_cortex.executive.uncertainty import UncertaintyGate
from mayon_cortex.executive.self_correction import SelfCorrector
from mayon_cortex.executive.decision_planner import DecisionPlanner, DecisionChain
from mayon_cortex.executive.cognitive_modes import CognitiveModeEngine, CognitiveMode, CognitiveParameters

# 6. Learning
from mayon_cortex.learning.online_learner import OnlineLearner
from mayon_cortex.learning.forward_forward import ForwardForwardLearner, GoodnessProfile
from mayon_cortex.learning.predictive_coder import PredictiveCoder, PredictionError
from mayon_cortex.learning.teacher import Teacher
from mayon_cortex.learning.bootstrap import KnowledgeBootstrapper

# 7. Affect
from mayon_cortex.affect.emotion import EmotionEngine, VAD

# 8. Language
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
from mayon_cortex.language.universal_loader import UniversalDataLoader, IngestionReport
from mayon_cortex.language.language_ingestion import LanguageIngester, IngestionStats
from mayon_cortex.language.prose_engine import HierarchicalProseEngine, ProseConfig
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator, WordNode, GraphPoetEngine
from mayon_cortex.language.sentence_assembler import SentenceAssembler
from mayon_cortex.language.sentence_templates import SentenceTemplateEngine
from mayon_cortex.language.broca import BrocaDecoder, BrocaConfig
from mayon_cortex.language.broca_graph import BrocaGraphDecoder, BrocaGraphConfig
from mayon_cortex.language.ngram_model import NGramModel

# 9. Perception
from mayon_cortex.perception.vision import VisionCortex, ImageNode
from mayon_cortex.perception.scene_graph import SceneGraph, ARCSpatialSceneGraph, VisualObject, SpatialRelation

# 10. Imagination
from mayon_cortex.imagination.imagination import (
    ImagineEngine,
    MentalSandbox,
    ConceptBlender,
    DreamConsolidator,
    CuriosityEngine,
    EmergentConcept,
    SimulationResult,
    DreamStats,
    CuriosityGap,
)

# 11. Frontier Multi-Hop & Formal SMT Reasoning
from mayon_cortex.language.hyper_parser import (
    HyperEdge,
    Modality,
    MathConstraint,
    SemanticRoleExtractor,
)
from mayon_cortex.reasoning.smt_solver import (
    MathSolver,
    MathSolution,
    CSPSolver,
    CSPResult,
    ResolutionProver,
    Literal,
    Clause,
    ProofResult,
)
from mayon_cortex.reasoning.mcts_reasoner import (
    GraphMCTSEngine,
    MCTSNode,
    MCTSProofResult,
    MCTSProofStep,
)
from mayon_cortex.executive.curiosity_resolver import (
    CuriosityGapResolver,
    ResolutionAttempt,
)

__all__ = [
    # Main Engine
    "MayonCortex",
    "CortexGraph",
    "CortexResponse",
    "DecisionResponse",
    "CortexPipeline",
    "PipelineStats",
    "PipelineMode",

    # Core
    "MayonGraph",
    "Node",
    "Edge",
    "Sector",
    "EmbeddingModel",
    "DataLoader",
    "EntityExtractor",
    "CortexConfig",
    "MayonConfig",
    "DecisionPriority",
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

    # Dynamics
    "ActivationWave",
    "ActivatedPath",
    "SalienceWeighter",
    "FeedforwardCascade",
    "CascadeLayer",
    "CascadeActivation",
    "EnergyCanvas",
    "MRFConfig",
    "MorphogeneticGenerator",
    "MorphogeneticParams",

    # Memory
    "WorkingMemory",
    "CaseMemory",
    "PrecedentCase",
    "CaseMatch",
    "HyperVector",
    "HyperdimensionalMemory",
    "ThoughtScratchpad",
    "ScratchpadStep",

    # Reasoning
    "PathRanker",
    "ScoredPath",
    "LogicEngine",
    "ProofStep",
    "FormalProof",
    "ThinkingEngine",
    "ThoughtResult",
    "ProblemSolver",
    "ProblemDecomposition",
    "SolutionStep",
    "SolvedProblem",
    "JudgmentEngine",
    "JudgmentVerdict",
    "AnalogyEngine",
    "AnalogyResult",
    "CrossDomainResolver",
    "CrossDomainEvidence",
    "FractalAbstractionEngine",
    "AbstractConcept",
    "KnowledgeCorrelator",
    "CorrelationPath",
    "SynthesisResult",
    "CorticalARCReasoner",
    "CorticalHypothesis",
    "ARCGrid",
    "ARCObject",
    "ARCPerception",
    "ARCDSL",
    "ARCProgramSynthesizer",
    "IntelligentSynthesizer",
    "ARCTask",
    "ARCSolution",

    # Executive
    "ExecutiveController",
    "ExecutivePlan",
    "Complexity",
    "ThinkingStrategy",
    "MetacognitionMonitor",
    "EpistemicState",
    "UncertaintyGate",
    "SelfCorrector",
    "DecisionPlanner",
    "DecisionChain",
    "CognitiveModeEngine",
    "CognitiveMode",
    "CognitiveParameters",

    # Learning
    "OnlineLearner",
    "ForwardForwardLearner",
    "GoodnessProfile",
    "PredictiveCoder",
    "PredictionError",
    "Teacher",
    "KnowledgeBootstrapper",

    # Affect
    "EmotionEngine",
    "VAD",

    # Language
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

    # Perception
    "VisionCortex",
    "ImageNode",
    "SceneGraph",
    "ARCSpatialSceneGraph",
    "VisualObject",
    "SpatialRelation",

    # Imagination
    "ImagineEngine",
    "MentalSandbox",
    "ConceptBlender",
    "DreamConsolidator",
    "CuriosityEngine",
    "EmergentConcept",
    "SimulationResult",
    "DreamStats",
    "CuriosityGap",

    # Frontier Multi-Hop & Formal Reasoning
    "HyperEdge",
    "Modality",
    "MathConstraint",
    "SemanticRoleExtractor",
    "MathSolver",
    "MathSolution",
    "CSPSolver",
    "CSPResult",
    "ResolutionProver",
    "Literal",
    "Clause",
    "ProofResult",
    "GraphMCTSEngine",
    "MCTSNode",
    "MCTSProofResult",
    "MCTSProofStep",
    "CuriosityGapResolver",
    "ResolutionAttempt",
]
