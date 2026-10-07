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
Cortex-Graph v4 — The Complete Graph-Native Intelligence
=========================================================
Main orchestrator that integrates all 6 pillars:
- MayonGraph (brain tissue — knowledge + language storage)
- NDU (Node Decision Unit — neuron firing rule)
- ActivationWave (parallel wavefront cascade)
- PathRanker (proof chain ranking)
- WordGraph & WordGraphGenerator (Pillar 1: Compositional word generation)
- FractalAbstractionEngine (Pillar 2: Automatic concept hierarchy)
- EmotionEngine (Pillar 3: VAD emotional resonance)
- VisionCortex (Pillar 4: Image understanding & multimodal graph)
- AnalogyEngine (Pillar 5: Structural & vector analogical reasoning)
- Teacher & OnlineLearner (Pillar 6: Active curriculum assessment & Hebbian plasticity)
- CrossDomainResolver & DecisionPlanner (Multi-sector reasoning & action planning)
- UncertaintyGate & SelfCorrector (Epistemic humility & conflict detection)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mayon_cortex.core.graph import MayonGraph

from mayon_cortex.reasoning.abstraction import FractalAbstractionEngine
from mayon_cortex.dynamics.activation_wave import ActivatedPath, ActivationWave
from mayon_cortex.reasoning.analogy import AnalogyEngine, AnalogyResult
from mayon_cortex.core.config import CortexConfig
from mayon_cortex.reasoning.cross_domain import CrossDomainEvidence, CrossDomainResolver
from mayon_cortex.executive.decision_planner import DecisionChain, DecisionPlanner
from mayon_cortex.affect.emotion import VAD, EmotionEngine
from mayon_cortex.language.language_ingestion import IngestionStats, LanguageIngester
from mayon_cortex.language.universal_loader import UniversalDataLoader, IngestionReport
from mayon_cortex.core.ndu import NodeDecisionUnit
from mayon_cortex.learning.online_learner import OnlineLearner
from mayon_cortex.reasoning.path_ranker import PathRanker, ScoredPath
from mayon_cortex.dynamics.salience import SalienceWeighter
from mayon_cortex.executive.self_correction import SelfCorrector
from mayon_cortex.language.sentence_assembler import SentenceAssembler
from mayon_cortex.executive.uncertainty import UncertaintyGate
from mayon_cortex.perception.vision import ImageNode, VisionCortex
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator, WordNode, GraphPoetEngine
from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.language.prose_engine import HierarchicalProseEngine, ProseConfig
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
from mayon_cortex.reasoning.thinking import ThinkingEngine, ThoughtResult
from mayon_cortex.reasoning.problem_solver import ProblemSolver, SolvedProblem
from mayon_cortex.executive.cognitive_modes import CognitiveModeEngine, CognitiveMode
from mayon_cortex.dynamics.feedforward_cascade import FeedforwardCascade
from mayon_cortex.reasoning.knowledge_correlation import KnowledgeCorrelator, SynthesisResult
from mayon_cortex.memory.case_memory import CaseMemory
from mayon_cortex.reasoning.judgment_engine import JudgmentEngine, JudgmentVerdict
from mayon_cortex.language.broca_graph import BrocaGraphDecoder, BrocaGraphConfig
from mayon_cortex.language.ngram_model import NGramModel
from mayon_cortex.language.sentence_templates import SentenceTemplateEngine
from mayon_cortex.core.storage import GraphStorageManager, StorageConfig
from mayon_cortex.core.deduplication import GraphDeduplicator, DeduplicationReport
from mayon_cortex.perception.scene_graph import SceneGraph
from mayon_cortex.memory.hyperdimensional import HyperdimensionalMemory
from mayon_cortex.dynamics.energy_canvas import EnergyCanvas
from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator
from mayon_cortex.learning.forward_forward import ForwardForwardLearner
from mayon_cortex.learning.predictive_coder import PredictiveCoder
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
from mayon_cortex.reasoning.arc_solver import (
    ARCGrid,
    ARCObject,
    ARCPerception,
    ARCDSL,
    ARCProgramSynthesizer,
    ARCTask,
    ARCSolution,
)
from mayon_cortex.reasoning.cortical_arc_reasoner import CorticalARCReasoner, CorticalHypothesis
from mayon_cortex.perception.scene_graph import ARCSpatialSceneGraph
from mayon_cortex.perception.visual_features import VisualFeatureExtractor
from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph
from mayon_cortex.perception.knowledge_boundary import (
    KnowledgeBoundary,
    GenerationCapabilityReport,
    ConceptKnowledge,
)
from mayon_cortex.language.autoregressive_engine import GraphAutoRegressiveEngine, TextGenerationConfig
from mayon_cortex.reasoning.symbolic_algebra import UniversalEquationSolver, Polynomial, ExprParser, Expr
from mayon_cortex.reasoning.number_theory import NumberTheoryEngine
from mayon_cortex.reasoning.matrix_engine import MatrixEngine
from mayon_cortex.reasoning.calculus_engine import CalculusEngine
from mayon_cortex.reasoning.puzzle_solver import (
    SudokuSolver,
    CryptarithmSolver,
    NQueensSolver,
    MagicSquareEngine,
)





_embedder_cache = {}

def _get_embedder(dim: int):
    """Get or create a shared embedding model."""
    if dim not in _embedder_cache:
        from mayon_cortex.core.loader import EmbeddingModel
        _embedder_cache[dim] = EmbeddingModel(dim=dim)
    return _embedder_cache[dim]


@dataclass
class CortexResponse:
    """Response from a query."""
    text: str
    paths: List[ScoredPath] = field(default_factory=list)
    confidence: float = 0.0
    is_uncertain: bool = False
    uncertainty_reason: str = ""
    partial_knowledge: List[str] = field(default_factory=list)
    sectors_consulted: List[str] = field(default_factory=list)
    emotion: Optional[VAD] = None


@dataclass
class DecisionResponse:
    """Response from a complex decision query."""
    text: str
    decision_chain: Optional[DecisionChain] = None
    cross_domain_evidence: Optional[CrossDomainEvidence] = None
    conflicts: List = field(default_factory=list)
    confidence: float = 0.0
    is_uncertain: bool = False


class CortexGraph:
    """
    The complete graph-native intelligence.
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

        # Core substrate
        self.graph = MayonGraph(concept_dim=self.config.concept_dim)
        self.embedder = _get_embedder(self.config.concept_dim)

        # Affective & Language substrates (Pillars 1 & 3)
        self.emotion_engine = EmotionEngine()
        self.word_graph = WordGraph(self.config, emotion_engine=self.emotion_engine)
        self.word_generator = WordGraphGenerator(self.word_graph, emotion_engine=self.emotion_engine)
        self.poet_engine = GraphPoetEngine(self.word_graph, emotion_engine=self.emotion_engine)

        # Reasoning & Multimodal substrates (Pillars 2, 4, 5, 7)
        self.abstraction_engine = FractalAbstractionEngine(self.config)
        self.vision_cortex = VisionCortex(self.config)
        self.analogy_engine = AnalogyEngine(self.config)
        self.imagine_engine = ImagineEngine(self.config)
        self.prose_engine = HierarchicalProseEngine(
            word_graph=self.word_graph,
            concept_graph=self.graph,
            embedder=self.embedder,
            emotion_engine=self.emotion_engine,
            imagine_engine=self.imagine_engine,
        )

        # Neuron firing & wavefronts
        self.ndu = NodeDecisionUnit(self.config)
        self.wave = ActivationWave(self.config)
        self.ranker = PathRanker(self.config)

        # Higher intelligence & meta-cognition
        self.resolver = CrossDomainResolver(self.config)
        self.planner = DecisionPlanner(self.config)
        self.uncertainty = UncertaintyGate(self.config)
        self.corrector = SelfCorrector(self.config)
        self.salience = SalienceWeighter(self.config)

        # Generation & Ingestion
        self.assembler = SentenceAssembler(self.config, word_generator=self.word_generator)
        self.ingester = LanguageIngester(
            self.embedder,
            self.config,
            word_graph=self.word_graph,
            emotion_engine=self.emotion_engine,
        )
        self.universal_loader = UniversalDataLoader(self.ingester)

        # Memory & Plasticity (Pillar 6)
        self.working_memory = WorkingMemory(self.config)
        self.learner = OnlineLearner(self.config)

        # ── Phase 1 & 1.5: System 2 Thinking, Cognitive Modes & Problem Solving ──
        self.cognitive_modes = CognitiveModeEngine(default_mode=self.config.default_cognitive_mode)
        self.cascade = FeedforwardCascade()
        self.correlator = KnowledgeCorrelator(embedder=self.embedder)
        self.thinking_engine = ThinkingEngine(cortex=self)
        self.problem_solver = ProblemSolver(cortex=self)

        # ── Phase 2 & 3: Generation & Fluency ──
        self.broca_graph = BrocaGraphDecoder(BrocaGraphConfig(
            max_tokens=self.config.broca_graph_max_tokens,
            beam_width=self.config.broca_graph_beam_width,
        ))
        self.ngram_model = NGramModel(max_n=self.config.ngram_max_order, discount=self.config.ngram_discount)
        self.sentence_templates = SentenceTemplateEngine()

        # ── Phase 5: Scalability & Deduplication ──
        self.storage_manager = GraphStorageManager(StorageConfig(
            base_dir=self.config.storage_base_dir,
            snapshot_interval_sec=self.config.storage_snapshot_interval_sec,
        ))
        self.deduplicator = GraphDeduplicator(
            string_similarity_thresh=self.config.dedup_string_similarity_thresh,
            vector_similarity_thresh=self.config.dedup_vector_similarity_thresh,
        )

        # ── Phase 7: Case-Based Reasoning & Judgment ──
        self.case_memory = CaseMemory(capacity=self.config.case_memory_capacity)
        self.judgment_engine = JudgmentEngine(case_memory=self.case_memory, cortex=self)

        # ── Phase Ω: Third Paradigm Mathematics (HDC, MRF, Morphogenesis, Predictive) ──
        self.hdc_memory = HyperdimensionalMemory(dim=self.config.hdc_dimension)
        self.energy_canvas = EnergyCanvas(width=self.config.energy_canvas_width, height=self.config.energy_canvas_height)
        self.morphogenetic_generator = MorphogeneticGenerator()
        self.forward_forward = ForwardForwardLearner(goodness_threshold=self.config.ff_goodness_threshold)
        self.predictive_coder = PredictiveCoder(learning_rate=self.config.predictive_learning_rate)

        # ── Global Multi-Source Ingestion & Export Engine ──
        self.global_loader = GlobalDataLoader(
            graph=self.graph,
            embedder=self.embedder,
            deduplicator=self.deduplicator,
        )

        # ── Frontier Multi-Hop MCTS, SMT & Hyper-Relational Solvers ──
        self.semantic_role_extractor = SemanticRoleExtractor()
        self.math_solver = MathSolver()
        self.csp_solver = CSPSolver()
        self.resolution_prover = ResolutionProver()
        self.mcts_reasoner = GraphMCTSEngine(config=self.config)
        self.curiosity_resolver = CuriosityGapResolver(extractor=self.semantic_role_extractor)
        self.arc_synthesizer = ARCProgramSynthesizer()

        # ── Visual Feature Extraction, Codebook & Spatial Graph (Multimodal Ingestion) ──
        self.visual_feature_extractor = VisualFeatureExtractor()
        self.visual_codebook = VisualCodebook(codebook_size=256, feature_dim=256)
        self.visual_spatial_graph = VisualSpatialGraph(num_entries=256)
        self.knowledge_boundary = KnowledgeBoundary()

        # ── Graph Autoregressive Language Engine ──
        self.ar_text_engine = GraphAutoRegressiveEngine(word_graph=self.word_graph)

        # ── Advanced Symbolic Mathematics, Linear Algebra, Calculus & Puzzle Solvers ──
        self.equation_solver = UniversalEquationSolver()
        self.number_theory = NumberTheoryEngine()
        self.matrix_engine = MatrixEngine()
        self.calculus_engine = CalculusEngine()
        self.sudoku_solver = SudokuSolver()
        self.cryptarithm_solver = CryptarithmSolver()
        self.nqueens_solver = NQueensSolver()
        self.magic_square_engine = MagicSquareEngine()
        self.arc_reasoner = CorticalARCReasoner(cortex=self)



    def query(self, question: str, sector: str = "general") -> CortexResponse:
        """
        Factual & reasoning query through the cortical graph.
        """
        # 1. Embed query
        query_vec = self.embedder.encode_single(question)

        # Augment with conversation context
        if not self.working_memory.is_empty:
            query_vec = self.working_memory.augment_query(query_vec)

        # 2. Emotional context
        q_vad = self.emotion_engine.compute_concept_vad(question)

        # 3. Activate wavefront
        activation = self.wave.activate(query_vec, self.graph, self.ndu)

        # 4. Rank proof chains
        scored = self.ranker.rank_paths(activation.paths, query_vec, self.graph)

        # 5. Self-correct (remove contradictions)
        if scored:
            clean_paths, _ = self.corrector.correct(scored, self.graph)
            scored = clean_paths if clean_paths else scored

        # 6. Check uncertainty
        assessment = self.uncertainty.evaluate(scored, query_vec, self.graph)
        if assessment.is_uncertain:
            text = self.assembler.assemble_uncertainty(assessment)
            self.learner._query_count += 1
            self.working_memory.push(
                query_text=question, query_vec=query_vec,
                answer_text=text, confidence=assessment.confidence,
            )
            return CortexResponse(
                text=text, paths=scored,
                confidence=assessment.confidence,
                is_uncertain=True,
                uncertainty_reason=assessment.reason,
                partial_knowledge=assessment.partial_knowledge,
                emotion=q_vad,
            )

        # 7. Assemble fluent response
        text = self.assembler.assemble(
            scored,
            graph=self.graph,
            thought_vector=query_vec,
            context_vad=q_vad,
            sector=sector,
        )

        # 8. Online Hebbian learning
        self.learner.learn(
            scored_paths=scored,
            reward=1.0,
            graph=self.graph,
            ndu=self.ndu,
            ranker=self.ranker,
            query_vec=query_vec,
            answer_text=text,
            embedder=self.embedder,
            word_graph=self.word_graph,
            abstraction_engine=self.abstraction_engine,
        )

        # 9. Push to working memory
        confidence = scored[0].score if scored else 0.0
        node_ids = scored[0].node_ids if scored else []
        self.working_memory.push(
            query_text=question, query_vec=query_vec,
            answer_text=text, confidence=confidence,
            retrieved_node_ids=node_ids,
        )

        return CortexResponse(
            text=text,
            paths=scored,
            confidence=confidence,
            is_uncertain=False,
            emotion=q_vad,
        )

    def decide(self, question: str) -> DecisionResponse:
        """
        Complex multi-domain decision query with ordered action plan.
        """
        query_vec = self.embedder.encode_single(question)
        if not self.working_memory.is_empty:
            query_vec = self.working_memory.augment_query(query_vec)

        evidence = self.resolver.resolve(query_vec, self.graph, self.ndu, self.wave)
        chain = self.planner.plan(evidence, self.graph, query=question)

        if chain.num_steps == 0:
            assessment = self.uncertainty.evaluate([], query_vec, self.graph)
            text = self.assembler.assemble_uncertainty(assessment)
            return DecisionResponse(
                text=text, decision_chain=chain,
                cross_domain_evidence=evidence,
                is_uncertain=True,
            )

        text = self.assembler.assemble_decision(chain, self.graph)

        scored_proxy = []
        for path in evidence.resolved_paths[:5]:
            scored_proxy.append(ScoredPath(
                path=path, score=path.confidence,
                features=np.zeros(self.config.ranker_num_features),
                explanation="decision path",
            ))
        self.learner.learn(
            scored_paths=scored_proxy, reward=1.0,
            graph=self.graph, ndu=self.ndu, ranker=self.ranker,
            word_graph=self.word_graph, abstraction_engine=self.abstraction_engine,
        )

        self.working_memory.push(
            query_text=question, query_vec=query_vec,
            answer_text=text[:200], confidence=chain.total_confidence,
        )

        return DecisionResponse(
            text=text, decision_chain=chain,
            cross_domain_evidence=evidence,
            conflicts=evidence.conflicts,
            confidence=chain.total_confidence,
        )

    def ingest(self, text: str, sector: str = "general", provenance: str = "ingestion") -> IngestionStats:
        """Feed text into the brain (populates knowledge, phrases, and word graph)."""
        return self.universal_loader.ingest_text(text, self.graph, sector=sector, provenance=provenance)

    def ingest_file(
        self,
        file_path: Union[str, Path],
        sector: Optional[str] = None,
        provenance: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> IngestionReport:
        """Ingest any file (.txt, .md, .csv, .tsv, .json, .jsonl, .html, .xml)."""
        return self.universal_loader.ingest_file(
            file_path, self.graph, sector=sector, provenance=provenance, encoding=encoding
        )

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        sector: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> IngestionReport:
        """Recursively scan and ingest all supported files in a directory."""
        return self.universal_loader.ingest_directory(
            directory_path, self.graph, pattern=pattern, recursive=recursive, sector=sector, encoding=encoding
        )

    def ingest_records(
        self,
        records: Any,
        text_key: Optional[str] = None,
        sector_key: Optional[str] = None,
        default_sector: str = "general",
        provenance: str = "record_stream",
    ) -> IngestionReport:
        """Ingest arbitrary list of dicts, strings, or data records."""
        return self.universal_loader.ingest_records(
            records, self.graph, text_key=text_key, sector_key=sector_key,
            default_sector=default_sector, provenance=provenance,
        )

    def see(
        self,
        image_input: Union[str, bytes, np.ndarray],
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ImageNode:
        """Ingest and ground an image into the unified cortical graph."""
        return self.vision_cortex.see(image_input, caption=caption, tags=tags, graph=self.graph)

    def query_image(self, image_input: Union[str, bytes, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]:
        """Query knowledge related to an image."""
        return self.vision_cortex.query_image(image_input, graph=self.graph, top_k=top_k)

    def solve_analogy(self, a: str, b: str, c: str, sector_hint: Optional[str] = None) -> Optional[AnalogyResult]:
        """Solve an analogy: A is to B as C is to [D]."""
        return self.analogy_engine.solve(a, b, c, self.graph, sector_hint=sector_hint)

    def imagine_what_if(
        self,
        hypothetical_fact: str,
        query: str,
        sector: str = "general",
    ) -> SimulationResult:
        """Run a counterfactual simulation in an isolated mental sandbox."""
        return self.imagine_engine.what_if(
            base_graph=self.graph,
            hypothetical_fact=hypothetical_fact,
            query=query,
            embedder=self.embedder,
            ndu=self.ndu,
            wave=self.wave,
            ranker=self.ranker,
            sector=sector,
        )

    def blend_concepts(
        self,
        concept_a: str,
        concept_b: str,
        alpha: float = 0.5,
    ) -> Optional[EmergentConcept]:
        """Synthesize emergent concept from two ideas across domains."""
        return self.imagine_engine.blend(
            concept_a=concept_a,
            concept_b=concept_b,
            graph=self.graph,
            embedder=self.embedder,
            alpha=alpha,
        )

    def dream(self, cycles: int = 10) -> DreamStats:
        """Run an offline sleep/REM memory replay, hypothesis synthesis, and synaptic pruning cycle."""
        return self.imagine_engine.dream(
            graph=self.graph,
            working_memory=self.working_memory,
            abstraction_engine=self.abstraction_engine,
            cycles=cycles,
        )

    def explore_curiosity(self, top_k: int = 5) -> List[CuriosityGap]:
        """Identify knowledge voids and curiosity exploration targets."""
        return self.imagine_engine.identify_gaps(self.graph, top_k=top_k)

    def abstract(self):
        """Run an automatic fractal concept abstraction cycle."""
        return self.abstraction_engine.run_abstraction_cycle(self.graph)

    def teach(self, phase: Optional[int] = None):
        """Run curriculum learning."""
        from mayon_cortex.learning.teacher import Teacher
        teacher = Teacher(self)
        teacher.run_curriculum(phase=phase)

    def assess(self, live_eval: bool = True) -> Dict[str, Any]:
        """Assess the brain's knowledge and capabilities."""
        from mayon_cortex.learning.teacher import Teacher
        teacher = Teacher(self)
        return teacher.assess(live_eval=live_eval)

    def articulate_broca(self, question: str) -> str:
        """Broca decoder articulation."""
        from mayon_cortex.language.broca import BrocaConfig, BrocaDecoder
        if not hasattr(self, "_broca") or self._broca is None:
            self._broca = BrocaDecoder(BrocaConfig(graph_dim=self.config.concept_dim))

        query_vec = self.embedder.encode_single(question)
        activation = self.wave.activate(query_vec, self.graph, self.ndu)
        scored = self.ranker.rank_paths(activation.paths, query_vec, self.graph)

        if not scored:
            return "I don't have enough evidence in the graph to articulate an answer."

        node_vectors = []
        for p in scored[:3]:
            for nid in p.node_ids:
                if nid in self.graph.nodes:
                    node_vectors.append(self.graph.nodes[nid].vector)

        if not node_vectors:
            return self.assembler.assemble(scored, self.graph)

        graph_matrix = np.stack(node_vectors)
        return self._broca.generate_from_graph(
            graph_matrix, prompt_text=question, word_generator=self.word_generator
        )

    def compose_poem(
        self,
        theme: Union[str, List[str]],
        form: str = "couplet",
        mood: str = "heroic",
        lines: int = 4,
        meter: str = "tetrameter",
        temperature: float = 0.6,
    ) -> str:
        """
        100% Graph-Native poetic composition (Zero Transformers).
        Uses Conceptual Blending + Phonetic Rhyme Graph + Energy-based Boltzmann Traversal.
        """
        # 1. Resolve Theme vector (support multi-concept metaphorical blending)
        if isinstance(theme, list):
            if len(theme) >= 2:
                blended = self.imagine_engine.blender.blend(theme[0], theme[1], self.graph, self.embedder)
                theme_vec = blended.vector if blended else self.embedder.encode_single(" ".join(theme))
            else:
                theme_vec = self.embedder.encode_single(theme[0] if theme else "beauty")
        elif isinstance(theme, str) and (" and " in theme.lower() or "," in theme):
            parts = [p.strip() for p in re.split(r'\band\b|,', theme, flags=re.IGNORECASE) if p.strip()]
            if len(parts) >= 2 and hasattr(self, "imagine_engine"):
                blended = self.imagine_engine.blender.blend(parts[0], parts[1], self.graph, self.embedder)
                theme_vec = blended.vector if blended else self.embedder.encode_single(theme)
            else:
                theme_vec = self.embedder.encode_single(theme)
        else:
            theme_vec = self.embedder.encode_single(str(theme))

        # 2. Mood to VAD mapping
        q_vad = self.emotion_engine.compute_concept_vad(str(theme) + " " + mood)

        # 3. Generate through GraphPoetEngine
        return self.poet_engine.compose_poem(
            theme_vector=theme_vec,
            form=form,
            mood=mood,
            lines_count=lines,
            meter=meter,
            temperature=temperature,
            context_vad=q_vad,
        )

    def compose_prose(
        self,
        prompt: str,
        style: str = "epic",
        temperature: float = 0.8,
        max_words: int = 50,
        novelty: Optional[float] = None,
        coherence: Optional[float] = None,
        cadence: Optional[float] = None,
        emotion_intensity: Optional[float] = None,
    ) -> str:
        """
        100% Graph-Native creative prose generation with optional 1-to-10 parameter adaptation.
        """
        if novelty is not None or coherence is not None or cadence is not None:
            cfg = ProseConfig(
                novelty=novelty if novelty is not None else (temperature * 5.0),
                coherence=coherence if coherence is not None else 7.0,
                cadence=cadence if cadence is not None else 5.5,
                emotion_intensity=emotion_intensity if emotion_intensity is not None else 6.0,
                genre=style,
            )
            thought_vec = self.embedder.encode_single(prompt)
            para, _ = self.prose_engine.compose_paragraph(
                theme_vector=thought_vec,
                config=cfg,
                num_sentences=max(2, max_words // 15),
            )
            return para
        else:
            thought_vec = self.embedder.encode_single(prompt)
            q_vad = self.emotion_engine.compute_concept_vad(prompt + " " + style)
            return self.poet_engine.compose_prose(
                thought_vector=thought_vec,
                style=style,
                max_words=max_words,
                temperature=temperature,
                context_vad=q_vad,
            )

    def compose_story(
        self,
        prompt: str,
        acts: int = 3,
        novelty: float = 6.0,
        coherence: float = 7.5,
        cadence: float = 5.5,
        emotion_intensity: float = 6.0,
        genre: str = "general",
    ) -> Dict[str, Any]:
        """
        Generate a multi-act coherent narrative story with full 1-to-10 parameter control.
        100% Graph-Native (Zero GPUs / Zero Transformers).
        """
        cfg = ProseConfig(
            novelty=novelty,
            coherence=coherence,
            cadence=cadence,
            emotion_intensity=emotion_intensity,
            genre=genre,
        )
        return self.prose_engine.compose_story(prompt=prompt, acts_count=acts, config=cfg)

    def compose_article(
        self,
        topic: str,
        paragraphs: int = 3,
        novelty: float = 4.0,
        coherence: float = 8.5,
        cadence: float = 6.0,
        genre: str = "academic",
    ) -> Dict[str, Any]:
        """
        Generate a structured non-fiction / analytical article.
        """
        cfg = ProseConfig(
            novelty=novelty,
            coherence=coherence,
            cadence=cadence,
            genre=genre,
        )
        return self.prose_engine.compose_article(topic=topic, paragraphs_count=paragraphs, config=cfg)

    def ingest_corpus(
        self,
        corpus: Union[str, List[str], Path],
        domain: str = "general",
        verbose: bool = False,
    ) -> Dict[str, int]:
        """
        Ingest a raw text string, list of sentences, or file into WordGraph and ConceptGraph.
        """
        sentences = []
        if isinstance(corpus, (str, Path)) and (str(corpus).endswith(".txt") or str(corpus).endswith(".md") or Path(corpus).is_file()):
            with open(corpus, "r", encoding="utf-8") as f:
                raw_text = f.read()
            sentences = [s.strip() for s in re.split(r'[.!?\n]+', raw_text) if len(s.strip().split()) >= 3]
        elif isinstance(corpus, str):
            sentences = [s.strip() for s in re.split(r'[.!?\n]+', corpus) if len(s.strip().split()) >= 3]
        elif isinstance(corpus, list):
            sentences = corpus

        total_knowledge = 0
        total_phrases = 0
        for s in sentences:
            stats = self.ingest(s, sector=domain)
            total_knowledge += stats.knowledge_nodes_added
            total_phrases += stats.phrase_nodes_added

        return {
            "sentences_ingested": len(sentences),
            "knowledge_nodes_added": total_knowledge,
            "phrase_nodes_added": total_phrases,
            "total_word_nodes": len(self.word_graph.nodes),
        }

    def load_data(
        self,
        source: Union[str, Path, Dict[str, Any], Sequence[Any], KnowledgePackage, StandardKnowledgeItem, StandardDocumentItem],
        format: str = "auto",
        sector: Optional[str] = None,
        default_confidence: float = 1.0,
        provenance: Optional[str] = None,
        auto_dedup: bool = True,
        resolve_conflicts: bool = True,
    ) -> GlobalIngestionReport:
        """
        Universal data ingestion method. Ingests StandardKnowledgeItem, StandardDocumentItem,
        KnowledgePackage, JSON, JSONL, CSV, Markdown, text, or Python dicts.
        """
        return self.global_loader.load_source(
            source=source,
            format=format,
            sector=sector,
            default_confidence=default_confidence,
            provenance=provenance,
            auto_dedup=auto_dedup,
            resolve_conflicts=resolve_conflicts,
        )

    def load_sources(
        self,
        sources: Sequence[Any],
        sector: Optional[str] = None,
        auto_dedup: bool = True,
        resolve_conflicts: bool = True,
    ) -> GlobalIngestionReport:
        """Ingest multiple diverse knowledge sources in a single call."""
        return self.global_loader.load_sources(
            sources=sources,
            sector=sector,
            auto_dedup=auto_dedup,
            resolve_conflicts=resolve_conflicts,
        )

    def export_knowledge(
        self,
        target_path: Union[str, Path],
        format: str = "json",
        sector: Optional[str] = None,
        min_confidence: float = 0.0,
        domain_name: str = "general",
        description: str = "Exported Mayon-Cortex Brain Knowledge",
    ) -> ExportReport:
        """
        Export brain knowledge into standard JSON/JSONL/KnowledgePackage formats.
        """
        return BrainKnowledgeExporter.export(
            graph=self.graph,
            target_path=target_path,
            format=format,
            sector=sector,
            min_confidence=min_confidence,
            domain_name=domain_name,
            description=description,
        )

    def save(self, save_dir: str):
        """Save entire brain state to disk."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.graph.save(str(path / "graph"))
        self.ndu.save(str(path / "ndu"))
        self.ranker.save(str(path / "ranker"))

        # Save WordGraph
        wg_data = {
            "num_nodes": len(self.word_graph.nodes),
            "words": list(self.word_graph.nodes.keys()),
        }
        with open(path / "word_graph.json", "w") as f:
            json.dump(wg_data, f, indent=2)

        meta = {
            "version": "0.4.0",
            "concept_dim": self.config.concept_dim,
            "num_nodes": self.graph.num_nodes,
            "num_edges": self.graph.num_edges,
            "word_nodes": len(self.word_graph.nodes),
            "queries_processed": self.learner._query_count,
        }
        with open(path / "brain_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, save_dir: str, config: Optional[CortexConfig] = None) -> "CortexGraph":
        """Load brain from disk."""
        path = Path(save_dir)
        brain = cls(config=config)

        graph_dir = path / "graph"
        if graph_dir.exists():
            brain.graph = MayonGraph.load(str(graph_dir))

        ndu_dir = path / "ndu"
        if ndu_dir.exists():
            brain.ndu.load(str(ndu_dir))

        ranker_dir = path / "ranker"
        if ranker_dir.exists():
            brain.ranker.load(str(ranker_dir))

        return brain

    def stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        return {
            "graph_nodes": self.graph.num_nodes,
            "graph_edges": self.graph.num_edges,
            "word_nodes": len(self.word_graph.nodes),
            "abstract_concepts": len(self.abstraction_engine.abstract_concepts),
            "sectors": list(self.graph.sector_indices.keys()),
            "ndu_trained": self.ndu._trained,
            "ranker_trained": self.ranker._trained,
            "queries_processed": self.learner._query_count,
            "working_memory_size": self.working_memory.size,
            "case_precedents": self.case_memory.size,
            "graph_stats": self.graph.stats(),
        }

    # ── High-Level Third Paradigm Intelligence API ──

    def think(self, prompt: str, mode: str = "balanced", timeout_ms: int = 5000) -> ThoughtResult:
        """
        Execute full System 2 multi-step reasoning with subgoals, logic proof, and self-monitoring.
        """
        return self.thinking_engine.think(prompt=prompt, mode=mode, timeout_ms=timeout_ms)

    def solve(self, problem_text: str) -> Any:
        """
        Solve mathematical, logical, or multi-step constraint problems using
        SMT constraint propagation or graph problem solving.
        """
        # 1. Attempt exact mathematical / algebraic solution
        math_res = self.math_solver.solve(problem_text)
        if math_res.is_solved:
            return math_res

        # 2. Fall back to problem solver
        return self.problem_solver.solve(problem_text)

    def solve_math(self, problem_text: str) -> MathSolution:
        """Execute exact algebraic, arithmetic, or formula deduction."""
        return self.math_solver.solve(problem_text)

    def solve_csp(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        binary_constraints: List[Tuple[str, str, Any]],
        unary_constraints: Optional[List[Tuple[str, Any]]] = None,
    ) -> CSPResult:
        """Solve a discrete Constraint Satisfaction Problem via AC-3 and Forward Checking."""
        return self.csp_solver.solve(variables, domains, binary_constraints, unary_constraints)

    def prove_logic(self, kb_clauses: List[Clause], query_literal: Literal) -> ProofResult:
        """Execute First-Order Logic theorem proving via resolution refutation."""
        return self.resolution_prover.prove(kb_clauses, query_literal)

    def solve_arc(self, task: ARCTask, max_depth: int = 3) -> ARCSolution:
        """
        Solve an ARC-AGI visual/spatial grid transformation task via DSL program synthesis.
        """
        return self.arc_synthesizer.synthesize(task, max_depth=max_depth)

    def solve_arc_grid(
        self,
        train_pairs: List[Tuple[Any, Any]],
        test_input: Any,
        task_id: str = "custom_arc_task",
        category: str = "general",
    ) -> ARCSolution:
        """
        Convenience wrapper to solve an ARC task from raw list/array matrices.
        """
        arc_train = [(ARCGrid(inp), ARCGrid(out)) for inp, out in train_pairs]
        arc_test = [(ARCGrid(test_input), None)]
        task = ARCTask(
            task_id=task_id,
            category=category,
            train_pairs=arc_train,
            test_pairs=arc_test,
        )
        return self.solve_arc(task)

    def prove(
        self,
        start_concept: str,
        goal_concept: Optional[str] = None,
        context_vars: Optional[Dict[str, Any]] = None,
    ) -> MCTSProofResult:
        """
        Execute Graph-MCTS over MayonGraph tissue to produce a verified multi-hop proof.
        """
        query_vec = self.embedder.encode_single(f"{start_concept} {goal_concept or ''}")
        return self.mcts_reasoner.prove_multi_hop(
            start_concept=start_concept,
            goal_concept=goal_concept,
            graph=self.graph,
            query_vector=query_vec,
            context_vars=context_vars,
        )

    def ingest_hyper(self, text: str, sector: str = "general", provenance: str = "hyper_ingest") -> List[HyperEdge]:
        """
        Ingest text with Semantic Role Labeling, capturing condition clauses and math constraints.
        """
        hyper_edges = self.semantic_role_extractor.parse_sentence(text, sector=sector, provenance=provenance)
        entity_node_map = {}
        for he in hyper_edges:
            src_low = he.source.lower()
            tgt_low = he.target.lower()

            if src_low not in entity_node_map:
                vec = self.embedder.encode_single(he.source)
                node_s = self.graph.add_node(vector=vec, source_text=he.source, level=1, sector=he.sector, provenance=provenance)
                entity_node_map[src_low] = node_s.id
            if tgt_low not in entity_node_map:
                vec = self.embedder.encode_single(he.target)
                node_t = self.graph.add_node(vector=vec, source_text=he.target, level=1, sector=he.sector, provenance=provenance)
                entity_node_map[tgt_low] = node_t.id

            try:
                self.graph.add_edge(
                    source_id=entity_node_map[src_low],
                    target_id=entity_node_map[tgt_low],
                    relation_type=he.relation,
                    confidence=he.confidence,
                    sector=he.sector,
                    provenance=provenance,
                )
            except Exception:
                pass
        return hyper_edges

    def judge(self, situation: str, relevant_facts: Optional[List[str]] = None) -> JudgmentVerdict:
        """
        Evaluate a complex situation using precedent case memory, statutory rules, and principle weights.
        """
        return self.judgment_engine.judge(situation=situation, relevant_facts=relevant_facts)

    def correlate(self, concept_a: str, concept_b: str, max_hops: int = 4) -> SynthesisResult:
        """
        Discover non-obvious cross-domain bridge paths and synthesize emergent insights.
        """
        return self.correlator.correlate_and_synthesize(concept_a, concept_b, self.graph, max_hops=max_hops)

    def see_scene(self, scene: SceneGraph) -> List[str]:
        """
        Ingest a full SceneGraph into the knowledge graph with bounding boxes and spatial edges.
        """
        return self.vision_cortex.see_scene(scene, graph=self.graph)

    def generate_image(self, prompt: str, width: int = 64, height: int = 64, iterations: int = 100) -> np.ndarray:
        """
        Generate an image via Markov Random Field energy minimization on a 2D pixel grid.
        Pure CPU. Zero backprop.
        """
        self.energy_canvas.width = width
        self.energy_canvas.height = height
        return self.energy_canvas.generate(prompt=prompt, iterations=iterations)

    def grow_pattern(self, pattern_type: str = "stripes", size: int = 64) -> np.ndarray:
        """
        Grow an emergent biological texture/pattern using Turing reaction-diffusion.
        """
        return self.morphogenetic_generator.grow_pattern(pattern_type=pattern_type, size=size)

    def deduplicate(self) -> DeduplicationReport:
        """
        Find fuzzy duplicates, cluster aliases, and reconcile conflicting nodes in the graph.
        """
        node_labels = [n.source_text for n in self.graph.nodes.values() if hasattr(n, "source_text")]
        node_vecs = {n.source_text: n.vector for n in self.graph.nodes.values() if hasattr(n, "source_text") and hasattr(n, "vector")}
        return self.deduplicator.run_deduplication(node_labels, node_vecs)

    def learn_image(
        self,
        image_input: Union[str, np.ndarray],
        label: str,
        details: Optional[Dict[str, Any]] = None,
        patch_size: Tuple[int, int] = (32, 32),
    ) -> Dict[str, Any]:
        """
        Ingest and learn visual structures from a labeled image into the cognitive graph.
        1. Decomposes image into patches and extracts 128-dim invariant descriptors
        2. Quantizes into visual codebook prototypes and updates 2D spatial transitions
        3. Wires concept label to visual prototypes via Hebbian reinforced graph edges
        """
        # Load image array if path is provided
        if isinstance(image_input, str):
            try:
                from PIL import Image
                img_pil = Image.open(image_input).convert("RGB")
                img_arr = np.array(img_pil, dtype=np.uint8)
            except Exception:
                # Fallback to random pattern if loading fails
                img_arr = np.full((128, 128, 3), 128, dtype=np.uint8)
        else:
            img_arr = image_input

        # 1. Learn into visual codebook
        entries_count = self.visual_codebook.learn(
            images_with_labels=[(img_arr, label, details or {})],
            patch_size=patch_size,
        )

        # 2. Re-synchronize spatial graph across ALL learned concept topologies
        self.visual_spatial_graph.reset()
        for lbl, c_grid in self.visual_codebook.concept_topologies.items():
            self.visual_spatial_graph.learn_from_grid(c_grid, label=lbl)

        grid = self.visual_codebook.concept_topologies.get(label, [])
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        indices = [tok for r in grid for tok in r]

        # 3. Integrate with MayonGraph
        added_edges = self.visual_codebook.integrate_with_graph(self.graph)

        # 5. Attach image node in VisionCortex
        img_node = self.vision_cortex.see(
            img_arr,
            caption=f"{label}: {details.get('color', '')} {details.get('pose', '')}" if details else label,
            tags=[label] + (details.get("features", []) if details else []),
            graph=self.graph,
        )

        # 6. Register concept with KnowledgeBoundary
        if hasattr(self, "knowledge_boundary"):
            norm_lbl = label.lower().strip()
            emb = self.visual_codebook.concept_embeddings.get(norm_lbl, np.zeros(self.visual_codebook.feature_dim, dtype=np.float32))
            var = self.visual_codebook.concept_variances.get(norm_lbl, 0.05)
            self.knowledge_boundary.register_concept(
                label=norm_lbl,
                images_count=1,
                patches_count=len(indices),
                avg_vector=emb,
                variance=var,
                codebook_coverage=len(self.visual_codebook.concept_to_entries.get(norm_lbl, {})) / max(1, len(self.visual_codebook.entries)),
            )

        return {
            "label": label,
            "image_id": img_node.image_id,
            "codebook_entries": entries_count,
            "grid_shape": (rows, cols),
            "tokens_extracted": len(indices),
            "graph_edges_created": added_edges,
        }



    def generate_text_ar(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.75,
        top_p: float = 0.90,
    ) -> str:
        """
        Autoregressive next-token text generation via multi-hop graph activation.
        """
        cfg = TextGenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return self.ar_text_engine.generate(prompt=prompt, config=cfg)

    def solve_equation(self, equation_str: str, var: str = "x") -> Dict[str, Any]:
        """Solve an algebraic equation symbolically (e.g. '2*x^2 - 8 = 0')."""
        return self.equation_solver.solve(equation_str, var=var)

    def solve_derivative(self, expr_str: str, var: str = "x", order: int = 1) -> str:
        """Compute the n-th derivative of a mathematical expression."""
        d_expr = self.calculus_engine.differentiate(expr_str, var=var, order=order)
        return str(d_expr)

    def solve_integral(self, func_or_str: Any, a: float, b: float) -> float:
        """Compute numerical definite integral over [a, b]."""
        if isinstance(func_or_str, str):
            ast = ExprParser(func_or_str).parse()
            fn = lambda x_val: ast.eval({"x": x_val})
        else:
            fn = func_or_str
        return self.calculus_engine.definite_integral(fn, a, b)

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Solve linear system of equations AX = B with step-by-step row reduction."""
        return self.matrix_engine.solve_linear_system(A, b)

    def solve_sudoku(self, grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Solve a 9x9 Sudoku puzzle."""
        return self.sudoku_solver.solve(grid)

    def solve_cryptarithm(self, puzzle_str: str) -> Optional[Dict[str, int]]:
        """Solve an alphametic puzzle like 'SEND + MORE = MONEY'."""
        return self.cryptarithm_solver.solve(puzzle_str)

    def solve_nqueens(self, n: int = 8) -> List[List[int]]:
        """Solve N-Queens puzzle."""
        return self.nqueens_solver.solve(n)

    def solve_arc(self, task: Union[ARCTask, Dict[str, Any], str, Path]) -> ARCSolution:
        """
        Solve an ARC-AGI visual reasoning task through the Cortical Graph.
        Converts 2D visual demonstrations into spatial relation graphs,
        induces the abstract transformation rule via analogical graph mapping,
        and verifies 100% precision on training examples before predicting test output.
        """
        return self.arc_reasoner.solve(task)

    def render_procedural_texture(self, texture_type: str, width: int = 128, height: int = 128, **kwargs) -> np.ndarray:
        """Synthesize procedural textures (marble, wood, clouds, perlin, voronoi)."""
        if texture_type == "marble":
            return self.procedural_textures.marble_texture(width, height, **kwargs)
        elif texture_type == "wood":
            return self.procedural_textures.wood_texture(width, height, **kwargs)
        elif texture_type == "clouds":
            return self.procedural_textures.clouds_sky(width, height, **kwargs)
        elif texture_type == "fbm":
            fbm = self.procedural_textures.fbm_noise_2d(width, height, **kwargs)
            return (fbm * 255).astype(np.uint8)
        else:
            perlin = self.procedural_textures.perlin_noise_2d(width, height, **kwargs)
            return (perlin * 255).astype(np.uint8)

    def __repr__(self) -> str:
        return (
            f"MayonCortex(nodes={self.graph.num_nodes}, edges={self.graph.num_edges}, "
            f"words={len(self.word_graph.nodes)}, visual_vocab={len(self.visual_codebook.entries)}, "
            f"known_visuals={len(self.knowledge_boundary.concepts)}, queries={self.learner._query_count})"
        )



# Flagship Project Class Alias
MayonCortex = CortexGraph


