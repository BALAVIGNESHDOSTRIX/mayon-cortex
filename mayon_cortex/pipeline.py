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
Unified Pipeline — Single Entry Point for Data → Intelligence
================================================================
Phase 0 of the Brain-Like Intelligence upgrade.

One file to: load data → clean → ingest → generate.
Works with ANY domain, ANY format.

Usage:
    brain = CortexPipeline.from_files("docs/*.txt", "data/*.csv")
    brain = CortexPipeline.from_text(["fact 1", "fact 2"], sector="legal")
    brain = CortexPipeline.from_records([{"text": "...", "sector": "medical"}])
    
    brain.ask("What treats hypertension?")
    brain.generate("Write about space exploration", style="lyrical")
    brain.solve("If 3x + 5 = 20, what is x?")
    brain.judge("Employee posted confidential data on social media")
    brain.explore("connections between music and math")
    brain.process("any input — auto-routes to right mode")
"""

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.engine import CortexGraph, CortexResponse, DecisionResponse
from mayon_cortex.executive.cognitive_modes import CognitiveModeController, CognitiveMode, ModeConfig
from mayon_cortex.reasoning.thinking import ThinkingLoop, ThinkingResult
from mayon_cortex.dynamics.feedforward_cascade import FeedforwardCascade, CascadeResult

# Alias for pipeline modes
PipelineMode = CognitiveMode



@dataclass
class PipelineStats:
    """Statistics from pipeline ingestion."""
    total_files: int = 0
    total_sentences: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    sectors: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CortexPipeline:
    """
    One entry point: Data → Intelligence. Any domain. Any format.
    
    The universal JARVIS interface that automatically:
    1. Loads and ingests data from any source
    2. Detects cognitive mode from query intent
    3. Routes to the appropriate reasoning system
    4. Returns labeled, traceable output
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.brain = CortexGraph(self.config)

        # Phase 1: Thinking
        self.thinker = ThinkingLoop(max_steps=15, min_confidence=0.6)

        # Phase 1.5: Cognitive Modes
        self.mode_controller = CognitiveModeController()
        self.cascade = FeedforwardCascade(concept_dim=self.config.concept_dim)

        # Stats
        self._stats = PipelineStats()

    # ── Factory Methods ──

    @classmethod
    def from_bootstrap(cls, config: Optional[CortexConfig] = None) -> "CortexPipeline":
        """
        Create a pipeline pre-loaded with comprehensive multi-sector bootstrap knowledge.
        """
        from mayon_cortex.learning.bootstrap import bootstrap_all_sectors
        pipeline = cls(config)
        bootstrap_all_sectors(pipeline.brain)
        return pipeline

    @classmethod
    def from_files(
        cls,
        *patterns: str,
        sector_map: Optional[Dict[str, str]] = None,
        config: Optional[CortexConfig] = None,
    ) -> "CortexPipeline":

        """
        Create a pipeline from file glob patterns.
        
        Args:
            *patterns: File glob patterns (e.g., "docs/*.txt", "data/**/*.csv")
            sector_map: Optional dict mapping file patterns to sectors
                       (e.g., {"legal_*": "legal", "med_*": "medical"})
            config: Optional CortexConfig
            
        Example:
            brain = CortexPipeline.from_files(
                "legal_docs/*.txt",
                "medical_data/*.csv",
                sector_map={"legal_*": "legal", "medical_*": "medical"}
            )
        """
        pipeline = cls(config)

        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            for filepath in files:
                if not os.path.isfile(filepath):
                    continue

                # Detect sector from filename
                sector = "general"
                if sector_map:
                    basename = os.path.basename(filepath)
                    for file_pattern, sec in sector_map.items():
                        if file_pattern.replace("*", "") in basename:
                            sector = sec
                            break

                try:
                    pipeline._ingest_file(filepath, sector)
                    pipeline._stats.total_files += 1
                except Exception as e:
                    pipeline._stats.errors.append(f"{filepath}: {str(e)}")

        pipeline._stats.total_nodes = pipeline.brain.graph.num_nodes
        pipeline._stats.total_edges = len(pipeline.brain.graph.edges)
        return pipeline

    @classmethod
    def from_text(
        cls,
        texts: Union[str, List[str]],
        sector: str = "general",
        config: Optional[CortexConfig] = None,
    ) -> "CortexPipeline":
        """
        Create a pipeline from text strings.
        
        Args:
            texts: Single string or list of strings to ingest
            sector: Knowledge sector for all texts
            config: Optional CortexConfig
            
        Example:
            brain = CortexPipeline.from_text([
                "Aspirin treats headaches.",
                "Aspirin is an NSAID.",
                "NSAIDs can cause stomach ulcers.",
            ], sector="medical")
        """
        pipeline = cls(config)

        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            if text.strip():
                try:
                    pipeline.brain.ingest(text, sector=sector)
                    pipeline._stats.total_sentences += 1
                except Exception as e:
                    pipeline._stats.errors.append(str(e))

        if sector not in pipeline._stats.sectors:
            pipeline._stats.sectors.append(sector)

        pipeline._stats.total_nodes = pipeline.brain.graph.num_nodes
        pipeline._stats.total_edges = len(pipeline.brain.graph.edges)
        return pipeline

    @classmethod
    def from_records(
        cls,
        records: List[Dict[str, Any]],
        text_key: str = "text",
        sector_key: str = "sector",
        config: Optional[CortexConfig] = None,
    ) -> "CortexPipeline":
        """
        Create a pipeline from structured records (dicts).
        
        Args:
            records: List of dicts with text and optional sector
            text_key: Key in dict containing the text
            sector_key: Key in dict containing the sector
            config: Optional CortexConfig
            
        Example:
            brain = CortexPipeline.from_records([
                {"text": "Contract clause 5.1 states...", "sector": "legal"},
                {"text": "The bridge load capacity is...", "sector": "engineering"},
            ])
        """
        pipeline = cls(config)

        for record in records:
            text = record.get(text_key, "")
            sector = record.get(sector_key, "general")
            if text.strip():
                try:
                    pipeline.brain.ingest(text, sector=sector)
                    pipeline._stats.total_sentences += 1
                    if sector not in pipeline._stats.sectors:
                        pipeline._stats.sectors.append(sector)
                except Exception as e:
                    pipeline._stats.errors.append(str(e))

        pipeline._stats.total_nodes = pipeline.brain.graph.num_nodes
        pipeline._stats.total_edges = len(pipeline.brain.graph.edges)
        return pipeline

    # ── Dynamic Ingestion Methods ──

    def ingest_text(self, text_or_sentences: Union[str, List[str]], sector: str = "general") -> Dict[str, int]:
        """
        Dynamically ingest raw text or list of sentences into the brain.
        """
        return self.brain.ingest_corpus(text_or_sentences, domain=sector)

    def ingest_file(self, filepath: str, sector: str = "general") -> Any:
        """
        Ingest a file (txt, md, csv, json, jsonl, etc.) into the brain.
        """
        return self.brain.universal_loader.load_file(filepath, graph=self.brain.graph, sector=sector)

    # ── Intelligence APIs ──

    def ask(self, question: str, sector: str = "general") -> CortexResponse:

        """
        Ask a factual question. Zero hallucination guarantee.
        Uses System 1 (direct) or System 2 (thinking) based on complexity.
        """
        return self.brain.query(question, sector=sector)

    def think(self, question: str, mode: str = "balanced", timeout_ms: int = 5000) -> ThinkingResult:
        """
        Engage System 2 deliberative reasoning.
        Multi-step thinking with full reasoning trace.
        """
        return self.brain.think(prompt=question, mode=mode, timeout_ms=timeout_ms)


    def generate(
        self,
        prompt: str,
        style: str = "neutral",
        max_tokens: int = 100,
    ) -> str:
        """
        Generate creative text using CREATIVE cognitive mode.
        Output is labeled as generated, not claimed as fact.
        """
        mode_config = self.mode_controller.get_mode_config(CognitiveMode.CREATIVE)
        query_vec = self.brain.embedder.encode_single(prompt)

        # Use creative cascade for wide activation
        cascade_result = self.cascade.cascade(
            query_vec, self.brain.graph, mode_config, max_steps=10
        )

        # Use word graph generator for text output
        try:
            generated = self.brain.word_generator.generate(
                seed_text=prompt,
                concept_graph=self.brain.graph,
                max_tokens=max_tokens,
            )
            return generated
        except Exception:
            # Fallback to assembler
            return f"[Creative mode] Based on {len(cascade_result.fired_nodes)} activated concepts for: {prompt}"

    def compose(
        self,
        theme: str,
        form: str = "free_verse",
        lines: int = 8,
    ) -> str:
        """Compose poetry using the GraphPoetEngine."""
        try:
            return self.brain.poet_engine.compose(
                theme=theme,
                form=form,
                num_lines=lines,
            )
        except Exception:
            return f"[Poetry] Theme: {theme}, Form: {form} — insufficient vocabulary"

    def solve(self, problem: str) -> Any:
        """
        Solve a problem (math, constraint, logic).
        Uses algebraic SMT solving and graph problem solver.
        """
        return self.brain.solve(problem)

    def prove(self, start_concept: str, goal_concept: Optional[str] = None) -> Any:
        """
        Execute Graph-MCTS to produce an auditable multi-hop proof path.
        """
        return self.brain.prove(start_concept, goal_concept)

    def judge(self, situation: str, relevant_facts: Optional[List[str]] = None) -> Any:
        """
        Render a judgment based on precedent memory and legal/ethical principles.
        """
        return self.brain.judge(situation, relevant_facts=relevant_facts)


    def explore(self, question: str) -> str:
        """
        Explore connections and patterns in the knowledge graph.
        Uses EXPLORATORY cognitive mode.
        """
        from mayon_cortex.reasoning.knowledge_correlation import KnowledgeCorrelationEngine
        correlator = KnowledgeCorrelationEngine(self.config)

        correlations = correlator.discover_correlations(self.brain.graph, top_k=5)
        bridges = correlator.find_latent_bridges(self.brain.graph, top_k=5)

        parts = [f"Exploration: {question}\n"]

        if correlations:
            parts.append("Structural Correlations Found:")
            for c in correlations[:3]:
                parts.append(f"  • {c.creative_insight}")

        if bridges:
            parts.append("\nLatent Bridges (hidden connections):")
            for b in bridges[:3]:
                parts.append(f"  • {b.insight}")

        if not correlations and not bridges:
            parts.append("No cross-domain patterns found yet. Feed more multi-domain data.")

        return "\n".join(parts)

    def see(
        self,
        image_input: Union[str, bytes],
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """
        Process an image — encode, add to graph, and return understanding.
        """
        return self.brain.vision_cortex.see(
            image_input, caption=caption, tags=tags, graph=self.brain.graph
        )

    def process(self, input_text: str) -> Any:
        """
        Universal JARVIS entry point.
        
        Automatically detects cognitive mode and routes to the right system:
        - "What treats X?" → FACTUAL → ask()
        - "Write a poem about X" → CREATIVE → compose/generate()
        - "Solve: 3x + 5 = 20" → PROBLEM_SOLVING → solve()
        - "Should X be done?" → JUDGING → think()
        - "What connects X and Y?" → EXPLORATORY → explore()
        """
        mode = self.mode_controller.detect_mode(input_text)
        label = self.mode_controller.get_output_label(mode)

        if mode == CognitiveMode.FACTUAL:
            result = self.think(input_text)
            return result

        elif mode == CognitiveMode.CREATIVE:
            text = self.generate(input_text)
            return f"[{label}]\n{text}"

        elif mode == CognitiveMode.PROBLEM_SOLVING:
            result = self.solve(input_text)
            return result

        elif mode == CognitiveMode.JUDGING:
            result = self.think(input_text)
            return result

        elif mode == CognitiveMode.EXPLORATORY:
            return self.explore(input_text)

        else:
            return self.ask(input_text)

    # ── Data Ingestion ──

    def ingest(self, text: str, sector: str = "general") -> None:
        """Ingest a single text into the brain."""
        self.brain.ingest(text, sector=sector)
        self._stats.total_sentences += 1

    def ingest_file(self, filepath: str, sector: str = "general") -> None:
        """Ingest a file into the brain."""
        self._ingest_file(filepath, sector)

    def teach_case(
        self,
        situation: str,
        judgment: str,
        reasoning: str,
        sector: str = "general",
        factors: Optional[List[str]] = None,
    ) -> None:
        """
        Teach the brain a case example for judgment learning.
        Ingests the case as connected knowledge for precedent-based reasoning.
        """
        # Ingest situation + judgment + reasoning as linked knowledge
        case_text = f"{situation}. The judgment was: {judgment}. The reasoning: {reasoning}"
        self.brain.ingest(case_text, sector=sector)

        # Also ingest individual factors
        if factors:
            for factor in factors:
                self.brain.ingest(
                    f"Factor in this case: {factor}. Relevant to: {situation[:100]}",
                    sector=sector,
                )

    # ── Internal Helpers ──

    def _ingest_file(self, filepath: str, sector: str) -> None:
        """Read and ingest a single file."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            self.brain.ingest(text, sector=sector)

        elif ext in (".csv",):
            import csv
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = " ".join(str(v) for v in row.values() if v)
                    if text.strip():
                        self.brain.ingest(text, sector=sector)

        elif ext in (".json",):
            import json
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text", item.get("content", str(item)))
                    else:
                        text = str(item)
                    if text.strip():
                        self.brain.ingest(text, sector=sector)
            elif isinstance(data, dict):
                text = data.get("text", data.get("content", str(data)))
                self.brain.ingest(text, sector=sector)

        else:
            # Try reading as plain text
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                self.brain.ingest(text, sector=sector)

    # ── Properties ──

    @property
    def stats(self) -> PipelineStats:
        """Get ingestion statistics."""
        self._stats.total_nodes = self.brain.graph.num_nodes
        self._stats.total_edges = len(self.brain.graph.edges)
        return self._stats

    @property
    def graph(self) -> Any:
        """Direct access to the underlying MayonGraph."""
        return self.brain.graph

    def __repr__(self) -> str:
        s = self._stats
        return (
            f"CortexPipeline(nodes={self.brain.graph.num_nodes}, "
            f"edges={len(self.brain.graph.edges)}, "
            f"sectors={s.sectors})"
        )
