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
Sentence Assembler — Paths → Fluent Grounded Text
=================================================
Converts graph proof chains + language phrase bridges + word graph into fluent text.

Strategy:
1. Primary Grounding: Phrase-bridge chaining from active proof paths (expressed-as + followed-by)
2. Compositional Generation: WordGraphGenerator for flexible bridging when phrases are sparse
3. Fallback: Structural relation connectors
4. Structured Decision Chain formatting (Pillar 6)
"""

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.affect.emotion import VAD
from mayon_cortex.language.word_graph import WordGraph, WordGraphGenerator


# Natural language connectors for chaining facts
RELATION_CONNECTORS = {
    "treats":          "which treats",
    "causes":          "which causes",
    "prevents":        "which prevents",
    "is-a":            "which is a type of",
    "part-of":         "which is part of",
    "contraindicates": "however, it is contraindicated with",
    "interacts-with":  "which interacts with",
    "inhibits":        "which inhibits",
    "induces":         "which induces",
    "regulates":       "which is regulated by",
    "metabolized-by":  "which is metabolized by",
    "calls":           "which calls",
    "inherits":        "which inherits from",
    "uses":            "which uses",
    "implies":         "which implies",
    "proves":          "which proves",
    "contradicts":     "which contradicts",
    "monitored-by":    "monitored by",
    "has-severity":    "with severity level",
    "diagnosed-by":    "diagnosed using",
    "risk-factor-for": "which is a risk factor for",
    "alternative-to":  "as an alternative to",
    "costs":           "which costs",
    "covered-by":      "which is covered by",
    "requires-check":  "which requires checking",
    "orbits":          "which orbits around",
    "defeats":         "which defeats",
    "protects":        "which protects",
    "produces":        "which produces",
    "lives-in":        "which lives in",
}

# Decision step formatting templates
PRIORITY_LABELS: Dict[int, str] = {
    100: "SAFETY",
    95:  "CONTRAINDICATION",
    90:  "LEGAL",
    85:  "ETHICAL",
    80:  "CLINICAL",
    75:  "SCIENTIFIC",
    70:  "FEASIBILITY",
    60:  "FINANCIAL",
    50:  "PREFERENCE",
    40:  "EFFICIENCY",
    30:  "MONITORING",
    20:  "DOCUMENTATION",
}


class SentenceAssembler:
    """
    Converts graph proof chains into fluent natural language text.
    """

    def __init__(
        self,
        config: Optional[CortexConfig] = None,
        word_generator: Optional[WordGraphGenerator] = None,
    ):
        self.config = config or CortexConfig()
        self.word_generator = word_generator

    def assemble(
        self,
        scored_paths: List,
        graph=None,
        thought_vector: Optional[np.ndarray] = None,
        context_vad: Optional[VAD] = None,
        sector: str = "general",
    ) -> str:
        """
        Assemble grounded, fluent text from ranked proof chains.
        """
        if not scored_paths:
            return ""

        # 1. Primary: Assemble from grounded phrase bridges and active proof edges
        output_sentences = []
        seen_sentences = set()
        primary_node_ids = set()

        if scored_paths:
            top_p = scored_paths[0].path if hasattr(scored_paths[0], "path") else scored_paths[0]
            primary_node_ids = set(getattr(top_p, "node_ids", []))

        for idx, scored_path in enumerate(scored_paths[:3]):
            path = scored_path.path if hasattr(scored_path, "path") else scored_path
            edges = path.edges if hasattr(path, "edges") else []
            node_ids = set(getattr(path, "node_ids", []))

            if not edges:
                continue

            # Only include follow-up paths if they share context with the top proof path
            if idx > 0 and primary_node_ids and not (node_ids & primary_node_ids):
                continue

            sentence = self._path_to_sentence(edges, graph)

            if sentence and sentence not in seen_sentences:
                output_sentences.append(sentence)
                seen_sentences.add(sentence)

        if output_sentences:
            return " ".join(output_sentences)

        # 2. Secondary: Compositional WordGraph generator
        if thought_vector is None:
            thought_vector = self._extract_thought_vector(scored_paths, graph)

        if self.word_generator is not None and getattr(self.word_generator.word_graph, "nodes", None):
            try:
                text = self.word_generator.generate(
                    thought_vector=thought_vector,
                    proof_paths=scored_paths,
                    context_vad=context_vad,
                    max_tokens=getattr(self.config, "word_max_tokens", 25),
                    beam_width=getattr(self.config, "word_beam_width", 6),
                    target_sector=sector,
                )
                if text and len(text.split()) >= 3:
                    return text
            except Exception:
                pass

        # 3. Fallback: list active facts
        return self._fallback_output(scored_paths, graph)

    def _extract_thought_vector(self, scored_paths: List, graph) -> np.ndarray:
        """Extract average thought vector from active proof nodes."""
        dim = getattr(self.config, "concept_dim", 384)
        vectors = []

        for sp in scored_paths:
            path = sp.path if hasattr(sp, "path") else sp
            node_ids = path.node_ids if hasattr(path, "node_ids") else []
            for nid in node_ids:
                if graph and hasattr(graph, "nodes") and nid in graph.nodes:
                    v = graph.nodes[nid].vector
                    if v is not None:
                        vectors.append(v)
                elif hasattr(nid, "vector") and nid.vector is not None:
                    vectors.append(nid.vector)

        if vectors:
            tv = np.mean(vectors, axis=0)
            return tv / (np.linalg.norm(tv) + 1e-8)
        return np.zeros(dim, dtype=np.float32)

    def _path_to_sentence(
        self, edges: List[Tuple[str, str, str]], graph
    ) -> str:
        """Convert a single path's edges into a fluent sentence."""
        if not edges or graph is None:
            return ""

        # 1. Check if phrase nodes along the path point to a grounded ingested sentence
        for src_id, relation, tgt_id in edges:
            for nid in (src_id, tgt_id):
                if nid in graph.nodes and hasattr(graph, "_out_edges"):
                    for eid in graph._out_edges.get(nid, []):
                        if eid in graph.edges:
                            edge = graph.edges[eid]
                            if edge.relation_type == "expressed-as" and edge.target_id in graph.nodes:
                                pn = graph.nodes[edge.target_id]
                                meta = getattr(pn, "metadata", {})
                                if meta and "sentence" in meta and len(meta["sentence"]) > 10:
                                    s = meta["sentence"].strip()
                                    if not s.endswith(('.', '!', '?')):
                                        s += '.'
                                    return s

        # 2. Assemble parts cleanly with connector
        parts = []
        for i, (src_id, relation, tgt_id) in enumerate(edges):
            src_text = self._get_node_text(src_id, graph)
            tgt_text = self._get_node_text(tgt_id, graph)

            if not src_text or not tgt_text:
                continue

            phrase_bridge = self._find_phrase_bridge(src_id, tgt_id, graph)
            if phrase_bridge and phrase_bridge.lower() not in src_text.lower():
                parts.append(f"{src_text} {phrase_bridge} {tgt_text}")
            else:
                connector = RELATION_CONNECTORS.get(relation, f"which {relation}")
                parts.append(f"{src_text} {connector} {tgt_text}")

        sentence = ", ".join(parts) if parts else ""
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'

        # Clean duplicate spaces and punctuation
        sentence = re.sub(r'\s+([,\.\?!;:])', r'\1', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    def _find_phrase_bridge(
        self, src_id: str, tgt_id: str, graph
    ) -> Optional[str]:
        """Find language phrase nodes that bridge two knowledge nodes within the same sentence."""
        if graph is None or not hasattr(graph, "_out_edges"):
            return None

        # Gather source phrases with sentence_id
        src_phrases = []
        for eid in graph._out_edges.get(src_id, []):
            if eid in graph.edges:
                edge = graph.edges[eid]
                if edge.relation_type == "expressed-as" and edge.target_id in graph.nodes:
                    pn = graph.nodes[edge.target_id]
                    if getattr(pn, "node_type", "") == "phrase":
                        src_phrases.append(pn)

        if not src_phrases:
            return None

        # Check target sentence IDs
        tgt_sentence_ids = set()
        for eid in graph._out_edges.get(tgt_id, []):
            if eid in graph.edges:
                edge = graph.edges[eid]
                if edge.relation_type == "expressed-as" and edge.target_id in graph.nodes:
                    tpn = graph.nodes[edge.target_id]
                    sid = getattr(tpn, "metadata", {}).get("sentence_id")
                    if sid:
                        tgt_sentence_ids.add(sid)

        best_chain = []
        for start_phrase in src_phrases:
            start_sid = getattr(start_phrase, "metadata", {}).get("sentence_id")
            # If target has sentence IDs, prefer phrases sharing the same sentence_id
            if tgt_sentence_ids and start_sid not in tgt_sentence_ids:
                continue

            chain = [start_phrase.source_text]
            current = start_phrase.id
            visited = {current}

            for _ in range(5):
                next_phrase = None
                for eid in graph._out_edges.get(current, []):
                    if eid in graph.edges:
                        edge = graph.edges[eid]
                        if edge.relation_type == "followed-by" and edge.target_id not in visited:
                            if edge.target_id in graph.nodes:
                                next_node = graph.nodes[edge.target_id]
                                next_sid = getattr(next_node, "metadata", {}).get("sentence_id")
                                # Ensure we stay strictly within the same sentence
                                if start_sid and next_sid and start_sid != next_sid:
                                    continue
                                if getattr(next_node, "node_type", "") == "phrase":
                                    next_phrase = next_node
                                    break

                if next_phrase:
                    chain.append(next_phrase.source_text)
                    visited.add(next_phrase.id)
                    current = next_phrase.id
                else:
                    break

            if len(chain) > len(best_chain):
                best_chain = chain

        return " ".join(best_chain) if best_chain else None

    def _get_node_text(self, node_id: str, graph) -> str:
        """Get clean display text for a node."""
        if graph is None or not hasattr(graph, "nodes") or node_id not in graph.nodes:
            return str(node_id)
        node = graph.nodes[node_id]
        text = getattr(node, "source_text", str(node_id))
        if ": " in text and len(text.split(": ")[0]) < 50:
            text = text.split(": ")[0]
        return text.strip()

    def _fallback_output(self, scored_paths: List, graph) -> str:
        """Fallback when no clean sentence can be assembled."""
        facts = []
        for sp in scored_paths[:3]:
            path = sp.path if hasattr(sp, "path") else sp
            node_ids = path.node_ids if hasattr(path, "node_ids") else []
            for nid in node_ids:
                if graph and hasattr(graph, "nodes") and nid in graph.nodes:
                    text = getattr(graph.nodes[nid], "source_text", "")
                    if text and text not in facts and len(text) > 3:
                        facts.append(text)
        return ". ".join(facts[:5]) + "." if facts else "No active proof paths."

    def assemble_decision(
        self, decision_chain, graph=None
    ) -> str:
        """Format a decision chain as a human-readable ordered action plan."""
        if not hasattr(decision_chain, 'steps') or not decision_chain.steps:
            return "No decision steps generated."

        parts = []
        parts.append("DECISION PLAN:")
        parts.append("=" * 50)

        for step in decision_chain.steps:
            priority_label = PRIORITY_LABELS.get(step.priority, f"[P{step.priority}]")
            bar_len = min(step.priority // 10, 10)
            bar = "#" * bar_len + "-" * (10 - bar_len)

            parts.append(f"\nStep {step.rank} [{priority_label} {bar} Priority: {step.priority}]")
            parts.append(f"  ACTION: {step.action}")
            parts.append(f"  REASON: {step.reason}")
            parts.append(f"  Confidence: {step.confidence:.2f}")
            parts.append(f"  Sector: {step.sector}")

            if step.depends_on:
                parts.append(f"  Depends on: Steps {', '.join(str(d) for d in step.depends_on)}")

        if decision_chain.steps:
            top = decision_chain.steps[0]
            parts.append(f"\n{'=' * 50}")
            parts.append(f"PRIORITY ACTION: {top.action}")

        return "\n".join(parts)

    def assemble_uncertainty(self, assessment) -> str:
        """Format an uncertainty assessment as human-readable text."""
        parts = []
        parts.append(f"I don't have sufficient evidence. {assessment.reason}.")

        if assessment.partial_knowledge:
            parts.append("\nWhat I do know:")
            for i, fact in enumerate(assessment.partial_knowledge, 1):
                parts.append(f"  {i}. {fact}")

        if assessment.missing_evidence:
            parts.append("\nTo answer fully, I would need:")
            for item in assessment.missing_evidence:
                parts.append(f"  - {item}")

        return "\n".join(parts)
