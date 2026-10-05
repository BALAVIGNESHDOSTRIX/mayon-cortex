# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# PROPRIETARY AND CONFIDENTIAL
# ==============================================================================

"""
Discourse Coherence Tracker — Multi-Sentence Topic Flow
=========================================================
Human brain analogy:
  When you read a paragraph, you maintain a "mental thread" —
  a sense of what the topic IS and how each new sentence connects.
  This is why paragraphs feel unified, not random.

Currently Mayon-Cortex assembles each sentence independently
from proof paths — there is no thread connecting them.

This module adds:
  1. TopicMemory — tracks active concepts, entities, pronouns across sentences
  2. CoherenceScorer — scores how well a candidate sentence connects to the thread
  3. DiscourseConnector — adds appropriate transition words (however, therefore, furthermore...)
  4. ParagraphPlanner — plans sentence ORDER for maximum coherence

Result: Multi-sentence outputs feel like real paragraphs, not disconnected facts.
"""

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Discourse Relation Types
# ──────────────────────────────────────────────────────────────────────────────

class DiscourseRelation(Enum):
    """How a new sentence relates to the previous one."""
    ELABORATION  = "elaboration"    # adds detail to same topic
    CONTRAST     = "contrast"       # contradicts or contrasts
    CAUSE        = "cause"          # explains why
    EFFECT       = "effect"         # what happened as a result
    EXAMPLE      = "example"        # illustrates the previous point
    SUMMARY      = "summary"        # wraps up the topic
    SEQUENCE     = "sequence"       # next step in a process
    CONDITION    = "condition"      # if-then relationship
    ADDITION     = "addition"       # adds a new related point


# Transition words by discourse relation
DISCOURSE_CONNECTORS: Dict[DiscourseRelation, List[str]] = {
    DiscourseRelation.ELABORATION: [
        "Furthermore,", "In addition,", "More specifically,",
        "To elaborate,", "Additionally,", "Indeed,",
    ],
    DiscourseRelation.CONTRAST: [
        "However,", "On the other hand,", "In contrast,",
        "Nevertheless,", "Despite this,", "Conversely,",
    ],
    DiscourseRelation.CAUSE: [
        "This is because", "The reason is that", "Due to",
        "As a result of", "Given that",
    ],
    DiscourseRelation.EFFECT: [
        "As a result,", "Consequently,", "Therefore,",
        "This leads to", "Hence,", "Thus,",
    ],
    DiscourseRelation.EXAMPLE: [
        "For example,", "For instance,", "To illustrate,",
        "As an example,", "Specifically,",
    ],
    DiscourseRelation.SUMMARY: [
        "In summary,", "To summarize,", "Overall,",
        "In conclusion,", "Ultimately,",
    ],
    DiscourseRelation.SEQUENCE: [
        "First,", "Next,", "Then,", "Subsequently,",
        "Finally,", "Following this,",
    ],
    DiscourseRelation.ADDITION: [
        "Moreover,", "Also,", "Furthermore,",
        "In addition to this,", "Besides,",
    ],
    DiscourseRelation.CONDITION: [
        "If this is the case,", "Under these conditions,",
        "Provided that", "In such cases,",
    ],
}

# Keywords that signal each discourse relation
RELATION_SIGNALS: Dict[DiscourseRelation, Set[str]] = {
    DiscourseRelation.CONTRAST:  {"however", "but", "although", "despite", "unlike", "whereas", "yet"},
    DiscourseRelation.CAUSE:     {"because", "since", "due", "reason", "cause", "result of"},
    DiscourseRelation.EFFECT:    {"therefore", "thus", "hence", "consequently", "so", "leads to"},
    DiscourseRelation.EXAMPLE:   {"example", "instance", "such as", "like", "including", "e.g."},
    DiscourseRelation.SEQUENCE:  {"first", "second", "next", "then", "finally", "step", "after"},
    DiscourseRelation.SUMMARY:   {"summary", "conclude", "overall", "thus", "in short"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Topic Memory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicFrame:
    """The current topic frame — what the discourse is about."""
    primary_topic: str = ""
    active_entities: List[str] = field(default_factory=list)   # nouns/concepts mentioned
    active_relations: List[str] = field(default_factory=list)  # verbs/predicates
    sentence_count: int = 0
    lexical_chain: List[str] = field(default_factory=list)     # word repetition chain


class TopicMemory:
    """
    Tracks the discourse topic across multiple sentences.
    Maintains a sliding window of active concepts and entities.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._sentence_window: deque = deque(maxlen=window_size)
        self._entity_freq: Counter = Counter()
        self._relation_freq: Counter = Counter()
        self.frame = TopicFrame()

    def update(self, sentence: str):
        """Update topic memory with a new sentence."""
        self._sentence_window.append(sentence)
        self.frame.sentence_count += 1

        words = self._extract_content_words(sentence)
        for w in words:
            self._entity_freq[w] += 1

        # Update primary topic (most frequent content word overall)
        if self._entity_freq:
            self.frame.primary_topic = self._entity_freq.most_common(1)[0][0]

        # Update active entities (top-5 most recently/frequently mentioned)
        self.frame.active_entities = [w for w, _ in self._entity_freq.most_common(5)]
        self.frame.lexical_chain = list(words)

    def get_active_concepts(self) -> List[str]:
        """Return concepts currently active in working memory."""
        return self.frame.active_entities

    def lexical_overlap(self, sentence: str) -> float:
        """
        Measure lexical overlap between candidate sentence and topic memory.
        High overlap = coherent continuation.
        """
        words = set(self._extract_content_words(sentence))
        active = set(self.frame.active_entities)
        if not words:
            return 0.0
        return len(words & active) / len(words)

    @staticmethod
    def _extract_content_words(text: str) -> List[str]:
        """Extract meaningful content words (nouns, verbs) from text."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "and", "or", "but",
            "if", "in", "on", "at", "to", "for", "of", "with", "by",
            "this", "that", "these", "those", "it", "its", "i", "you",
            "he", "she", "we", "they", "not", "no", "can", "will", "would",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in stop]

    def reset(self):
        """Reset topic memory (e.g. for a new paragraph)."""
        self._sentence_window.clear()
        self._entity_freq.clear()
        self.frame = TopicFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Coherence Scorer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoherenceScore:
    """Coherence assessment for a candidate sentence."""
    sentence: str
    score: float                     # 0.0 (incoherent) to 1.0 (perfectly coherent)
    lexical_overlap: float
    entity_continuity: float
    relation_type: DiscourseRelation
    suggested_connector: str = ""


class CoherenceScorer:
    """
    Scores how coherent a candidate sentence is given the topic memory.
    Used by SentenceAssembler to pick the most coherent next sentence.
    """

    def __init__(self, topic_memory: TopicMemory):
        self.memory = topic_memory

    def score(self, candidate: str) -> CoherenceScore:
        """Score a candidate sentence for coherence with the topic."""
        # 1. Lexical overlap with active topic
        lex = self.memory.lexical_overlap(candidate)

        # 2. Entity continuity (does it mention an entity we know about?)
        active = set(self.memory.get_active_concepts())
        cand_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", candidate.lower()))
        entity_cont = len(cand_words & active) / max(len(active), 1)

        # 3. Detect discourse relation
        rel = self._detect_relation(candidate)

        # 4. Combined score
        score = 0.5 * lex + 0.5 * entity_cont

        # Penalty for completely unrelated sentences
        if score < 0.05 and self.memory.frame.sentence_count > 0:
            score = 0.0

        # Connector suggestion
        connectors = DISCOURSE_CONNECTORS.get(rel, [""])
        connector = connectors[self.memory.frame.sentence_count % len(connectors)] if connectors else ""

        return CoherenceScore(
            sentence=candidate,
            score=score,
            lexical_overlap=lex,
            entity_continuity=entity_cont,
            relation_type=rel,
            suggested_connector=connector,
        )

    def rank_candidates(self, candidates: List[str]) -> List[CoherenceScore]:
        """Rank a list of candidate sentences by coherence score."""
        scored = [self.score(c) for c in candidates]
        return sorted(scored, key=lambda s: s.score, reverse=True)

    @staticmethod
    def _detect_relation(sentence: str) -> DiscourseRelation:
        """Detect what discourse relation the sentence expresses."""
        lower = sentence.lower()
        for rel, signals in RELATION_SIGNALS.items():
            if any(sig in lower for sig in signals):
                return rel
        return DiscourseRelation.ELABORATION


# ──────────────────────────────────────────────────────────────────────────────
# Paragraph Planner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParagraphPlan:
    """A planned paragraph with ordered sentences and connectors."""
    sentences: List[str]
    connectors: List[str]   # connector[i] goes before sentence[i] (empty for first)
    topic: str
    coherence_scores: List[float]

    def render(self) -> str:
        """Render the paragraph as a coherent text block."""
        parts = []
        for i, (sent, conn) in enumerate(zip(self.sentences, self.connectors)):
            if i == 0 or not conn:
                parts.append(sent)
            else:
                # Add connector before the sentence
                if sent and sent[0].isupper():
                    sent = sent[0].lower() + sent[1:]
                parts.append(f"{conn} {sent}")
        return " ".join(parts)


class DiscourseCoherenceTracker:
    """
    Main coherence engine — plans and generates coherent multi-sentence paragraphs.

    Usage:
        tracker = DiscourseCoherenceTracker()
        tracker.start_paragraph("aspirin treats hypertension")
        paragraph = tracker.build_paragraph(candidate_sentences, max_sentences=4)
        print(paragraph.render())
    """

    def __init__(self, window_size: int = 5):
        self.memory = TopicMemory(window_size=window_size)
        self.scorer = CoherenceScorer(self.memory)
        self._paragraph_sentences: List[str] = []

    def start_paragraph(self, topic_hint: str = ""):
        """Start a new paragraph, reset topic memory."""
        self.memory.reset()
        self._paragraph_sentences = []
        if topic_hint:
            self.memory.frame.primary_topic = topic_hint.split()[0] if topic_hint else ""

    def add_sentence(self, sentence: str):
        """Add a sentence to the current paragraph and update topic memory."""
        self._paragraph_sentences.append(sentence)
        self.memory.update(sentence)

    def build_paragraph(
        self,
        candidates: List[str],
        max_sentences: int = 4,
        min_coherence: float = 0.1,
    ) -> ParagraphPlan:
        """
        Build a coherent paragraph by selecting and ordering candidate sentences.

        Args:
            candidates: Pool of candidate sentences to choose from
            max_sentences: Maximum sentences in the paragraph
            min_coherence: Minimum coherence score to include a sentence

        Returns:
            ParagraphPlan with ordered sentences + connectors
        """
        selected: List[str] = []
        connectors: List[str] = []
        scores: List[float] = []
        remaining = list(candidates)

        for i in range(max_sentences):
            if not remaining:
                break

            # Score all remaining candidates
            ranked = self.scorer.rank_candidates(remaining)

            # Pick best coherent candidate
            best = None
            for scored in ranked:
                if scored.score >= min_coherence or i == 0:
                    best = scored
                    break

            if best is None:
                break

            # Apply connector (not for first sentence)
            conn = best.suggested_connector if i > 0 else ""
            selected.append(best.sentence)
            connectors.append(conn)
            scores.append(best.score)

            # Update topic memory
            self.memory.update(best.sentence)
            remaining.remove(best.sentence)

        return ParagraphPlan(
            sentences=selected,
            connectors=connectors,
            topic=self.memory.frame.primary_topic,
            coherence_scores=scores,
        )

    def improve_sentence_flow(self, sentences: List[str]) -> str:
        """
        Take a list of pre-generated sentences and improve their flow
        by adding discourse connectors between them.
        """
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        self.start_paragraph()
        parts = [sentences[0]]
        self.memory.update(sentences[0])

        for sent in sentences[1:]:
            cs = self.scorer.score(sent)
            conn = cs.suggested_connector
            self.memory.update(sent)

            if conn:
                # Lowercase first letter of sentence after connector
                s = sent[0].lower() + sent[1:] if sent else sent
                parts.append(f"{conn} {s}")
            else:
                parts.append(sent)

        return " ".join(parts)

    def score_paragraph(self, text: str) -> float:
        """
        Score an already-written paragraph for coherence.
        Returns 0.0 (incoherent) to 1.0 (perfectly coherent).
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) < 2:
            return 1.0

        self.start_paragraph()
        total_score = 0.0

        for i, sent in enumerate(sentences):
            if i == 0:
                self.memory.update(sent)
                total_score += 1.0
            else:
                cs = self.scorer.score(sent)
                total_score += cs.score
                self.memory.update(sent)

        return total_score / len(sentences)
