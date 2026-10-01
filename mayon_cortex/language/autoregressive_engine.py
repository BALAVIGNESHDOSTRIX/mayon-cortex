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
Graph Autoregressive Engine — Graph-Native Next-Token Generation
================================================================
Pillar of Cortex-Graph Natural Language Articulation.

Replaces neural transformer next-token prediction with pure graph traversal:
1. Multi-hop Context Spreading: Attends to recent N words via activation decay
2. Dynamic Theme Attractor: Maintains persistent global topic coherence
3. Bigram, Skip-gram, and Trigram Transition Topology
4. POS Grammar Constraint Filtering
5. Boltzmann Nucleus (Top-P) & Top-K Sampling
"""

import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from mayon_cortex.language.word_graph import WordGraph, WordNode


# POS Transition grammar matrix (validity priors)
POS_TRANSITIONS: Dict[str, Set[str]] = {
    "DET": {"NOUN", "ADJ", "ADV", "NUM"},
    "ADJ": {"NOUN", "ADJ", "CONJ"},
    "NOUN": {"VERB", "VERB_PAST", "VERB_GER", "PREP", "CONJ", "ADV", "PUNCT"},
    "PRON": {"VERB", "VERB_PAST", "VERB_GER", "ADV", "AUX"},
    "VERB": {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT"},
    "VERB_PAST": {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT"},
    "VERB_GER": {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT"},
    "ADV": {"VERB", "VERB_PAST", "VERB_GER", "ADJ", "ADV", "PUNCT"},
    "PREP": {"DET", "NOUN", "PRON", "ADJ", "NUM"},
    "CONJ": {"DET", "NOUN", "PRON", "ADJ", "ADV", "VERB", "VERB_PAST"},
    "PUNCT": {"DET", "NOUN", "PRON", "ADJ", "ADV", "CONJ"},
    "START": {"DET", "NOUN", "PRON", "ADJ", "ADV", "CONJ"},
}


@dataclass
class TextGenerationConfig:
    """Hyperparameters for graph autoregressive text generation."""
    max_tokens: int = 150
    temperature: float = 0.75
    top_k: int = 40
    top_p: float = 0.90
    repetition_penalty: float = 1.3
    context_window: int = 32
    theme_weight: float = 0.35
    transition_weight: float = 0.40
    pos_weight: float = 0.15
    frequency_weight: float = 0.10


class GraphAutoRegressiveEngine:
    """
    Graph-native autoregressive language engine.
    Performs next-token predictions through multi-hop graph activation.
    """

    def __init__(self, word_graph: Optional[WordGraph] = None):
        self.word_graph = word_graph or WordGraph()
        self.trigram_counts: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def learn_text(self, text: str):
        """Ingest text into word graph and trigram transition matrix."""
        self.word_graph.learn_text(text)
        tokens = [w.strip().lower() for w in re.findall(r"\b\w+\b|[.,!?;]", text) if w.strip()]
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            self.trigram_counts[(w1, w2)][w3] += 1.0

    def _get_context_vector(self, recent_words: List[str]) -> Optional[np.ndarray]:
        """Compute exponential decay weighted average vector over recent context tokens."""
        if not recent_words:
            return None

        vecs = []
        weights = []
        decay_factor = 0.9

        for idx, word in enumerate(reversed(recent_words)):
            node = self.word_graph.get_node(word)
            if node is not None and node.vector is not None:
                w = math.pow(decay_factor, idx)
                vecs.append(node.vector * w)
                weights.append(w)

        if not vecs:
            return None

        sum_vec = np.sum(vecs, axis=0)
        norm = np.linalg.norm(sum_vec) + 1e-7
        return (sum_vec / norm).astype(np.float32)

    def _score_candidates(
        self,
        candidates: Set[str],
        recent_words: List[str],
        theme_vec: Optional[np.ndarray],
        last_word: str,
        second_last_word: Optional[str],
        last_pos: str,
        config: TextGenerationConfig,
    ) -> Dict[str, float]:
        """
        Score each candidate next-word token based on graph transitions,
        theme alignment, skip-grams, trigrams, and POS grammar.
        """
        scores: Dict[str, float] = {}
        last_node = self.word_graph.get_node(last_word)

        # Trigram candidates for the 2-word prefix
        trigram_targets = {}
        if second_last_word:
            trigram_targets = self.trigram_counts.get((second_last_word, last_word), {})

        # Recent words frequency map for repetition penalty
        recent_counts = Counter(recent_words[-16:])

        valid_next_pos = POS_TRANSITIONS.get(last_pos, set())

        for word in candidates:
            node = self.word_graph.get_node(word)
            if node is None:
                continue

            # 1. Bigram transition score
            bigram_prob = 0.0
            if last_node is not None:
                bigram_prob = last_node.next_words.get(word, 0.0)

            # 2. Trigram transition bonus
            trigram_prob = trigram_targets.get(word, 0.0)

            # 3. Skip-gram lookahead score
            skip_prob = 0.0
            if second_last_word:
                s_node = self.word_graph.get_node(second_last_word)
                if s_node is not None:
                    skip_prob = s_node.skip_1.get(word, 0.0)

            transition_score = bigram_prob * 0.6 + trigram_prob * 0.3 + skip_prob * 0.1

            # 4. Semantic theme alignment (cosine similarity)
            semantic_score = 0.0
            if theme_vec is not None and node.vector is not None:
                semantic_score = max(0.0, float(np.dot(node.vector, theme_vec)))

            # 5. POS grammar alignment
            pos_score = 1.0 if (node.pos in valid_next_pos or not valid_next_pos) else 0.2

            # 6. Frequency prior
            freq_score = math.log1p(node.frequency) / 10.0

            # Holistic combination
            raw_score = (
                config.transition_weight * transition_score +
                config.theme_weight * semantic_score +
                config.pos_weight * pos_score +
                config.frequency_weight * freq_score
            )

            # 7. Repetition penalty
            if word in recent_counts:
                raw_score /= (config.repetition_penalty ** recent_counts[word])

            scores[word] = raw_score

        return scores

    def _sample(
        self,
        scores: Dict[str, float],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        """Boltzmann nucleus sampling over candidate tokens."""
        if not scores:
            return "."

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_items = sorted_items[:max(1, top_k)]

        words = [it[0] for it in sorted_items]
        raw_vals = np.array([it[1] for it in sorted_items], dtype=np.float64)

        temp = max(1e-4, temperature)
        scaled_vals = (raw_vals - np.max(raw_vals)) / temp
        exp_vals = np.exp(np.clip(scaled_vals, -50.0, 50.0))
        probs = exp_vals / (np.sum(exp_vals) + 1e-9)

        # Top-P (Nucleus) Truncation
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1

        valid_indices = sorted_indices[:cutoff]
        valid_words = [words[i] for i in valid_indices]
        valid_probs = probs[valid_indices]
        valid_probs /= np.sum(valid_probs)

        idx = np.random.choice(len(valid_words), p=valid_probs)
        return valid_words[idx]

    def generate(
        self,
        prompt: str,
        config: Optional[TextGenerationConfig] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_words: Optional[Set[str]] = None,
    ) -> str:
        """
        Generate continuous text continuation from prompt using graph autoregression.
        """
        cfg = config or TextGenerationConfig()
        if max_tokens is not None:
            cfg.max_tokens = max_tokens
        if temperature is not None:
            cfg.temperature = temperature

        stop_set = stop_words or {"<EOS>", "<END>"}

        # Tokenize prompt
        prompt_tokens = [w.strip().lower() for w in re.findall(r"\b\w+\b|[.,!?;]", prompt) if w.strip()]
        if not prompt_tokens:
            prompt_tokens = ["the"]

        generated_words: List[str] = list(prompt_tokens)

        # Extract theme vector from prompt tokens
        theme_vec = self._get_context_vector(prompt_tokens)

        for step in range(cfg.max_tokens):
            last_word = generated_words[-1]
            second_last = generated_words[-2] if len(generated_words) >= 2 else None

            last_node = self.word_graph.get_node(last_word)
            last_pos = last_node.pos if last_node else "NOUN"

            # 1. Determine candidate set of next words
            candidates: Set[str] = set()

            # Immediate outgoing edges from last word node
            if last_node is not None:
                candidates.update(last_node.next_words.keys())
                candidates.update(last_node.skip_1.keys())

            # Trigram candidates
            if second_last:
                tri_dict = self.trigram_counts.get((second_last, last_word))
                if tri_dict:
                    candidates.update(tri_dict.keys())

            # Fallback: if candidate set is too small, retrieve nearest semantic neighbors
            if len(candidates) < 5:
                top_nodes = self.word_graph.get_all_nodes()[:50]
                candidates.update([n.word for n in top_nodes])

            # 2. Score candidates
            recent_window = generated_words[-cfg.context_window:]
            scores = self._score_candidates(
                candidates=candidates,
                recent_words=recent_window,
                theme_vec=theme_vec,
                last_word=last_word,
                second_last_word=second_last,
                last_pos=last_pos,
                config=cfg,
            )

            # 3. Sample next word
            next_word = self._sample(
                scores=scores,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            if next_word in stop_set:
                break

            generated_words.append(next_word)

            # Update rolling theme vector every 8 tokens
            if step % 8 == 0:
                new_theme = self._get_context_vector(generated_words[-16:])
                if new_theme is not None and theme_vec is not None:
                    theme_vec = 0.7 * theme_vec + 0.3 * new_theme
                    theme_vec /= (np.linalg.norm(theme_vec) + 1e-7)

        # 4. Format and assemble clean punctuation & capitalization
        return self._format_text(generated_words)

    def _format_text(self, tokens: List[str]) -> str:
        """Assemble tokens into grammatically punctuated text with correct spacing."""
        if not tokens:
            return ""

        punct = {".", ",", "!", "?", ";", ":"}
        out = []
        capitalize_next = True

        for tok in tokens:
            if tok in punct:
                if out:
                    out[-1] += tok
                else:
                    out.append(tok)
                if tok in {".", "!", "?"}:
                    capitalize_next = True
            else:
                word = tok.capitalize() if capitalize_next else tok
                out.append(word)
                capitalize_next = False

        return " ".join(out)
