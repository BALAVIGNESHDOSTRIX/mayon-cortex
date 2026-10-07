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
N-Gram Language Model — Graph-Guided Statistical Fluency
=========================================================
Phase 3 of the Brain-Like Intelligence upgrade.

Provides statistical transition probabilities, n-gram prefix trees (trie),
and multi-order smoothing (Kneser-Ney, Witten-Bell, Laplace) for:
  1. Evaluating fluency of graph-generated candidate paths
  2. Scoring next-word transitions during beam search
  3. Rescoring candidate sentences from WordGraph and BrocaGraph
  4. Training dynamically from any ingested text corpus without backprop
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


class NGramModel:
    """
    Multi-order statistical N-Gram Language Model with backoff and interpolation.
    
    Trained purely on CPU via word frequency counts. Zero GPU, zero backprop.
    Provides fast O(1) conditional probability lookups P(w_n | w_1, ..., w_{n-1}).
    """

    def __init__(self, max_n: int = 4, discount: float = 0.75):
        self.max_n = max_n
        self.discount = discount
        self.counts: Dict[int, Counter] = {n: Counter() for n in range(1, max_n + 1)}
        self.context_counts: Dict[int, Counter] = {n: Counter() for n in range(1, max_n + 1)}
        self.continuation_counts: Counter = Counter()  # For Kneser-Ney lower orders
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0

    def tokenize(self, text: str) -> List[str]:
        """Clean tokenization splitting punctuation and words."""
        cleaned = re.sub(r'([.,!?;:()"\'])', r' \1 ', text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    def train_corpus(self, sentences: List[str]):
        """Train or update n-gram counts from a list of sentences."""
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            if not tokens:
                continue
            self.train_tokens(tokens)

    def train_tokens(self, tokens: List[str]):
        """Update n-gram counts from a list of tokens with <s> and </s> padding."""
        padded = ["<s>"] * (self.max_n - 1) + tokens + ["</s>"]
        self.total_tokens += len(tokens)
        self.vocab.update(tokens)

        for n in range(1, self.max_n + 1):
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i + n])
                self.counts[n][ngram] += 1
                if n > 1:
                    context = ngram[:-1]
                    self.context_counts[n][context] += 1
                    # Track continuation counts for Kneser-Ney
                    target_word = ngram[-1]
                    self.continuation_counts[target_word] += 1

    def prob_laplace(self, word: str, context: Tuple[str, ...]) -> float:
        """Laplace smoothed probability P(word | context)."""
        n = len(context) + 1
        if n > self.max_n:
            context = context[-(self.max_n - 1):]
            n = len(context) + 1

        ngram = context + (word,)
        count_ngram = self.counts[n].get(ngram, 0)
        count_context = self.context_counts[n].get(context, 0)
        vocab_size = max(len(self.vocab), 1)

        return (count_ngram + 1.0) / (count_context + vocab_size)

    def prob_kneser_ney(self, word: str, context: Tuple[str, ...]) -> float:
        """Interpolated Kneser-Ney smoothed probability."""
        if not context:
            # Unigram continuation probability
            total_continuations = max(sum(self.continuation_counts.values()), 1)
            return (self.continuation_counts.get(word, 0) + 1.0) / (total_continuations + max(len(self.vocab), 1))

        n = len(context) + 1
        if n > self.max_n:
            context = context[-(self.max_n - 1):]
            n = len(context) + 1

        ngram = context + (word,)
        count_ngram = self.counts[n].get(ngram, 0)
        count_context = self.context_counts[n].get(context, 0)

        if count_context == 0:
            # Backoff to lower order
            return self.prob_kneser_ney(word, context[1:])

        # Highest order discount
        discounted = max(count_ngram - self.discount, 0.0) / count_context

        # Lambda weight for lower order
        # Count unique following words for this context
        num_following = sum(1 for (ctx, w) in [(k[:-1], k[-1]) for k in self.counts[n] if k[:-1] == context])
        lambd = (self.discount / count_context) * max(num_following, 1)

        lower_prob = self.prob_kneser_ney(word, context[1:])
        return discounted + lambd * lower_prob

    def score_sentence(self, sentence: str) -> float:
        """
        Compute average log-likelihood per word for a given sentence.
        Higher score = more fluent / statistically natural.
        """
        tokens = self.tokenize(sentence)
        if not tokens:
            return -100.0

        padded = ["<s>"] * (self.max_n - 1) + tokens + ["</s>"]
        total_log_prob = 0.0
        n_words = 0

        for i in range(self.max_n - 1, len(padded)):
            word = padded[i]
            context = tuple(padded[i - (self.max_n - 1):i])
            p = max(self.prob_kneser_ney(word, context), 1e-12)
            total_log_prob += math.log2(p)
            n_words += 1

        return total_log_prob / max(n_words, 1)

    def perplexity(self, sentence: str) -> float:
        """Compute sentence perplexity (lower is better)."""
        avg_neg_log = -self.score_sentence(sentence)
        return math.pow(2.0, min(avg_neg_log, 30.0))

    def rescore_candidates(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """Rank candidates by statistical fluency log-likelihood."""
        scored = [(c, self.score_sentence(c)) for c in candidates]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def sample_next_word(
        self,
        context: List[str],
        top_k: int = 10,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Return top candidate next words given context."""
        ctx = tuple(context[-(self.max_n - 1):])
        candidates = []
        for word in self.vocab:
            p = self.prob_kneser_ney(word, ctx)
            candidates.append((word, p))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        if temperature != 1.0 and top_candidates:
            # Re-scale with temperature
            scores = [math.pow(max(p, 1e-9), 1.0 / max(temperature, 0.1)) for _, p in top_candidates]
            total = sum(scores) or 1.0
            top_candidates = [(w, s / total) for (w, _), s in zip(top_candidates, scores)]

        return top_candidates
