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
Grammar Pattern Inducer — Learn English Grammar from Reading
=============================================================
Human brain analogy:
  A child reads "The cat sat on the mat."
  Their brain INDUCES: DET + NOUN + VERB_PAST + PREP + DET + NOUN
  After 1000 sentences → knows 200+ valid English patterns.
  They never needed a grammar textbook.

Same here:
  1. Reads ingested sentences
  2. POS-tags each word using suffix rules + known word lists
  3. Extracts the POS sequence as a grammar "pattern"
  4. Counts frequency → high-frequency patterns = valid English
  5. SentenceAssembler uses these patterns to build better sentences

Result: Mayon-Cortex generates grammatically correct, varied sentences
        without any external grammar rules.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# POS Tagger (rule-based, no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "all", "both", "each", "every", "few", "many", "much", "several", "no"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "who", "which", "what"}
PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "under", "over", "around", "near", "within", "without", "against", "along", "across"}
CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "because", "although", "since", "while", "if", "unless", "therefore", "however", "whereas", "when", "that", "which"}
AUXILIARIES = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "must", "ought"}
NEGATIONS = {"not", "never", "no", "neither", "nor"}
PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "(", ")", "\"", "'"}

COMMON_ADJECTIVES = {
    "good", "bad", "new", "old", "great", "small", "large", "big", "little",
    "long", "short", "high", "low", "early", "young", "important", "public",
    "different", "real", "best", "right", "following", "able", "few", "main",
    "other", "last", "next", "first", "free", "strong", "own", "same", "full",
    "sure", "clear", "open", "possible", "true", "false", "natural", "human",
    "common", "local", "major", "social", "recent", "cold", "hot", "dark",
    "light", "hard", "soft", "fast", "slow", "deep", "close", "white", "black",
    "red", "green", "blue", "yellow", "orange", "brown", "general", "specific",
}

COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "say", "says", "said", "get", "gets", "got",
    "make", "makes", "made", "go", "goes", "went", "know", "knows", "knew",
    "think", "thinks", "thought", "take", "takes", "took", "see", "sees", "saw",
    "come", "comes", "came", "want", "wants", "wanted", "use", "uses", "used",
    "find", "finds", "found", "give", "gives", "gave", "tell", "tells", "told",
    "work", "works", "worked", "call", "calls", "called", "need", "needs", "needed",
    "feel", "feels", "felt", "become", "becomes", "became", "show", "shows", "showed",
    "provide", "provides", "provided", "include", "includes", "included",
    "consider", "considers", "considered", "contain", "contains", "contained",
    "produce", "produces", "produced", "allow", "allows", "allowed",
    "require", "requires", "required", "involve", "involves", "involved",
    "treat", "treats", "treated", "cause", "causes", "caused", "prevent",
    "prevents", "prevented", "reduce", "reduces", "reduced", "increase",
    "increases", "increased", "improve", "improves", "improved",
}


def pos_tag(word: str) -> str:
    """
    Rule-based POS tagger. No external libraries.
    Returns POS tag string: DET, PRON, PREP, CONJ, AUX, NEG,
    PUNCT, ADJ, ADV, VERB, VERB_PAST, VERB_GER, NOUN, NUM
    """
    w = word.lower().strip()

    if w in PUNCTUATION:
        return "PUNCT"
    if w in DETERMINERS:
        return "DET"
    if w in PRONOUNS:
        return "PRON"
    if w in PREPOSITIONS:
        return "PREP"
    if w in CONJUNCTIONS:
        return "CONJ"
    if w in AUXILIARIES:
        return "AUX"
    if w in NEGATIONS:
        return "NEG"
    if w in COMMON_ADJECTIVES:
        return "ADJ"
    if w in COMMON_VERBS:
        return "VERB"

    # Suffix rules (in priority order)
    if re.match(r"^\d+(\.\d+)?$", w):
        return "NUM"
    if re.match(r".*ingly$|.*ously$|.*fully$|.*lessly$|.*wardly?$", w):
        return "ADV"
    if w.endswith("ly") and len(w) > 4:
        return "ADV"
    if w.endswith("ing") and len(w) > 5:
        return "VERB_GER"
    if w.endswith("ed") and len(w) > 4:
        return "VERB_PAST"
    if re.match(r".*able$|.*ible$|.*ous$|.*ful$|.*less$|.*ive$|.*ish$|.*al$|.*ic$", w):
        return "ADJ"
    if re.match(r".*tion$|.*ment$|.*ness$|.*ity$|.*ance$|.*ence$|.*ship$|.*dom$|.*ism$|.*ist$|.*ure$|.*age$", w):
        return "NOUN"
    if re.match(r".*ize$|.*ise$|.*ify$|.*en$", w) and len(w) > 4:
        return "VERB"

    # Default: NOUN (most common in English)
    return "NOUN"


def pos_tag_sentence(sentence: str) -> List[Tuple[str, str]]:
    """Tag all words in a sentence. Returns (word, POS) pairs."""
    words = re.sub(r"([.,!?;:\"'()])", r" \1 ", sentence).split()
    return [(w, pos_tag(w)) for w in words if w.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Grammar Pattern
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GrammarPattern:
    """A POS sequence pattern with frequency and example sentences."""
    pos_sequence: Tuple[str, ...]
    frequency: int = 0
    examples: List[str] = field(default_factory=list)
    max_examples: int = 3

    @property
    def pattern_str(self) -> str:
        return " → ".join(self.pos_sequence)

    @property
    def length(self) -> int:
        return len(self.pos_sequence)

    def add_example(self, sentence: str):
        if len(self.examples) < self.max_examples:
            self.examples.append(sentence)


@dataclass
class InductionStats:
    """Statistics from grammar induction."""
    sentences_processed: int = 0
    unique_patterns: int = 0
    top_patterns: List[GrammarPattern] = field(default_factory=list)
    pos_frequency: Dict[str, int] = field(default_factory=dict)
    avg_sentence_length: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Grammar Pattern Inducer
# ──────────────────────────────────────────────────────────────────────────────

class GrammarPatternInducer:
    """
    Induces valid English grammar patterns from ingested sentences.

    After induction:
    - knows which POS sequences are grammatically valid
    - can validate if a generated sentence is grammatical
    - can suggest next POS tag given current POS sequence
    - feeds into SentenceAssembler for better sentence construction
    """

    # Valid transitions between POS tags (what can follow what)
    VALID_TRANSITIONS: Dict[str, Set[str]] = {
        "DET":       {"NOUN", "ADJ", "ADV", "NUM", "VERB_GER"},
        "ADJ":       {"NOUN", "ADJ", "CONJ", "PUNCT", "NOUN"},
        "NOUN":      {"VERB", "AUX", "VERB_PAST", "VERB_GER", "PREP", "CONJ", "ADV", "PUNCT", "NOUN"},
        "PRON":      {"VERB", "AUX", "VERB_PAST", "VERB_GER", "ADV", "NEG", "PREP"},
        "VERB":      {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT", "VERB_GER", "AUX"},
        "VERB_PAST": {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT"},
        "VERB_GER":  {"NOUN", "PRON", "ADJ", "ADV", "DET", "PREP", "CONJ", "PUNCT"},
        "AUX":       {"VERB", "VERB_PAST", "VERB_GER", "ADV", "NEG", "ADJ", "NOUN"},
        "ADV":       {"VERB", "VERB_PAST", "VERB_GER", "ADJ", "ADV", "PUNCT", "NOUN"},
        "PREP":      {"DET", "NOUN", "PRON", "ADJ", "NUM", "VERB_GER"},
        "CONJ":      {"DET", "NOUN", "PRON", "ADJ", "ADV", "VERB", "VERB_PAST", "AUX"},
        "NEG":       {"VERB", "VERB_PAST", "VERB_GER", "ADJ", "NOUN", "AUX"},
        "NUM":       {"NOUN", "ADJ", "PUNCT", "PREP"},
        "PUNCT":     {"DET", "NOUN", "PRON", "ADJ", "ADV", "CONJ", "AUX"},
    }

    def __init__(self, min_pattern_freq: int = 2, max_pattern_len: int = 8):
        self.min_pattern_freq = min_pattern_freq
        self.max_pattern_len = max_pattern_len
        self._raw_counts: Counter = Counter()
        self._pattern_examples: Dict[Tuple, List[str]] = defaultdict(list)
        self._transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self._pos_freq: Counter = Counter()
        self.patterns: Dict[Tuple[str, ...], GrammarPattern] = {}

    def induce_from_sentences(self, sentences: List[str]) -> InductionStats:
        """
        Learn grammar patterns from a list of sentences.
        Call multiple times to keep learning (online learning).
        """
        total_len = 0
        processed = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 3:
                continue

            tagged = pos_tag_sentence(sent)
            if not tagged:
                continue

            words, tags = zip(*tagged)
            total_len += len(tags)
            processed += 1

            # Count individual POS frequencies
            for tag in tags:
                self._pos_freq[tag] += 1

            # Count transitions
            for i in range(len(tags) - 1):
                self._transition_counts[tags[i]][tags[i + 1]] += 1

            # Extract n-gram POS patterns (length 3 to max_pattern_len)
            for n in range(3, min(self.max_pattern_len + 1, len(tags) + 1)):
                for start in range(len(tags) - n + 1):
                    pattern = tuple(tags[start:start + n])
                    self._raw_counts[pattern] += 1
                    if len(self._pattern_examples[pattern]) < 3:
                        self._pattern_examples[pattern].append(sent)

        # Build final pattern objects (filter by frequency)
        self.patterns = {}
        for pattern, freq in self._raw_counts.items():
            if freq >= self.min_pattern_freq:
                p = GrammarPattern(pos_sequence=pattern, frequency=freq)
                for ex in self._pattern_examples.get(pattern, []):
                    p.add_example(ex)
                self.patterns[pattern] = p

        # Sort by frequency
        top = sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True)[:20]

        return InductionStats(
            sentences_processed=processed,
            unique_patterns=len(self.patterns),
            top_patterns=top,
            pos_frequency=dict(self._pos_freq),
            avg_sentence_length=total_len / max(processed, 1),
        )

    def is_grammatical(self, sentence: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Check if a sentence is grammatically valid based on learned patterns.
        Returns (is_valid, confidence_score).
        """
        tagged = pos_tag_sentence(sentence)
        if len(tagged) < 2:
            return False, 0.0

        _, tags = zip(*tagged)

        # Check transitions
        valid_transitions = 0
        total_transitions = len(tags) - 1

        for i in range(total_transitions):
            curr, nxt = tags[i], tags[i + 1]
            allowed = self.VALID_TRANSITIONS.get(curr, set())
            learned_freq = self._transition_counts[curr].get(nxt, 0)
            if nxt in allowed or learned_freq > 0:
                valid_transitions += 1

        score = valid_transitions / max(total_transitions, 1)
        return score >= threshold, score

    def suggest_next_pos(self, current_pos_sequence: List[str], top_k: int = 3) -> List[str]:
        """
        Given a POS sequence so far, suggest the most likely next POS tags.
        Used by SentenceAssembler to guide generation.
        """
        if not current_pos_sequence:
            return ["DET", "PRON", "NOUN"]

        last_pos = current_pos_sequence[-1]

        # 1. From learned transitions (data-driven)
        learned = self._transition_counts.get(last_pos, Counter())
        if learned:
            return [pos for pos, _ in learned.most_common(top_k)]

        # 2. Fall back to rule-based valid transitions
        allowed = list(self.VALID_TRANSITIONS.get(last_pos, {"NOUN"}))
        return allowed[:top_k]

    def get_template_for_length(self, target_len: int) -> Optional[List[str]]:
        """
        Return a grammar template (POS sequence) matching the target word count.
        """
        candidates = [
            p for p in self.patterns.values()
            if abs(p.length - target_len) <= 2 and p.frequency >= self.min_pattern_freq
        ]
        if not candidates:
            return None
        # Pick highest-frequency template
        best = max(candidates, key=lambda p: p.frequency)
        return list(best.pos_sequence)

    def reorder_words_by_grammar(self, words: List[str]) -> List[str]:
        """
        Attempt to reorder a bag of words into a grammatically valid sequence
        using the most likely POS template for that length.
        """
        if len(words) < 2:
            return words

        template = self.get_template_for_length(len(words))
        if not template:
            return words

        # Tag all words
        tagged = [(w, pos_tag(w)) for w in words]

        # Build a pool by POS
        pos_pool: Dict[str, List[str]] = defaultdict(list)
        for w, pos in tagged:
            pos_pool[pos].append(w)

        # Fill template slots
        result = []
        used: Set[str] = set()
        for slot_pos in template:
            candidates = [w for w in pos_pool.get(slot_pos, []) if w not in used]
            if candidates:
                chosen = candidates[0]
                result.append(chosen)
                used.add(chosen)
            else:
                # Try semantic fallback
                remaining = [w for w, _ in tagged if w not in used]
                if remaining:
                    result.append(remaining[0])
                    used.add(remaining[0])

        # Append any unused words at the end
        for w, _ in tagged:
            if w not in used:
                result.append(w)

        return result

    def summary(self) -> str:
        """Return a human-readable summary of learned patterns."""
        lines = [
            f"GrammarPatternInducer Summary",
            f"  Total patterns: {len(self.patterns)}",
            f"  POS frequencies: {dict(self._pos_freq.most_common(8))}",
            f"  Top 5 patterns:",
        ]
        top5 = sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True)[:5]
        for p in top5:
            lines.append(f"    [{p.frequency}x] {p.pattern_str}")
            if p.examples:
                lines.append(f"           e.g. \"{p.examples[0][:80]}\"")
        return "\n".join(lines)
