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
Word Graph — The Dynamic Graph-Native Generative Language Engine
================================================================
Pillar 1 of Cortex-Graph: 100% Graph-Native Generation & Creative Poetic Composition.

Replaces fixed-token transformer LLMs with graph-native generative articulation:
1. Every word is a graph node with a 384-dim embedding, POS tags, VAD emotion,
   syllable count, stress pattern, rhyme sound key, frequency, and sector affinities.
2. Edges encode natural dynamic transitions:
   - `next-word`: sequential bigram transition with frequency-derived confidence
   - `skip-1` / `skip-2`: skip-gram contextual lookahead
   - `composed-of`: bridges between concepts and word tokens
3. Rhyme & Prosodic Graph Substrate:
   - Phonetic rhyme buckets (vowel + coda phonetic clusters)
   - Syllable budget budgeting and meter alignment (tetrameter, pentameter, ballad, couplets)
4. GraphPoetEngine:
   - Pure Energy-based Boltzmann Graph Traversal with backtrack-capable beam search.
   - Zero transformers: produces metered, rhyming, lyrical poetry and creative prose.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from mayon_cortex.core.config import CortexConfig
from mayon_cortex.affect.emotion import VAD, EmotionEngine


POS_SUFFIXES = [
    (r".*ing$", "VERB_GER"),
    (r".*ed$", "VERB_PAST"),
    (r".*ly$", "ADV"),
    (r".*able$|.*ible$|.*ous$|.*ful$|.*less$|.*ive$|.*ish$", "ADJ"),
    (r".*tion$|.*ment$|.*ness$|.*ity$|.*ance$|.*ence$|.*ship$|.*dom$", "NOUN"),
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "don", "should", "now", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
}

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our"}
PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through", "during", "before", "after", "above", "below", "beneath", "beyond", "across"}
CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "because", "although", "since", "while", "if", "unless", "therefore"}
PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}


# ─────────────────────────────────────────────────────────────────────────────
# Phonetics, Rhyme & Meter Analyzer
# ─────────────────────────────────────────────────────────────────────────────

RHYME_PHONEME_PATTERNS = [
    (r".*ight$|.*ite$|.*byte$", "AY_T"),
    (r".*ound$|.*bound$", "AW_N_D"),
    (r".*eep$|.*eap$", "IY_P"),
    (r".*old$|.*ould$", "OW_L_D"),
    (r".*ave$|.*wave$", "EY_V"),
    (r".*ain$|.*ane$|.*eign$", "EY_N"),
    (r".*ire$|.*yre$|.*igher$", "AY_R"),
    (r".*ark$|.*arc$", "AA_R_K"),
    (r".*art$|.*eart$", "AA_R_T"),
    (r".*ore$|.*oar$|.*oor$|.*our$", "AO_R"),
    (r".*air$|.*are$|.*ear$", "EH_R"),
    (r".*eer$|.*ier$", "IH_R"),
    (r".*ace$|.*ase$", "EY_S"),
    (r".*all$|.*awl$", "AO_L"),
    (r".*ing$", "IH_NG"),
    (r".*est$", "EH_S_T"),
    (r".*end$|.*friend$", "EH_N_D"),
    (r".*ow$|.*glow$|.*flow$|.*know$|.*grow$", "OW"),
    (r".*y$|.*sky$|.*fly$|.*high$|.*die$|.*eye$|.*lie$", "AY"),
    (r".*ee$|.*sea$|.*free$|.*tree$|.*see$", "IY"),
    (r".*un$|.*sun$|.*run$|.*one$|.*won$", "AH_N"),
    (r".*eam$|.*eem$", "IY_M"),
    (r".*time$|.*climb$|.*rhyme$|.*prime$|.*ime$", "AY_M"),
    (r".*oul$|.*ole$|.*oal$|.*oll$", "OW_L"),
    (r".*ind$|.*mind$|.*find$|.*blind$|.*wind$", "AY_N_D"),
    (r".*ar$|.*star$|.*far$|.*scar$", "AA_R"),
    (r".*one$|.*stone$|.*alone$|.*shone$", "OW_N"),
    (r".*oon$|.*moon$|.*soon$|.*dune$|.*une$", "UW_N"),
    (r".*ue$|.*true$|.*blue$|.*through$|.*new$|.*flew$|.*ew$", "UW"),
    (r".*race$|.*space$|.*grace$|.*trace$", "EY_S"),
    (r".*song$|.*long$|.*strong$|.*ong$", "AO_NG"),
    (r".*deep$|.*keep$|.*sleep$", "IY_P"),
    (r".*rise$|.*skies$|.*eyes$|.*wise$|.*ise$|.*ize$|.*ies$", "AY_Z"),
    (r".*breath$|.*death$", "EH_TH"),
    (r".*light$|.*bright$|.*night$|.*sight$", "AY_T"),
    (r".*power$|.*flower$|.*tower$|.*hour$", "AW_ER"),
    (r".*grace$|.*embrace$", "EY_S"),
    (r".*flame$|.*name$|.*game$|.*came$|.*ame$", "EY_M"),
    (r".*glow$|.*shadow$", "OW"),
    (r".*life$|.*strife$|.*ife$", "AY_F"),
    (r".*peace$|.*cease$|.*ease$", "IY_S"),
    (r".*sea$|.*eternity$", "IY"),
]

SYLLABLE_EXCEPTIONS = {
    "fire": 1, "hour": 1, "our": 1, "power": 2, "flower": 2, "tower": 2,
    "ocean": 2, "heaven": 2, "silent": 2, "shadow": 2, "shadows": 2,
    "whisper": 2, "whispers": 2, "eternity": 4, "memory": 3, "memories": 3,
    "radiance": 3, "infinity": 4, "ancient": 2, "golden": 2, "silver": 2,
    "endless": 2, "darkness": 2, "brightness": 2, "solitude": 3, "universe": 3,
    "horizon": 3, "starlight": 2, "moonlight": 2, "sunlight": 2, "destiny": 3,
    "harmony": 3, "symphony": 3, "courage": 2, "wisdom": 2, "serenity": 4,
    "glorious": 3, "triumphant": 3, "splendor": 2, "peaceful": 2, "kingdom": 2,
    "river": 2, "mountain": 2, "mountains": 2, "forest": 2, "forests": 2,
    "quiet": 2, "beauty": 2, "rhythm": 2, "poem": 2, "poetry": 3, "the": 1, "a": 1,
}


def guess_pos(word: str) -> str:
    """Fast rule-based POS tagger."""
    w = word.lower().strip()
    if not w:
        return "UNKNOWN"
    if w in PUNCTUATION_TOKENS:
        return "PUNCT"
    if w in {"is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "shall"}:
        return "AUX"
    if w in PRONOUNS:
        return "PRON"
    if w in PREPOSITIONS:
        return "PREP"
    if w in CONJUNCTIONS:
        return "CONJ"
    if w in {"a", "an", "the"}:
        return "DET"
    for pattern, tag in POS_SUFFIXES:
        if re.match(pattern, w):
            return tag
    return "NOUN"


def count_syllables(word: str) -> int:
    """Accurate rule-based English syllable counter with poetic exceptions."""
    w = word.lower().strip()
    if not w:
        return 0
    w_clean = re.sub(r'[^a-z]', '', w)
    if not w_clean:
        return 1
    if w_clean in SYLLABLE_EXCEPTIONS:
        return SYLLABLE_EXCEPTIONS[w_clean]
    if len(w_clean) <= 3:
        return 1
    processed = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w_clean)
    vowels = re.findall(r'[aeiouy]+', processed)
    count = len(vowels)
    return max(1, count)


def extract_rhyme_key(word: str) -> str:
    """Extract phonological rhyme key (vowel nucleus + coda consonants)."""
    w = word.lower().strip()
    w_clean = re.sub(r'[^a-z]', '', w)
    if not w_clean:
        return "UNKNOWN"
    for pattern, key in RHYME_PHONEME_PATTERNS:
        if re.match(pattern, w_clean):
            return key
    v_match = list(re.finditer(r'[aeiouy]+', w_clean))
    if v_match:
        last_v = v_match[-1]
        ending = w_clean[last_v.start():]
        return "PHON_" + ending.upper()
    return "PHON_" + w_clean[-3:].upper()


def get_stress_pattern(word: str) -> str:
    """Approximate binary meter stress pattern (0 = unstressed, 1 = stressed)."""
    sylls = count_syllables(word)
    w = word.lower().strip()
    if sylls == 1:
        return "0" if (w in STOP_WORDS or w in PRONOUNS or w in PREPOSITIONS or w in {"a", "an", "the"}) else "1"
    if sylls == 2:
        return "10" if guess_pos(w) in {"NOUN", "ADJ"} else "01"
    if sylls == 3:
        return "100"
    if sylls == 4:
        return "0100"
    return "1" * sylls


@dataclass
class WordNode:
    """A word node in the brain's vocabulary graph with phonetic and prosodic features."""
    word: str
    vector: np.ndarray  # ℝ³⁸⁴
    frequency: int = 1
    pos_tags: Set[str] = field(default_factory=set)
    vad: VAD = field(default_factory=VAD)
    syllables: int = 1
    rhyme_key: str = "UNKNOWN"
    stress_pattern: str = "1"
    sector_affinity: Dict[str, float] = field(default_factory=dict)
    next_words: Dict[str, float] = field(default_factory=dict)   # target_word -> count
    skip_1: Dict[str, float] = field(default_factory=dict)        # target_word -> count
    skip_2: Dict[str, float] = field(default_factory=dict)        # target_word -> count
    phrase_memberships: Set[str] = field(default_factory=set)    # phrase IDs

    def __post_init__(self):
        if not self.syllables or self.syllables <= 0:
            self.syllables = count_syllables(self.word)
        if self.rhyme_key == "UNKNOWN":
            self.rhyme_key = extract_rhyme_key(self.word)
        if self.stress_pattern == "1":
            self.stress_pattern = get_stress_pattern(self.word)

    @property
    def total_outgoing(self) -> float:
        return sum(self.next_words.values()) or 1.0


@dataclass
class BeamCandidate:
    """A partial sentence sequence during beam search."""
    words: List[str]
    score: float
    features_history: List[List[float]] = field(default_factory=list)
    ended: bool = False
    syllables: int = 0


class WordSelector:
    """
    12-feature word decision scoring model.
    """

    DEFAULT_WEIGHTS = np.array([
        0.45,   # 1. cos(word_vec, thought_vec) - semantic intent
        0.50,   # 2. next_word_edge_confidence - natural bigram flow
        0.05,   # 3. frequency_score - common vs rare balance
        0.20,   # 4. pos_tag_fit - grammatical compatibility
        0.30,   # 5. emotional_match - VAD tone
        0.20,   # 6. sector_match - domain appropriateness
        0.25,   # 7. skip_gram_score - 2-step lookahead coherence
        0.20,   # 8. phrase_bridge_exists - part of proof phrase
        -0.70,  # 9. repetition_penalty - penalize repeated words
        0.50,   # 10. proof_path_distance - relevance to active proof nodes
        0.30,   # 11. bigram_coherence - log transition likelihood
        0.20,   # 12. sentence_position_fit - start/end suitability
    ], dtype=np.float32)

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()

    def extract_features(
        self,
        candidate_word: str,
        candidate_node: Optional[WordNode],
        prev_word: Optional[str],
        prev_node: Optional[WordNode],
        generated_so_far: List[str],
        thought_vec: np.ndarray,
        proof_vectors: List[np.ndarray],
        context_vad: VAD,
        target_sector: str,
        position: int,
        max_position: int,
    ) -> np.ndarray:
        """Extract the 12 decision features for a candidate word."""
        feats = np.zeros(12, dtype=np.float32)
        if candidate_node is None:
            return feats

        w_vec = candidate_node.vector
        norm_w = np.linalg.norm(w_vec)
        norm_t = np.linalg.norm(thought_vec)
        if norm_w > 1e-8 and norm_t > 1e-8:
            feats[0] = float(np.dot(w_vec, thought_vec) / (norm_w * norm_t))

        if prev_node and prev_node.total_outgoing > 0:
            count = prev_node.next_words.get(candidate_word, 0.0)
            feats[1] = count / prev_node.total_outgoing

        feats[2] = min(1.0, math.log1p(candidate_node.frequency) / 10.0)

        prev_pos = guess_pos(prev_word) if prev_word else "START"
        cand_pos = guess_pos(candidate_word)
        feats[3] = self._check_pos_compatibility(prev_pos, cand_pos, position)

        c_vad = candidate_node.vad
        vad_diff = math.sqrt(
            (c_vad.valence - context_vad.valence) ** 2 +
            (c_vad.arousal - context_vad.arousal) ** 2 +
            (c_vad.dominance - context_vad.dominance) ** 2
        )
        feats[4] = max(0.0, 1.0 - (vad_diff / 2.5))

        if target_sector and candidate_node.sector_affinity:
            feats[5] = candidate_node.sector_affinity.get(target_sector, 0.2)
        else:
            feats[5] = 0.5

        if len(generated_so_far) >= 2 and prev_node:
            feats[6] = 1.0 if candidate_word in prev_node.skip_1 else 0.0

        feats[7] = 1.0 if len(candidate_node.phrase_memberships) > 0 else 0.0

        rep_count = generated_so_far.count(candidate_word)
        if candidate_word.lower() in STOP_WORDS or candidate_word in PUNCTUATION_TOKENS:
            feats[8] = 0.0 if rep_count <= 2 else float(rep_count - 2)
        else:
            feats[8] = float(rep_count) * 1.5

        if proof_vectors:
            sims = []
            for pv in proof_vectors:
                npv = np.linalg.norm(pv)
                if norm_w > 1e-8 and npv > 1e-8:
                    sims.append(float(np.dot(w_vec, pv) / (norm_w * npv)))
            feats[9] = max(sims) if sims else 0.0

        if prev_node:
            p_val = (prev_node.next_words.get(candidate_word, 0.0) + 0.01) / (prev_node.total_outgoing + 1.0)
            feats[10] = max(0.0, (math.log(p_val) + 5.0) / 5.0)

        if position == 0:
            feats[11] = 1.0 if cand_pos in {"NOUN", "PRON", "DET", "ADJ", "PREP"} else 0.2
        elif position >= 6 and (candidate_word in PUNCTUATION_TOKENS or cand_pos == "NOUN"):
            feats[11] = 1.0
        else:
            feats[11] = 0.6

        return feats

    def score(self, features: np.ndarray) -> float:
        """Compute scalar score from feature vector."""
        return float(np.dot(self.weights, features))

    def _check_pos_compatibility(self, prev_pos: str, curr_pos: str, position: int) -> float:
        """Rule-based grammatical transition score."""
        if prev_pos == "START":
            return 1.0 if curr_pos in {"DET", "NOUN", "PRON", "ADJ", "PREP"} else 0.4
        if prev_pos == "DET":
            return 1.0 if curr_pos in {"NOUN", "ADJ"} else 0.1
        if prev_pos == "ADJ":
            return 1.0 if curr_pos in {"NOUN", "ADJ", "CONJ"} else 0.2
        if prev_pos in {"NOUN", "PRON"}:
            return 1.0 if curr_pos in {"VERB_GER", "VERB_PAST", "AUX", "CONJ", "PREP", "PUNCT"} else 0.3
        if prev_pos in {"VERB_GER", "VERB_PAST", "AUX"}:
            return 1.0 if curr_pos in {"DET", "NOUN", "PRON", "ADJ", "ADV", "PREP"} else 0.2
        if prev_pos == "PREP":
            return 1.0 if curr_pos in {"DET", "NOUN", "PRON", "ADJ"} else 0.1
        if prev_pos == "CONJ":
            return 1.0 if curr_pos in {"DET", "NOUN", "PRON", "ADJ", "VERB_GER", "VERB_PAST"} else 0.3
        if prev_pos == "PUNCT":
            return 1.0 if curr_pos in {"DET", "NOUN", "PRON", "ADJ"} else 0.3
        return 0.5


class WordGraph:
    """
    Compositional Word Graph holding all word nodes, transitions, embeddings,
    and phonetic rhyming/prosodic index.
    """

    def __init__(self, config: Optional[CortexConfig] = None, emotion_engine: Optional[EmotionEngine] = None):
        self.config = config or CortexConfig()
        self.emotion_engine = emotion_engine or EmotionEngine()
        self.nodes: Dict[str, WordNode] = {}
        self.rhyme_buckets: Dict[str, Set[str]] = {}
        self.dim: int = getattr(self.config, "concept_dim", 384)
        self.selector = WordSelector()

        self._matrix_stale = True
        self._vocab_list: List[str] = []
        self._matrix: Optional[np.ndarray] = None

        self._seed_core_poetic_lexicon()

    def _seed_core_poetic_lexicon(self):
        """Pre-populate core rhyming pairs and poetic vocabulary."""
        core_rhymes = [
            ("light", "night", "bright", "sight", "flight", "might", "white"),
            ("deep", "sleep", "keep", "weep", "leap", "steep"),
            ("gold", "bold", "cold", "hold", "fold", "old", "told"),
            ("wave", "brave", "grave", "save", "cave", "gave"),
            ("fire", "desire", "aspire", "higher", "pyre"),
            ("stream", "dream", "beam", "gleam", "seem"),
            ("skies", "rise", "eyes", "wise", "cries", "lies"),
            ("rain", "pain", "gain", "remain", "chain", "plain"),
            ("heart", "art", "part", "dart", "start"),
            ("time", "climb", "rhyme", "prime"),
            ("soul", "whole", "goal", "toll", "roll"),
            ("wind", "mind", "find", "blind", "kind", "bind"),
            ("star", "far", "scar"),
            ("grace", "face", "space", "place", "trace", "embrace"),
            ("stone", "alone", "tone", "bone", "shone"),
            ("moon", "soon", "noon", "tune", "dune"),
            ("true", "blue", "through", "new", "grew", "flew"),
            ("flow", "glow", "know", "grow", "show", "shadow"),
            ("sound", "ground", "bound", "found", "round"),
            ("song", "long", "strong"),
            ("sun", "run", "one", "won", "spun"),
            ("peace", "cease", "release"),
            ("flame", "name", "game", "came"),
        ]

        for cluster in core_rhymes:
            rkey = extract_rhyme_key(cluster[0])
            for w in cluster:
                node = self.get_or_create_node(w, sector="poetry")
                if node.rhyme_key in self.rhyme_buckets and w.lower() in self.rhyme_buckets[node.rhyme_key]:
                    self.rhyme_buckets[node.rhyme_key].discard(w.lower())
                node.rhyme_key = rkey
                if rkey not in self.rhyme_buckets:
                    self.rhyme_buckets[rkey] = set()
                self.rhyme_buckets[rkey].add(w.lower())

    def get_or_create_node(
        self,
        word: str,
        vector: Optional[np.ndarray] = None,
        sector: str = "general",
        vad: Optional[VAD] = None,
    ) -> WordNode:
        """Retrieve existing word node or instantiate a new one with prosodic metadata."""
        cleaned = word.strip()
        if not cleaned:
            cleaned = " "
        key = cleaned.lower()

        if key in self.nodes:
            node = self.nodes[key]
            node.frequency += 1
            if sector:
                node.sector_affinity[sector] = node.sector_affinity.get(sector, 0.0) + 1.0
            return node

        if vector is None:
            rng = np.random.RandomState(abs(hash(key)) % (2**31 - 1))
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        else:
            vec = vector.copy().astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm

        if vad is None:
            vad = self.emotion_engine.get_word_vad(key)

        sylls = count_syllables(cleaned)
        rkey = extract_rhyme_key(cleaned)
        stress = get_stress_pattern(cleaned)

        node = WordNode(
            word=cleaned,
            vector=vec,
            frequency=1,
            pos_tags={guess_pos(cleaned)},
            vad=vad,
            syllables=sylls,
            rhyme_key=rkey,
            stress_pattern=stress,
            sector_affinity={sector: 1.0} if sector else {"general": 1.0},
        )
        self.nodes[key] = node

        if rkey not in self.rhyme_buckets:
            self.rhyme_buckets[rkey] = set()
        self.rhyme_buckets[rkey].add(key)

        self._matrix_stale = True
        return node

    def add_sentence_tokens(
        self,
        tokens: List[str],
        token_vectors: Optional[List[np.ndarray]] = None,
        sector: str = "general",
        sentence_vad: Optional[VAD] = None,
        phrase_id: Optional[str] = None,
    ):
        """
        Ingest a sequence of word tokens into the word graph:
        - Creates/updates word nodes with phonetic and prosodic features
        - Adds bigram next-word edges
        - Adds skip-1 and skip-2 edges
        - Indexes rhyme clusters
        """
        if not tokens:
            return

        nodes: List[WordNode] = []
        for i, tok in enumerate(tokens):
            vec = token_vectors[i] if token_vectors and i < len(token_vectors) else None
            node = self.get_or_create_node(tok, vector=vec, sector=sector, vad=sentence_vad)
            if phrase_id:
                node.phrase_memberships.add(phrase_id)
            nodes.append(node)

        for i in range(len(nodes) - 1):
            curr_word = nodes[i].word.lower()
            next_word = nodes[i + 1].word.lower()
            nodes[i].next_words[next_word] = nodes[i].next_words.get(next_word, 0.0) + 1.0

            if i + 2 < len(nodes):
                skip1_word = nodes[i + 2].word.lower()
                nodes[i].skip_1[skip1_word] = nodes[i].skip_1.get(skip1_word, 0.0) + 1.0

            if i + 3 < len(nodes):
                skip2_word = nodes[i + 3].word.lower()
                nodes[i].skip_2[skip2_word] = nodes[i].skip_2.get(skip2_word, 0.0) + 1.0

    def get_rhyming_words(self, word: str, target_syllables: Optional[int] = None) -> List[str]:
        """Find all words in the graph that rhyme with the given word."""
        w = word.lower().strip()
        rkey = extract_rhyme_key(w)
        candidates = list(self.rhyme_buckets.get(rkey, set()))
        candidates = [c for c in candidates if c != w]
        if target_syllables is not None:
            filtered = [c for c in candidates if count_syllables(c) == target_syllables]
            if filtered:
                return filtered
        return candidates

    def find_poetic_rhyme_pair(
        self,
        theme_vec: np.ndarray,
        mood_vad: Optional[VAD] = None,
        exclude_words: Optional[Set[str]] = None,
    ) -> Tuple[str, str]:
        """
        Find a harmonious rhyming pair (WordA, WordB) matching theme vector and mood.
        """
        exclude = exclude_words or set()
        best_pair = ("night", "light")
        best_score = -999.0

        for rkey, bucket in self.rhyme_buckets.items():
            words = [w for w in bucket if w in self.nodes and w not in exclude and count_syllables(w) <= 2]
            if len(words) < 2:
                continue

            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    n1, n2 = self.nodes[w1], self.nodes[w2]

                    sim1 = float(np.dot(n1.vector, theme_vec) / (np.linalg.norm(n1.vector) * np.linalg.norm(theme_vec) + 1e-8))
                    sim2 = float(np.dot(n2.vector, theme_vec) / (np.linalg.norm(n2.vector) * np.linalg.norm(theme_vec) + 1e-8))
                    score = sim1 + sim2

                    if mood_vad:
                        v_diff1 = math.sqrt((n1.vad.valence - mood_vad.valence)**2 + (n1.vad.arousal - mood_vad.arousal)**2)
                        v_diff2 = math.sqrt((n2.vad.valence - mood_vad.valence)**2 + (n2.vad.arousal - mood_vad.arousal)**2)
                        score += 2.0 - (v_diff1 + v_diff2)

                    if score > best_score:
                        best_score = score
                        best_pair = (w1, w2)

        return best_pair

    def _rebuild_matrix(self):
        """Rebuild matrix for fast vectorized cosine similarity."""
        if not self._matrix_stale and self._matrix is not None:
            return
        self._vocab_list = list(self.nodes.keys())
        if not self._vocab_list:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._matrix_stale = False
            return
        matrix = np.zeros((len(self._vocab_list), self.dim), dtype=np.float32)
        for i, word in enumerate(self._vocab_list):
            matrix[i] = self.nodes[word].vector
        self._matrix = matrix
        self._matrix_stale = False

    def find_nearest_words(self, query_vec: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """Find the top-K words closest to a query vector in ℝ³⁸⁴."""
        self._rebuild_matrix()
        if self._matrix is None or len(self._vocab_list) == 0:
            return []

        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-8:
            return [(w, 0.5) for w in self._vocab_list[:top_k]]

        norm_q = query_vec / q_norm
        sims = np.dot(self._matrix, norm_q)
        top_indices = np.argsort(-sims)[:top_k]

        return [(self._vocab_list[idx], float(sims[idx])) for idx in top_indices]


class WordGraphGenerator:
    """
    Dynamic Graph-Native Generative Language Engine using Compositional Word Graph.
    Generates novel, non-static sentences token-by-token from graph topology.
    """

    def __init__(self, word_graph: WordGraph, emotion_engine: Optional[EmotionEngine] = None):
        self.word_graph = word_graph
        self.emotion_engine = emotion_engine or word_graph.emotion_engine
        self.selector = word_graph.selector

    def generate(
        self,
        thought_vector: np.ndarray,
        proof_paths: Optional[List[Any]] = None,
        context_vad: Optional[VAD] = None,
        temperature: float = 0.5,
        max_tokens: int = 25,
        beam_width: int = 8,
        target_sector: str = "general",
        seed_words: Optional[List[str]] = None,
    ) -> str:
        """
        Dynamically generate fluent, novel, grounded text token-by-token.
        """
        if not self.word_graph.nodes:
            return ""

        proof_vectors: List[np.ndarray] = []
        proof_entities: List[str] = []
        proof_words: List[str] = []

        if proof_paths:
            for item in proof_paths:
                p = item.path if hasattr(item, "path") else item
                nodes = getattr(p, "nodes", [])
                for n in nodes:
                    if hasattr(n, "vector") and n.vector is not None:
                        proof_vectors.append(n.vector)
                    if hasattr(n, "source_text"):
                        proof_entities.append(str(n.source_text))
                        for w in re.findall(r"\w+", n.source_text.lower()):
                            if w in self.word_graph.nodes and w not in STOP_WORDS:
                                proof_words.append(w)

        if context_vad is None:
            context_vad = self.emotion_engine.compute_concept_vad(
                " ".join(proof_entities) if proof_entities else "response"
            )

        candidates: List[str] = []
        if seed_words:
            candidates.extend(w.lower() for w in seed_words if w.lower() in self.word_graph.nodes)

        for pw in proof_words:
            if pw not in candidates:
                candidates.append(pw)

        nearest = self.word_graph.find_nearest_words(thought_vector, top_k=8)
        for w, _ in nearest:
            if w not in candidates:
                candidates.append(w)

        if not candidates:
            candidates = list(self.word_graph.nodes.keys())[:8]

        beam: List[BeamCandidate] = []
        for cand in candidates[:beam_width * 2]:
            node = self.word_graph.nodes.get(cand)
            feats = self.selector.extract_features(
                candidate_word=cand,
                candidate_node=node,
                prev_word=None,
                prev_node=None,
                generated_so_far=[],
                thought_vec=thought_vector,
                proof_vectors=proof_vectors,
                context_vad=context_vad,
                target_sector=target_sector,
                position=0,
                max_position=max_tokens,
            )
            score = self.selector.score(feats)
            if temperature > 0:
                score += float(np.random.normal(0, temperature * 0.1))

            beam.append(BeamCandidate(words=[cand], score=score, features_history=[feats]))

        beam = sorted(beam, key=lambda b: b.score, reverse=True)[:beam_width]

        for step in range(1, max_tokens):
            all_extensions: List[BeamCandidate] = []
            all_ended = True

            for candidate in beam:
                if candidate.ended:
                    all_extensions.append(candidate)
                    continue

                all_ended = False
                prev_w = candidate.words[-1]
                prev_node = self.word_graph.nodes.get(prev_w.lower())

                if prev_w in PUNCTUATION_TOKENS or (step >= 8 and prev_w.endswith(".")):
                    candidate.ended = True
                    all_extensions.append(candidate)
                    continue

                next_options: Set[str] = set()
                if prev_node and prev_node.next_words:
                    for nw in sorted(prev_node.next_words.keys(), key=lambda k: prev_node.next_words[k], reverse=True)[:6]:
                        next_options.add(nw)
                if prev_node and prev_node.skip_1:
                    for sw in sorted(prev_node.skip_1.keys(), key=lambda k: prev_node.skip_1[k], reverse=True)[:3]:
                        next_options.add(sw)

                for pw in proof_words[:4]:
                    if pw not in candidate.words:
                        next_options.add(pw)

                if len(next_options) < 3:
                    for nw, _ in self.word_graph.find_nearest_words(thought_vector, top_k=3):
                        next_options.add(nw)

                for nw in next_options:
                    node = self.word_graph.nodes.get(nw.lower())
                    feats = self.selector.extract_features(
                        candidate_word=nw,
                        candidate_node=node,
                        prev_word=prev_w,
                        prev_node=prev_node,
                        generated_so_far=candidate.words,
                        thought_vec=thought_vector,
                        proof_vectors=proof_vectors,
                        context_vad=context_vad,
                        target_sector=target_sector,
                        position=step,
                        max_position=max_tokens,
                    )
                    step_score = self.selector.score(feats)
                    if temperature > 0:
                        step_score += float(np.random.normal(0, temperature * 0.1))

                    new_words = candidate.words + [nw]
                    new_score = candidate.score + step_score
                    is_terminal = (nw in {".", "!", "?"}) or (step >= max_tokens - 1)

                    if is_terminal and step >= 4:
                        new_score += 1.5

                    all_extensions.append(BeamCandidate(
                        words=new_words,
                        score=new_score,
                        features_history=candidate.features_history + [feats],
                        ended=is_terminal,
                    ))

            if all_ended or not all_extensions:
                break

            beam = sorted(all_extensions, key=lambda b: (b.score / (len(b.words) ** 0.5)), reverse=True)[:beam_width]

        if not beam:
            return ""

        ended_candidates = [b for b in beam if b.ended and len(b.words) >= 4]
        if ended_candidates:
            best = max(ended_candidates, key=lambda b: (b.score / (len(b.words) ** 0.5)))
        else:
            best = max(beam, key=lambda b: (b.score / (len(b.words) ** 0.5)))

        return self._detokenize(best.words)

    def _detokenize(self, words: List[str]) -> str:
        """Clean token list into natural, punctuated text."""
        if not words:
            return ""

        text = ""
        for i, w in enumerate(words):
            if not w:
                continue
            if w in {".", ",", "!", "?", ";", ":"}:
                text = text.rstrip() + w + " "
            elif w.startswith("'") or (i > 0 and words[i-1] in {"$", "@", "#"}):
                text = text.rstrip() + w + " "
            else:
                text += w + " "

        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
            if not text.endswith((".", "!", "?")):
                text += "."

        text = re.sub(r'\s+([,\.\?!;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# GraphPoetEngine — 100% Graph-Native Poetic Composition (Zero Transformers)
# ─────────────────────────────────────────────────────────────────────────────

class GraphPoetEngine:
    """
    100% Graph-Native Poetic Composition & Prose Engine (Zero Transformers).
    Uses Energy-based Boltzmann Graph Traversal with exact syllable budgeting,
    phonetic rhyme steering, and affective VAD resonance.
    """

    POETIC_OPENERS = [
        "The", "In", "Through", "Across", "When", "With", "Where",
        "Beneath", "Beyond", "Silent", "Golden", "Ancient", "Dark",
        "Bright", "Upon", "Above", "Within", "From", "Into",
    ]

    def __init__(self, word_graph: WordGraph, emotion_engine: Optional[EmotionEngine] = None):
        self.word_graph = word_graph
        self.emotion_engine = emotion_engine or word_graph.emotion_engine

    def compose_line(
        self,
        theme_vector: np.ndarray,
        target_syllables: int = 8,
        end_word: Optional[str] = None,
        context_vad: Optional[VAD] = None,
        prev_words: Optional[List[str]] = None,
        exclude_openers: Optional[Set[str]] = None,
        temperature: float = 0.6,
    ) -> str:
        """
        Compose a single poetic line satisfying target syllable budget and ending on end_word.
        """
        if context_vad is None:
            context_vad = VAD(valence=0.7, arousal=0.6, dominance=0.6)

        end_node = self.word_graph.nodes.get(end_word.lower()) if end_word else None
        end_sylls = end_node.syllables if end_node else (count_syllables(end_word) if end_word else 0)
        body_sylls_target = max(1, target_syllables - end_sylls) if end_word else target_syllables

        excluded = {e.lower() for e in exclude_openers} if exclude_openers else set()
        openers: List[str] = []
        for op in self.POETIC_OPENERS:
            if op.isalpha() and op.lower() in self.word_graph.nodes and op.lower() not in excluded and count_syllables(op) <= body_sylls_target:
                openers.append(op)

        thematic_words = [tw for tw, _ in self.word_graph.find_nearest_words(theme_vector, top_k=10) if tw.isalpha()]
        for tw in thematic_words:
            if tw not in openers and tw.lower() not in excluded and count_syllables(tw) <= body_sylls_target:
                openers.append(tw)

        if not openers:
            openers = [op for op in self.POETIC_OPENERS if op.lower() in self.word_graph.nodes] or ["The", "In", "Through"]

        beam: List[BeamCandidate] = []
        for op in openers[:12]:
            node = self.word_graph.get_or_create_node(op)
            s_count = count_syllables(op)
            norm_w = np.linalg.norm(node.vector)
            norm_t = np.linalg.norm(theme_vector)
            sim = float(np.dot(node.vector, theme_vector) / (norm_w * norm_t + 1e-8)) if norm_w > 0 and norm_t > 0 else 0.5
            beam.append(BeamCandidate(
                words=[op],
                score=sim * 2.0,
                syllables=s_count,
                ended=(s_count == body_sylls_target),
            ))

        # Iterative expansion towards body_sylls_target
        max_hops = 10
        prev_stanza_set = {pw.lower() for pw in (prev_words or []) if pw.lower() not in STOP_WORDS}
        for hop in range(max_hops):
            new_beam: List[BeamCandidate] = []
            all_done = True

            for cand in beam:
                if cand.ended or cand.syllables >= body_sylls_target:
                    new_beam.append(cand)
                    continue

                all_done = False
                rem_sylls = body_sylls_target - cand.syllables
                last_w = cand.words[-1].lower()
                last_node = self.word_graph.nodes.get(last_w)

                # Candidates from graph transitions
                next_cands: Set[str] = set()
                if last_node and last_node.next_words:
                    for nw in sorted(last_node.next_words.keys(), key=lambda k: last_node.next_words[k], reverse=True)[:8]:
                        if nw.isalpha():
                            next_cands.add(nw)
                if last_node and last_node.skip_1:
                    for sw in sorted(last_node.skip_1.keys(), key=lambda k: last_node.skip_1[k], reverse=True)[:4]:
                        if sw.isalpha():
                            next_cands.add(sw)

                for tw in thematic_words[:6]:
                    next_cands.add(tw)

                for fill in ["the", "a", "of", "in", "and", "with", "through", "pure", "deep", "soft", "calm", "wild", "bright", "where", "upon"]:
                    if fill in self.word_graph.nodes:
                        next_cands.add(fill)

                for nw in next_cands:
                    if not nw.isalpha() or nw in PUNCTUATION_TOKENS:
                        continue
                    w_sylls = count_syllables(nw)
                    if w_sylls > rem_sylls:
                        continue

                    node = self.word_graph.get_or_create_node(nw)
                    norm_w = np.linalg.norm(node.vector)
                    norm_t = np.linalg.norm(theme_vector)
                    theme_sim = float(np.dot(node.vector, theme_vector) / (norm_w * norm_t + 1e-8)) if norm_w > 0 and norm_t > 0 else 0.5

                    # Syntax score
                    prev_pos = guess_pos(cand.words[-1])
                    curr_pos = guess_pos(nw)
                    syntax_fit = self.word_graph.selector._check_pos_compatibility(prev_pos, curr_pos, len(cand.words))

                    # Transition confidence
                    edge_conf = 0.0
                    if last_node and last_node.total_outgoing > 0:
                        edge_conf = last_node.next_words.get(nw, 0.0) / last_node.total_outgoing

                    # Emotional fit
                    v_diff = math.sqrt((node.vad.valence - context_vad.valence)**2 + (node.vad.arousal - context_vad.arousal)**2)
                    vad_fit = max(0.0, 1.0 - (v_diff / 2.0))

                    # Repetition penalty
                    rep_pen = 1.5 if nw in [w.lower() for w in cand.words] else 0.0
                    if nw.lower() in prev_stanza_set:
                        rep_pen += 2.0

                    step_energy = (theme_sim * 1.8) + (syntax_fit * 1.5) + (edge_conf * 1.2) + (vad_fit * 0.8) - rep_pen
                    if temperature > 0:
                        step_energy += float(np.random.normal(0, temperature * 0.15))

                    new_sylls = cand.syllables + w_sylls
                    new_beam.append(BeamCandidate(
                        words=cand.words + [nw],
                        score=cand.score + step_energy,
                        syllables=new_sylls,
                        ended=(new_sylls == body_sylls_target),
                    ))

            if all_done or not new_beam:
                break

            beam = sorted(new_beam, key=lambda b: b.score, reverse=True)[:10]

        # Select best matching candidate
        exact_matches = [b for b in beam if b.syllables == body_sylls_target]
        best_cand = exact_matches[0] if exact_matches else beam[0]

        final_words = list(best_cand.words)
        if end_word:
            final_words.append(end_word)

        # Capitalize and format line
        line_text = " ".join(final_words).strip()
        line_text = re.sub(r'\s+([,\.\?!;:])', r'\1', line_text)
        line_text = line_text[0].upper() + line_text[1:] if line_text else ""
        return line_text

    def compose_poem(
        self,
        theme_vector: np.ndarray,
        form: str = "couplet",
        mood: str = "heroic",
        lines_count: int = 4,
        meter: str = "tetrameter",
        temperature: float = 0.6,
        context_vad: Optional[VAD] = None,
    ) -> str:
        """
        Compose a complete, metered, rhyming poem using graph energy traversal.
        """
        if context_vad is None:
            mood_lower = mood.lower()
            if "dark" in mood_lower or "sorrow" in mood_lower or "grief" in mood_lower:
                context_vad = VAD(valence=0.25, arousal=0.6, dominance=0.3)
            elif "serene" in mood_lower or "peace" in mood_lower or "calm" in mood_lower:
                context_vad = VAD(valence=0.8, arousal=0.25, dominance=0.5)
            elif "mystic" in mood_lower or "cosmic" in mood_lower or "stars" in mood_lower:
                context_vad = VAD(valence=0.65, arousal=0.7, dominance=0.7)
            else:  # heroic / epic
                context_vad = VAD(valence=0.85, arousal=0.85, dominance=0.85)

        # Syllable budgets per meter
        if meter == "pentameter":
            syllable_pattern = [10] * lines_count
        elif meter == "trimeter":
            syllable_pattern = [6] * lines_count
        elif meter == "ballad":
            syllable_pattern = [8, 6, 8, 6] * ((lines_count + 3) // 4)
        else:  # tetrameter
            syllable_pattern = [8] * lines_count

        lines: List[str] = []
        used_rhyme_words: Set[str] = set()
        used_openers: Set[str] = set()
        stanza_words: List[str] = []

        if form.lower() in {"couplet", "aabb"}:
            num_couplets = (lines_count + 1) // 2
            for c in range(num_couplets):
                w_a, w_b = self.word_graph.find_poetic_rhyme_pair(theme_vector, mood_vad=context_vad, exclude_words=used_rhyme_words)
                used_rhyme_words.add(w_a)
                used_rhyme_words.add(w_b)

                s1 = syllable_pattern[len(lines)] if len(lines) < len(syllable_pattern) else 8
                line1 = self.compose_line(theme_vector, target_syllables=s1, end_word=w_a, context_vad=context_vad, prev_words=stanza_words, exclude_openers=used_openers, temperature=temperature)
                used_openers.add(line1.split()[0].lower())
                stanza_words.extend(line1.split())
                lines.append(line1 + ",")

                if len(lines) < lines_count:
                    s2 = syllable_pattern[len(lines)] if len(lines) < len(syllable_pattern) else 8
                    line2 = self.compose_line(theme_vector, target_syllables=s2, end_word=w_b, context_vad=context_vad, prev_words=stanza_words, exclude_openers=used_openers, temperature=temperature)
                    used_openers.add(line2.split()[0].lower())
                    stanza_words.extend(line2.split())
                    lines.append(line2 + ".")

        elif form.lower() in {"quatrain", "alternating", "abab"}:
            w_a1, w_a2 = self.word_graph.find_poetic_rhyme_pair(theme_vector, mood_vad=context_vad, exclude_words=used_rhyme_words)
            used_rhyme_words.add(w_a1)
            used_rhyme_words.add(w_a2)

            w_b1, w_b2 = self.word_graph.find_poetic_rhyme_pair(theme_vector, mood_vad=context_vad, exclude_words=used_rhyme_words)
            used_rhyme_words.add(w_b1)
            used_rhyme_words.add(w_b2)

            end_words = [w_a1, w_b1, w_a2, w_b2]
            puncts = [",", ",", ",", "."]

            for i in range(min(lines_count, 4)):
                s = syllable_pattern[i] if i < len(syllable_pattern) else 8
                l_text = self.compose_line(theme_vector, target_syllables=s, end_word=end_words[i], context_vad=context_vad, prev_words=stanza_words, exclude_openers=used_openers, temperature=temperature)
                used_openers.add(l_text.split()[0].lower())
                stanza_words.extend(l_text.split())
                lines.append(l_text + puncts[i])

        elif form.lower() in {"ballad", "abcb"}:
            w_b1, w_b2 = self.word_graph.find_poetic_rhyme_pair(theme_vector, mood_vad=context_vad, exclude_words=used_rhyme_words)
            used_rhyme_words.add(w_b1)
            used_rhyme_words.add(w_b2)

            w_a = self.word_graph.find_nearest_words(theme_vector, top_k=5)[0][0]
            w_c = self.word_graph.find_nearest_words(theme_vector, top_k=5)[1][0]
            end_words = [w_a, w_b1, w_c, w_b2]
            puncts = [",", ";", ",", "."]

            for i in range(min(lines_count, 4)):
                s = syllable_pattern[i] if i < len(syllable_pattern) else 8
                l_text = self.compose_line(theme_vector, target_syllables=s, end_word=end_words[i], context_vad=context_vad, prev_words=stanza_words, exclude_openers=used_openers, temperature=temperature)
                used_openers.add(l_text.split()[0].lower())
                stanza_words.extend(l_text.split())
                lines.append(l_text + puncts[i])

        else:  # Free verse
            for i in range(lines_count):
                s = syllable_pattern[i] if i < len(syllable_pattern) else 8
                l_text = self.compose_line(theme_vector, target_syllables=s, context_vad=context_vad, prev_words=stanza_words, exclude_openers=used_openers, temperature=temperature)
                used_openers.add(l_text.split()[0].lower())
                stanza_words.extend(l_text.split())
                punct = "." if i == lines_count - 1 else ","
                lines.append(l_text + punct)

        return "\n".join(lines[:lines_count])

    def compose_prose(
        self,
        thought_vector: np.ndarray,
        style: str = "epic",
        max_words: int = 50,
        temperature: float = 0.8,
        context_vad: Optional[VAD] = None,
    ) -> str:
        """
        Generate rich, evocative creative prose using Graph-Native Boltzmann Walk.
        """
        if context_vad is None:
            if style == "epic":
                context_vad = VAD(valence=0.85, arousal=0.85, dominance=0.85)
            elif style == "lyrical":
                context_vad = VAD(valence=0.75, arousal=0.5, dominance=0.6)
            elif style == "dark":
                context_vad = VAD(valence=0.2, arousal=0.7, dominance=0.4)
            else:
                context_vad = VAD(valence=0.6, arousal=0.5, dominance=0.5)

        generator = WordGraphGenerator(self.word_graph, emotion_engine=self.emotion_engine)
        return generator.generate(
            thought_vector=thought_vector,
            context_vad=context_vad,
            temperature=temperature,
            max_tokens=max_words,
            beam_width=8,
            target_sector="story",
        )
