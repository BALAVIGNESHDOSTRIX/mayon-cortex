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
Emotional Resonance Layer — The Brain's Limbic System
======================================================
Every node in the graph gets an emotional signature: a 3D vector
[valence, arousal, dominance] that modulates how the brain speaks.

When the brain encounters "death" in a proof path, nearby phrases
in vector space naturally have negative valence — so the generated
response is empathetic, not cheerful. This is structural emotional
intelligence, not a post-hoc filter.

The VAD (Valence-Arousal-Dominance) model:
  Valence:   [-1, +1]  negative ↔ positive (sad ↔ happy)
  Arousal:   [0, 1]    calm ↔ intense (sleepy ↔ panicked)
  Dominance: [0, 1]    submissive ↔ dominant (asking ↔ commanding)

Usage:
    emotion = EmotionEngine()
    vad = emotion.score_text("I'm so sorry for your loss")
    # → VAD(valence=-0.6, arousal=0.3, dominance=0.3)

    bias = emotion.compute_word_bias(thought_vad, candidate_word_vad)
    # → 0.85 (high match = prefer this word)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VAD:
    """Valence-Arousal-Dominance emotional signature."""
    valence: float = 0.0     # [-1, +1] negative ↔ positive
    arousal: float = 0.5     # [0, 1]   calm ↔ intense
    dominance: float = 0.5   # [0, 1]   submissive ↔ dominant

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VAD":
        return cls(valence=float(arr[0]), arousal=float(arr[1]), dominance=float(arr[2]))

    @classmethod
    def neutral(cls) -> "VAD":
        return cls(0.0, 0.5, 0.5)

    def distance(self, other: "VAD") -> float:
        """Euclidean distance between two emotional states."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def similarity(self, other: "VAD") -> float:
        """Emotional similarity [0, 1]. 1 = identical emotion."""
        dist = self.distance(other)
        return 1.0 / (1.0 + dist)


# ── Emotional Lexicon ──
# ~800 core words with hand-tuned VAD values covering real-world scenarios.
# Not medical-focused — everyday emotions, stories, human experience.
# Format: word → (valence, arousal, dominance)

EMOTIONAL_LEXICON: Dict[str, Tuple[float, float, float]] = {
    # ── Strong Positive ──
    "love": (0.9, 0.6, 0.6),
    "happy": (0.9, 0.7, 0.6),
    "joy": (0.95, 0.8, 0.6),
    "wonderful": (0.9, 0.6, 0.5),
    "beautiful": (0.85, 0.5, 0.5),
    "amazing": (0.9, 0.8, 0.5),
    "excellent": (0.85, 0.6, 0.7),
    "fantastic": (0.9, 0.8, 0.6),
    "brilliant": (0.85, 0.7, 0.7),
    "paradise": (0.95, 0.4, 0.5),
    "celebrate": (0.85, 0.8, 0.6),
    "triumph": (0.9, 0.8, 0.8),
    "victory": (0.85, 0.8, 0.8),
    "success": (0.8, 0.7, 0.7),
    "dream": (0.6, 0.4, 0.4),
    "hope": (0.7, 0.5, 0.4),
    "smile": (0.8, 0.5, 0.5),
    "laugh": (0.85, 0.7, 0.5),
    "fun": (0.8, 0.7, 0.5),
    "delight": (0.9, 0.7, 0.5),
    "excited": (0.8, 0.9, 0.6),
    "proud": (0.8, 0.6, 0.8),
    "grateful": (0.8, 0.4, 0.4),
    "thankful": (0.75, 0.4, 0.4),
    "blessed": (0.8, 0.4, 0.5),
    "kind": (0.7, 0.3, 0.5),
    "gentle": (0.6, 0.2, 0.4),
    "warm": (0.7, 0.4, 0.5),
    "bright": (0.7, 0.5, 0.5),
    "peaceful": (0.7, 0.1, 0.5),
    "calm": (0.5, 0.1, 0.5),
    "safe": (0.6, 0.2, 0.5),
    "free": (0.7, 0.5, 0.7),
    "friend": (0.7, 0.4, 0.5),
    "family": (0.7, 0.5, 0.5),
    "home": (0.7, 0.3, 0.5),
    "courage": (0.7, 0.7, 0.8),
    "brave": (0.7, 0.7, 0.8),
    "hero": (0.8, 0.7, 0.8),
    "inspire": (0.8, 0.7, 0.6),
    "magic": (0.7, 0.6, 0.5),
    "treasure": (0.7, 0.5, 0.5),
    "gift": (0.7, 0.5, 0.4),
    "adventure": (0.7, 0.8, 0.6),
    "discover": (0.7, 0.7, 0.6),
    "create": (0.6, 0.6, 0.7),
    "grow": (0.5, 0.4, 0.5),
    "learn": (0.5, 0.5, 0.5),
    "play": (0.7, 0.6, 0.5),
    "sing": (0.7, 0.6, 0.5),
    "dance": (0.7, 0.7, 0.5),

    # ── Mild Positive ──
    "good": (0.5, 0.3, 0.5),
    "nice": (0.5, 0.3, 0.5),
    "okay": (0.2, 0.2, 0.5),
    "fine": (0.3, 0.2, 0.5),
    "interesting": (0.4, 0.5, 0.5),
    "helpful": (0.5, 0.4, 0.5),
    "useful": (0.4, 0.3, 0.5),
    "correct": (0.4, 0.3, 0.6),
    "true": (0.3, 0.3, 0.6),
    "important": (0.3, 0.5, 0.6),
    "strong": (0.4, 0.6, 0.8),
    "smart": (0.5, 0.5, 0.7),
    "wise": (0.5, 0.3, 0.7),
    "patient": (0.4, 0.2, 0.5),
    "careful": (0.3, 0.3, 0.5),
    "steady": (0.3, 0.2, 0.5),

    # ── Neutral ──
    "the": (0.0, 0.0, 0.5),
    "is": (0.0, 0.0, 0.5),
    "are": (0.0, 0.0, 0.5),
    "was": (0.0, 0.0, 0.5),
    "and": (0.0, 0.0, 0.5),
    "but": (-0.1, 0.2, 0.5),
    "however": (-0.1, 0.2, 0.5),
    "because": (0.0, 0.2, 0.5),
    "therefore": (0.0, 0.2, 0.6),
    "said": (0.0, 0.3, 0.5),
    "think": (0.1, 0.4, 0.5),
    "know": (0.1, 0.3, 0.6),
    "believe": (0.2, 0.4, 0.5),
    "understand": (0.3, 0.4, 0.5),
    "remember": (0.1, 0.4, 0.4),
    "perhaps": (0.0, 0.2, 0.3),
    "maybe": (0.0, 0.2, 0.3),
    "sometimes": (0.0, 0.2, 0.4),
    "always": (0.1, 0.3, 0.6),
    "never": (-0.2, 0.3, 0.6),
    "every": (0.0, 0.2, 0.5),
    "each": (0.0, 0.2, 0.5),
    "also": (0.0, 0.1, 0.5),
    "very": (0.1, 0.3, 0.5),
    "most": (0.0, 0.2, 0.5),
    "many": (0.0, 0.2, 0.5),
    "some": (0.0, 0.1, 0.5),

    # ── Mild Negative ──
    "wrong": (-0.4, 0.4, 0.5),
    "bad": (-0.5, 0.4, 0.5),
    "difficult": (-0.3, 0.5, 0.4),
    "hard": (-0.2, 0.5, 0.5),
    "problem": (-0.4, 0.5, 0.4),
    "trouble": (-0.5, 0.5, 0.4),
    "mistake": (-0.4, 0.5, 0.4),
    "fail": (-0.6, 0.5, 0.3),
    "failure": (-0.6, 0.5, 0.3),
    "lost": (-0.5, 0.4, 0.3),
    "miss": (-0.4, 0.4, 0.3),
    "worried": (-0.4, 0.6, 0.3),
    "nervous": (-0.4, 0.7, 0.3),
    "confused": (-0.3, 0.5, 0.3),
    "doubt": (-0.3, 0.4, 0.3),
    "uncertain": (-0.2, 0.4, 0.3),
    "tired": (-0.3, 0.2, 0.3),
    "bored": (-0.3, 0.1, 0.3),
    "lonely": (-0.6, 0.3, 0.2),
    "alone": (-0.4, 0.2, 0.3),
    "cold": (-0.3, 0.2, 0.4),
    "dark": (-0.3, 0.3, 0.4),
    "empty": (-0.4, 0.2, 0.3),
    "broken": (-0.5, 0.4, 0.3),
    "hurt": (-0.6, 0.5, 0.3),
    "pain": (-0.6, 0.6, 0.3),
    "sick": (-0.5, 0.4, 0.3),
    "weak": (-0.4, 0.3, 0.2),
    "poor": (-0.4, 0.3, 0.3),
    "unfortunately": (-0.4, 0.3, 0.4),
    "sorry": (-0.3, 0.3, 0.3),
    "regret": (-0.5, 0.4, 0.3),
    "shame": (-0.6, 0.5, 0.2),
    "guilt": (-0.5, 0.5, 0.2),

    # ── Strong Negative ──
    "death": (-0.8, 0.5, 0.3),
    "die": (-0.8, 0.6, 0.3),
    "died": (-0.8, 0.5, 0.3),
    "dead": (-0.8, 0.4, 0.3),
    "kill": (-0.9, 0.8, 0.7),
    "murder": (-0.95, 0.9, 0.7),
    "war": (-0.7, 0.9, 0.6),
    "destroy": (-0.8, 0.8, 0.7),
    "hate": (-0.8, 0.8, 0.6),
    "anger": (-0.6, 0.8, 0.7),
    "angry": (-0.6, 0.8, 0.7),
    "furious": (-0.7, 0.9, 0.7),
    "rage": (-0.7, 0.95, 0.8),
    "fear": (-0.7, 0.8, 0.2),
    "afraid": (-0.7, 0.7, 0.2),
    "terrified": (-0.8, 0.9, 0.1),
    "horror": (-0.8, 0.9, 0.2),
    "terrible": (-0.7, 0.6, 0.4),
    "awful": (-0.7, 0.5, 0.4),
    "evil": (-0.9, 0.6, 0.6),
    "cruel": (-0.8, 0.6, 0.7),
    "violent": (-0.7, 0.9, 0.7),
    "suffer": (-0.7, 0.6, 0.2),
    "suffering": (-0.7, 0.6, 0.2),
    "tragedy": (-0.8, 0.6, 0.3),
    "tragic": (-0.8, 0.6, 0.3),
    "grief": (-0.8, 0.6, 0.2),
    "loss": (-0.6, 0.5, 0.3),
    "cry": (-0.5, 0.6, 0.2),
    "tears": (-0.5, 0.5, 0.2),
    "scream": (-0.5, 0.9, 0.4),
    "disaster": (-0.8, 0.8, 0.3),
    "crisis": (-0.6, 0.8, 0.4),
    "danger": (-0.6, 0.8, 0.4),
    "threat": (-0.6, 0.7, 0.5),
    "attack": (-0.7, 0.9, 0.7),
    "weapon": (-0.6, 0.7, 0.6),
    "prison": (-0.7, 0.5, 0.3),
    "punishment": (-0.5, 0.5, 0.6),
    "betray": (-0.8, 0.7, 0.4),
    "lie": (-0.5, 0.4, 0.5),
    "cheat": (-0.6, 0.5, 0.5),
    "steal": (-0.6, 0.6, 0.5),
    "abandon": (-0.7, 0.5, 0.3),

    # ── Empathetic / Compassionate ──
    "condolences": (-0.3, 0.2, 0.3),
    "sympathy": (-0.1, 0.3, 0.3),
    "compassion": (0.3, 0.3, 0.4),
    "comfort": (0.5, 0.2, 0.4),
    "support": (0.4, 0.4, 0.5),
    "care": (0.5, 0.4, 0.5),
    "hug": (0.6, 0.4, 0.4),
    "heal": (0.5, 0.4, 0.5),
    "recover": (0.5, 0.4, 0.5),
    "forgive": (0.4, 0.3, 0.5),
    "mercy": (0.3, 0.3, 0.4),

    # ── Surprise / Wonder ──
    "surprise": (0.3, 0.8, 0.4),
    "shocked": (-0.2, 0.9, 0.3),
    "astonished": (0.2, 0.8, 0.3),
    "incredible": (0.5, 0.7, 0.5),
    "unbelievable": (0.1, 0.7, 0.4),
    "miracle": (0.8, 0.7, 0.4),
    "wonder": (0.6, 0.6, 0.4),
    "mysterious": (0.2, 0.5, 0.4),
    "strange": (-0.1, 0.5, 0.4),
    "weird": (-0.2, 0.4, 0.4),

    # ── Story / Narrative Words ──
    "once": (0.1, 0.3, 0.4),
    "upon": (0.1, 0.2, 0.4),
    "story": (0.3, 0.4, 0.4),
    "tale": (0.3, 0.4, 0.4),
    "journey": (0.4, 0.6, 0.5),
    "quest": (0.4, 0.7, 0.6),
    "kingdom": (0.3, 0.4, 0.5),
    "forest": (0.2, 0.3, 0.4),
    "castle": (0.3, 0.4, 0.5),
    "dragon": (0.0, 0.8, 0.6),
    "sword": (0.0, 0.6, 0.7),
    "battle": (-0.2, 0.9, 0.6),
    "rescue": (0.6, 0.7, 0.7),
    "escape": (0.3, 0.8, 0.5),
    "return": (0.3, 0.4, 0.5),
    "beginning": (0.3, 0.4, 0.4),
    "end": (-0.1, 0.3, 0.5),
    "finally": (0.2, 0.4, 0.5),

    # ── Nature / World ──
    "sun": (0.6, 0.4, 0.5),
    "moon": (0.3, 0.2, 0.4),
    "star": (0.5, 0.3, 0.4),
    "ocean": (0.4, 0.4, 0.4),
    "mountain": (0.3, 0.4, 0.5),
    "river": (0.3, 0.3, 0.4),
    "rain": (0.0, 0.3, 0.4),
    "storm": (-0.3, 0.8, 0.5),
    "fire": (-0.2, 0.8, 0.6),
    "flower": (0.6, 0.3, 0.4),
    "tree": (0.3, 0.2, 0.4),
    "earth": (0.3, 0.3, 0.5),
    "sky": (0.4, 0.3, 0.4),
    "wind": (0.1, 0.4, 0.4),
    "snow": (0.2, 0.3, 0.4),
    "spring": (0.5, 0.4, 0.4),
    "summer": (0.5, 0.5, 0.5),
    "winter": (-0.1, 0.3, 0.4),
    "night": (-0.1, 0.3, 0.4),
    "morning": (0.4, 0.4, 0.4),
    "light": (0.5, 0.4, 0.5),
    "shadow": (-0.2, 0.3, 0.4),
}


class EmotionEngine:
    """
    Emotional resonance system for the Cortex-Graph brain.

    Assigns emotional signatures (VAD) to words and phrases,
    and modulates word selection during generation.
    """

    def __init__(self):
        self.lexicon = dict(EMOTIONAL_LEXICON)

    def score_word(self, word: str) -> VAD:
        """Get the emotional signature of a single word."""
        key = word.lower().strip(".,!?;:'\"()-")
        if key in self.lexicon:
            v, a, d = self.lexicon[key]
            return VAD(valence=v, arousal=a, dominance=d)
        return VAD.neutral()

    def get_word_vad(self, word: str) -> VAD:
        """Alias for score_word."""
        return self.score_word(word)

    def compute_concept_vad(self, text: str) -> VAD:
        """Alias for score_text."""
        return self.score_text(text)

    def score_text(self, text: str) -> VAD:
        """
        Score a text's overall emotional tone.
        Weighted average of word emotions, with emotional words
        weighted more heavily (neutral words get low weight).
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return VAD.neutral()

        total_v, total_a, total_d = 0.0, 0.0, 0.0
        total_weight = 0.0

        for word in words:
            vad = self.score_word(word)
            # Weight by emotional intensity (distance from neutral)
            intensity = abs(vad.valence) + abs(vad.arousal - 0.5) + abs(vad.dominance - 0.5)
            weight = max(0.1, intensity)  # Minimum weight so neutral words still count

            total_v += vad.valence * weight
            total_a += vad.arousal * weight
            total_d += vad.dominance * weight
            total_weight += weight

        if total_weight > 0:
            return VAD(
                valence=np.clip(total_v / total_weight, -1.0, 1.0),
                arousal=np.clip(total_a / total_weight, 0.0, 1.0),
                dominance=np.clip(total_d / total_weight, 0.0, 1.0),
            )
        return VAD.neutral()

    def compute_word_bias(self, context_vad: VAD, candidate_vad: VAD) -> float:
        """
        Compute how well a candidate word's emotion matches the context.

        Returns a score [0, 1] where:
          1.0 = perfect emotional match
          0.0 = completely wrong emotion for this context

        Used by the Word Graph generator to bias word selection.
        """
        return candidate_vad.similarity(context_vad)

    def get_emotional_direction(self, vad: VAD) -> str:
        """Get a human-readable label for an emotional state."""
        if vad.valence > 0.5 and vad.arousal > 0.6:
            return "excited-positive"
        elif vad.valence > 0.5 and vad.arousal <= 0.6:
            return "calm-positive"
        elif vad.valence > 0.1:
            return "mildly-positive"
        elif vad.valence > -0.1:
            return "neutral"
        elif vad.valence > -0.5 and vad.arousal <= 0.5:
            return "calm-negative"
        elif vad.valence > -0.5 and vad.arousal > 0.5:
            return "anxious"
        elif vad.valence <= -0.5 and vad.arousal > 0.6:
            return "distressed"
        else:
            return "somber"

    def register_word(self, word: str, valence: float, arousal: float, dominance: float):
        """Register a new word in the emotional lexicon (for online learning)."""
        self.lexicon[word.lower()] = (
            np.clip(valence, -1.0, 1.0),
            np.clip(arousal, 0.0, 1.0),
            np.clip(dominance, 0.0, 1.0),
        )

    def propagate_emotion(
        self, source_vad: VAD, edge_type: str, decay: float = 0.5
    ) -> VAD:
        """
        Propagate emotion through a graph edge.
        Certain edge types amplify or invert the emotion.
        """
        v, a, d = source_vad.valence, source_vad.arousal, source_vad.dominance

        # Causal edges amplify the emotion
        if edge_type in ("causes", "leads-to", "results-in"):
            v *= 1.2
            a *= 1.1
        # Preventive edges invert somewhat
        elif edge_type in ("prevents", "stops", "cures"):
            v *= -0.5
            a *= 0.8
        # Contrast edges invert
        elif edge_type in ("contraindicates", "contradicts", "opposes"):
            v *= -0.8
        # Default: decay
        else:
            v *= decay
            a *= decay

        return VAD(
            valence=np.clip(v, -1.0, 1.0),
            arousal=np.clip(a, 0.0, 1.0),
            dominance=np.clip(d, 0.0, 1.0),
        )

    @property
    def lexicon_size(self) -> int:
        return len(self.lexicon)
