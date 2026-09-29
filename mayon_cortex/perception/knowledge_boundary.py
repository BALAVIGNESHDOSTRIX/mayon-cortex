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
Knowledge Boundary System — Honest Visual Capability & Refusal Engine
======================================================================
Tracks visual concept knowledge boundaries to provide honest meta-cognitive
assessment of whether Mayon-Cortex can faithfully generate an image from a
given prompt, refusing hallucination when concepts are unknown.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "with", "by",
    "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "and", "but", "if", "or", "because", "as", "until", "while",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "it", "its",
    "sitting", "standing", "lying", "looking", "photo", "image", "picture",
    "realistic", "photorealistic", "rendered", "drawing", "illustration",
}


@dataclass
class ConceptKnowledge:
    """Tracks what the system knows about a visual concept."""
    label: str
    images_learned: int
    total_patches: int
    avg_feature_vector: np.ndarray        # Mean 256-dim concept embedding
    feature_variance: float               # How consistent the learning is
    codebook_coverage: float              # % of codebook entries associated
    last_learned_at: datetime
    confidence: float                     # 0.0-1.0 overall confidence


@dataclass
class GenerationCapabilityReport:
    """Assessment of whether the system can generate a given prompt."""
    can_generate: bool
    confidence: float                     # 0.0-1.0
    known_concepts: List[str]             # Concepts in prompt that ARE known
    unknown_concepts: List[str]           # Concepts in prompt that are NOT known
    closest_known: List[Tuple[str, float]] # Nearest known concepts with similarity
    reason: str                           # Human-readable explanation
    suggestion: str                       # What the system CAN do instead


class KnowledgeBoundary:
    """
    Tracks visual concept knowledge boundaries and arbitrates generation capabilities.
    """

    def __init__(self, similarity_threshold: float = 0.40):
        self.similarity_threshold = similarity_threshold
        self.concepts: Dict[str, ConceptKnowledge] = {}

    def register_concept(
        self,
        label: str,
        images_count: int,
        patches_count: int,
        avg_vector: np.ndarray,
        variance: float = 0.05,
        codebook_coverage: float = 0.10,
    ) -> ConceptKnowledge:
        """Register or update a learned concept in the knowledge boundary registry."""
        norm_label = label.lower().strip()
        vec = avg_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            vec = vec / norm

        # Calculate capability confidence
        # Calibrated on: images learned (40%), patch coverage (30%), codebook coverage (30%)
        img_score = min(1.0, images_count / 3.0)
        patch_score = min(1.0, patches_count / 40.0)
        cov_score = min(1.0, codebook_coverage / 0.15)
        var_penalty = max(0.0, min(0.2, variance * 0.5))

        conf = float(np.clip(
            (0.40 * img_score + 0.30 * patch_score + 0.30 * cov_score) - var_penalty,
            0.10,
            1.0,
        ))

        ck = ConceptKnowledge(
            label=norm_label,
            images_learned=images_count,
            total_patches=patches_count,
            avg_feature_vector=vec,
            feature_variance=variance,
            codebook_coverage=codebook_coverage,
            last_learned_at=datetime.now(),
            confidence=conf,
        )
        self.concepts[norm_label] = ck
        return ck

    def extract_prompt_tokens(self, prompt: str) -> List[str]:
        """Extract meaningful semantic concept tokens from text prompt."""
        clean = re.sub(r"[^\w\s-]", "", prompt.lower())
        tokens = [w.strip() for w in clean.split() if len(w.strip()) > 1]
        meaningful = [t for t in tokens if t not in STOP_WORDS]
        return meaningful if meaningful else tokens

    def get_known_concepts(self) -> List[ConceptKnowledge]:
        """Return list of all registered known visual concepts."""
        return list(self.concepts.values())

    def get_concept_confidence(self, label: str) -> float:
        """Return confidence score for a concept label (0.0 if unlearned)."""
        norm = label.lower().strip()
        ck = self.concepts.get(norm)
        return ck.confidence if ck else 0.0

    def can_generate(self, prompt: str) -> bool:
        """Quick boolean capability check."""
        report = self.assess_prompt(prompt)
        return report.can_generate

    def assess_prompt(
        self,
        prompt: str,
        codebook: Optional[Any] = None,
        spatial_graph: Optional[Any] = None,
    ) -> GenerationCapabilityReport:
        """
        Evaluate whether the cortex has sufficient visual knowledge to generate
        the image described by the prompt.
        """
        # Sync with codebook if provided and not yet registered
        if codebook is not None and hasattr(codebook, "concept_embeddings"):
            for lbl, emb in codebook.concept_embeddings.items():
                if lbl not in self.concepts:
                    img_cnt = len([s for s in getattr(codebook, "history_samples", []) if s[1].lower().strip() == lbl])
                    patch_cnt = sum(codebook.concept_to_entries.get(lbl, {}).values()) if hasattr(codebook, "concept_to_entries") else 16
                    self.register_concept(
                        label=lbl,
                        images_count=max(1, img_cnt),
                        patches_count=max(16, patch_cnt),
                        avg_vector=emb,
                        variance=codebook.concept_variances.get(lbl, 0.05) if hasattr(codebook, "concept_variances") else 0.05,
                        codebook_coverage=len(codebook.concept_to_entries.get(lbl, {})) / max(1, len(codebook.entries)) if hasattr(codebook, "concept_to_entries") else 0.1,
                    )

        if not self.concepts:
            return GenerationCapabilityReport(
                can_generate=False,
                confidence=0.0,
                known_concepts=[],
                unknown_concepts=[prompt],
                closest_known=[],
                reason="I haven't learned any visual concepts yet. No images have been trained into my codebook.",
                suggestion="Please teach me some concepts using `learn_image()` before asking to generate.",
            )

        prompt_tokens = self.extract_prompt_tokens(prompt)
        known_found: List[str] = []
        unknown_found: List[str] = []

        # 1. Check direct matches against known concept labels
        for token in prompt_tokens:
            matched = False
            for c_lbl in self.concepts:
                if token == c_lbl or token in c_lbl or c_lbl in token:
                    if c_lbl not in known_found:
                        known_found.append(c_lbl)
                    matched = True
                    break
            if not matched:
                unknown_found.append(token)

        # Also check whole-phrase / substring matches for multi-word concepts
        p_lower = prompt.lower()
        for c_lbl in self.concepts:
            if c_lbl in p_lower and c_lbl not in known_found:
                known_found.append(c_lbl)
                unknown_found = [u for u in unknown_found if u not in c_lbl]

        # 2. Compute similarity with nearest known concepts
        closest: List[Tuple[str, float]] = []
        for c_lbl, ck in self.concepts.items():
            # Keyword / character-ngram similarity as pure mathematical proxy
            # when neural word embeddings are intentionally not present
            sim = 0.0
            if c_lbl in known_found:
                sim = 1.0
            else:
                # N-gram overlap similarity
                set_c = set(c_lbl[i:i+3] for i in range(max(1, len(c_lbl) - 2)))
                sims_t = []
                for tok in unknown_found:
                    set_t = set(tok[i:i+3] for i in range(max(1, len(tok) - 2)))
                    inter = len(set_c & set_t)
                    union = len(set_c | set_t) or 1
                    sims_t.append(inter / union)
                sim = max(sims_t) if sims_t else 0.0
            closest.append((c_lbl, float(sim)))

        closest.sort(key=lambda x: x[1], reverse=True)

        # 3. Decision arbitration
        all_known_labels = list(self.concepts.keys())

        if known_found and not unknown_found:
            # Fully known concept
            conf = float(np.mean([self.concepts[c].confidence for c in known_found]))
            return GenerationCapabilityReport(
                can_generate=True,
                confidence=conf,
                known_concepts=known_found,
                unknown_concepts=[],
                closest_known=closest[:3],
                reason=f"Generating '{prompt}'... (confidence: {conf:.2f})",
                suggestion=f"Ready to synthesize {', '.join(known_found)}.",
            )

        elif known_found and unknown_found:
            # Partial match
            conf = float(np.mean([self.concepts[c].confidence for c in known_found])) * 0.75
            known_str = ", ".join(known_found)
            unknown_str = ", ".join(unknown_found[:2])
            return GenerationCapabilityReport(
                can_generate=True,
                confidence=conf,
                known_concepts=known_found,
                unknown_concepts=unknown_found,
                closest_known=closest[:3],
                reason=(
                    f"I partially know this prompt. I can generate the '{known_str}' part, "
                    f"but I haven't learned '{unknown_str}' yet."
                ),
                suggestion=f"I can generate just the {known_str}, or you can train me on '{unknown_str}'.",
            )

        else:
            # Unknown concept — honest refusal
            best_sim = closest[0][1] if closest else 0.0
            closest_name = closest[0][0] if closest else "none"
            known_list_str = ", ".join(all_known_labels[:5])
            primary_unknown = unknown_found[0] if unknown_found else prompt

            return GenerationCapabilityReport(
                can_generate=False,
                confidence=0.0,
                known_concepts=[],
                unknown_concepts=unknown_found,
                closest_known=closest[:3],
                reason=(
                    f"I haven't learned what a '{primary_unknown}' looks like yet. "
                    f"I can generate: {known_list_str}. "
                    f"Closest concept I know: '{closest_name}' ({best_sim:.2f} similarity)."
                ),
                suggestion=(
                    f"I can generate '{closest_name}' instead, or you can teach me '{primary_unknown}' "
                    f"using `brain.learn_image(path, label='{primary_unknown}')`."
                ),
            )
