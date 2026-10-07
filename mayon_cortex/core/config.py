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
Cortex-Graph Configuration
===========================
All hyperparameters for the brain-like growing intelligence system.

Every component is configurable from this single file:
- NDU (Node Decision Unit) — the neuron firing rule
- Activation Wave — parallel wavefront expansion
- Path Ranker — proof chain ranking
- Language — phrase node extraction and assembly
- Learning — Hebbian plasticity and online tree updates
- Cross-Domain — multi-sector parallel reasoning
- Decision Planner — A→Z ordered action chains
- Teacher — curriculum learning phases
- Meta-Cognitive — uncertainty, self-correction, salience
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class DecisionPriority(IntEnum):
    """Universal priority hierarchy — same as real-world expert triage."""
    SAFETY           = 100   # "Will this cause harm?" — always first
    CONTRAINDICATION = 95    # "Is this explicitly forbidden?"
    LEGAL            = 90    # "Is this legal?"
    ETHICAL          = 85    # "Is this ethical?"
    CLINICAL         = 80    # "Does this work medically/technically?"
    SCIENTIFIC       = 75    # "Is there evidence?"
    FEASIBILITY      = 70    # "Can we actually do this?"
    FINANCIAL        = 60    # "Can we afford this?"
    PREFERENCE       = 50    # "What does the user/patient prefer?"
    EFFICIENCY       = 40    # "What's the fastest way?"
    MONITORING       = 30    # "How do we follow up?"
    DOCUMENTATION    = 20    # "How do we record this?"


# ── Relation type → Decision priority mapping ──
RELATION_PRIORITY_MAP: Dict[str, int] = {
    "contraindicates":  DecisionPriority.CONTRAINDICATION,
    "interacts-with":   DecisionPriority.SAFETY,
    "causes":           DecisionPriority.SAFETY,
    "prevents":         DecisionPriority.SAFETY,
    "regulated-by":     DecisionPriority.LEGAL,
    "prohibits":        DecisionPriority.LEGAL,
    "treats":           DecisionPriority.CLINICAL,
    "diagnoses":        DecisionPriority.CLINICAL,
    "proves":           DecisionPriority.SCIENTIFIC,
    "implies":          DecisionPriority.SCIENTIFIC,
    "enables":          DecisionPriority.FEASIBILITY,
    "costs":            DecisionPriority.FINANCIAL,
    "covered-by":       DecisionPriority.FINANCIAL,
    "alternative-to":   DecisionPriority.PREFERENCE,
    "monitored-by":     DecisionPriority.MONITORING,
    "requires-check":   DecisionPriority.SAFETY,
    "risk-factor-for":  DecisionPriority.SAFETY,
}

# ── Opposing relation pairs for conflict detection ──
OPPOSING_RELATIONS: Dict[str, str] = {
    "treats":    "contraindicates",
    "enables":   "prevents",
    "implies":   "contradicts",
    "safe":      "causes",
    "approved":  "prohibited",
}
CONFLICTING_RELATIONS = OPPOSING_RELATIONS


# ── Salience multipliers by relation type ──
SALIENCE_WEIGHTS: Dict[str, float] = {
    "contraindicates":  10.0,   # Life-threatening — maximum salience
    "interacts-with":   8.0,    # Drug interaction — very high
    "causes":           7.0,    # Causal harm
    "risk-factor-for":  6.0,    # Risk elevation
    "prevents":         5.0,    # Protective factor
    "prohibits":        5.0,    # Legal prohibition
    "treats":           3.0,    # Treatment relationship
    "regulated-by":     2.0,    # Regulatory
    "costs":            1.5,    # Financial
    "is-a":             1.0,    # Taxonomic (baseline)
    "part-of":          1.0,    # Structural (baseline)
    "expressed-as":     0.5,    # Language pattern (lower salience)
    "followed-by":      0.3,    # Phrase ordering (lowest)
}

# ── Domain priority for cross-domain conflict resolution ──
DOMAIN_PRIORITY: Dict[str, int] = {
    "safety":   100,
    "medical":  80,
    "legal":    90,
    "ethical":  85,
    "science":  75,
    "code":     70,
    "finance":  60,
    "math":     70,
    "general":  30,
}


@dataclass
class CortexConfig:
    """Master configuration for the Cortex-Graph brain."""

    # ── Graph ──
    concept_dim: int = 384

    # ── NDU (Node Decision Unit) ──
    ndu_n_estimators: int = 100        # number of trees in ensemble
    ndu_max_depth: int = 6             # tree depth (2^6 = 64 leaf nodes)
    ndu_num_features: int = 52         # feature vector size per node decision
    ndu_follow_threshold: float = 0.25 # fire if P(follow) > this
    ndu_retrain_interval: int = 500    # retrain trees every N queries
    ndu_learning_rate: float = 0.1     # tree learning rate
    ndu_min_samples_to_train: int = 50 # minimum samples before first training

    # ── Activation Wave ──
    wave_max_hops: int = 5             # max cascade depth
    wave_max_active: int = 200         # cap on parallel wavefront size
    wave_seed_top_k: int = 10          # FAISS seed nodes per query
    wave_halt_threshold: float = 0.8   # halt if P(halt) > this

    # ── Path Ranker ──
    ranker_n_estimators: int = 200     # XGBoost ranker ensemble size
    ranker_max_depth: int = 8          # ranker tree depth
    ranker_num_features: int = 15      # features per path
    ranker_top_k: int = 5             # return top-K proof chains
    ranker_retrain_interval: int = 300 # retrain every N queries

    # ── Language ──
    min_phrase_len: int = 2            # minimum words in a phrase node
    max_phrase_len: int = 15           # maximum words in a phrase node

    # ── Online Learning ──
    hebbian_lr: float = 0.05           # confidence boost per successful use
    decay_factor: float = 0.995        # forgetting rate per cycle
    consolidation_threshold: float = 0.7  # min confidence to add new nodes
    max_training_buffer: int = 5000    # max stored training examples

    # ── Working Memory ──
    wm_max_items: int = 10             # conversation buffer size
    wm_context_blend: float = 0.25     # blend weight for context augmentation
    wm_topic_threshold: float = 0.6    # similarity threshold for same topic

    # ── Uncertainty ──
    uncertainty_min_paths: int = 1     # minimum paths for confident answer
    uncertainty_min_confidence: float = 0.50  # below this = "I don't know"
    uncertainty_min_coverage: float = 0.55    # query coverage threshold

    # ── Self-Correction ──
    contradiction_threshold: float = 0.7  # above this = conflict detected
    max_backtrack_depth: int = 3          # max steps to backtrack

    # ── Cross-Domain ──
    cross_domain_min_sectors: int = 2  # trigger cross-domain if >= 2 sectors relevant
    sector_relevance_threshold: float = 0.3  # min similarity to consider a sector

    # ── Decision Planner ──
    max_decision_steps: int = 10       # max steps in A→Z chain
    min_step_confidence: float = 0.3   # drop steps below this confidence

    # ── Teacher ──
    teacher_pass_threshold: float = 0.8  # 80% correct = lesson passed
    teacher_max_reteach: int = 3         # max reteaching attempts
    teacher_reinforcement_boost: float = 0.2  # confidence boost on passed lessons

    # ── Word Graph (Pillar 1) ──
    word_beam_width: int = 8
    word_max_tokens: int = 40
    word_faiss_top_k: int = 20
    word_repetition_penalty: float = 1.2
    word_min_frequency: int = 1

    # ── Emotional Resonance (Pillar 3) ──
    emotion_valence_weight: float = 0.35
    emotion_arousal_weight: float = 0.20
    emotion_dominance_weight: float = 0.15

    # ── Fractal Abstraction (Pillar 2) ──
    abstraction_min_instances: int = 3
    abstraction_cluster_k: int = 4
    abstraction_trigger_interval: int = 50

    # ── Vision Cortex (Pillar 4) ──
    vision_proj_dim: int = 384
    vision_encoder: str = "mobilenet_v3_small"

    # ── Analogical Reasoning (Pillar 5) ──
    analogy_top_candidates: int = 15
    analogy_min_similarity: float = 0.35

    # ── Imagine Engine / Imagination (Pillar 7) ──
    imagination_sandbox_max_hops: int = 4
    imagination_blend_alpha: float = 0.5
    dream_replay_cycles: int = 10
    dream_prune_threshold: float = 0.15
    dream_mutation_rate: float = 0.2
    curiosity_top_k_gaps: int = 5
    curiosity_entropy_threshold: float = 0.3

    # ── System 2 Thinking & Executive Control (Phase 1) ──
    thinking_max_subgoals: int = 6
    thinking_budget_ms: int = 5000
    logic_strict_soundness: bool = True
    scratchpad_max_steps: int = 15

    # ── Cognitive Modes & Feedforward Cascade (Phase 1.5) ──
    default_cognitive_mode: str = "balanced"  # analytical, creative, factual, problem_solving, rapid
    cascade_max_depth: int = 6
    correlation_hop_limit: int = 4

    # ── Case-Based Reasoning & Judgment (Phase 7) ──
    case_memory_capacity: int = 2000
    case_match_threshold: float = 0.65
    case_top_precedents: int = 3

    # ── Hyperdimensional Computing (Phase Ω) ──
    hdc_dimension: int = 10000
    hdc_seed: int = 42

    # ── Energy Canvas & Morphogenetics (Phase Ω) ──
    energy_canvas_width: int = 64
    energy_canvas_height: int = 64
    energy_canvas_iterations: int = 100
    morphogenetic_steps: int = 500

    # ── Forward-Forward & Predictive Coding (Phase Ω) ──
    ff_goodness_threshold: float = 2.0
    predictive_learning_rate: float = 0.05

    # ── Broca Graph & N-Gram Fluency (Phase 2 & 3) ──
    broca_graph_beam_width: int = 5
    broca_graph_max_tokens: int = 120
    ngram_max_order: int = 4
    ngram_discount: float = 0.75

    # ── Scalability, Storage & Deduplication (Phase 5) ──
    storage_base_dir: str = "./cortex_storage"
    storage_snapshot_interval_sec: float = 300.0
    dedup_string_similarity_thresh: float = 0.88
    dedup_vector_similarity_thresh: float = 0.92

