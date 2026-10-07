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
Mayon Graph Configuration
=========================
Configuration dataclasses and constants for the Mayon Knowledge Graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ── Supported Knowledge Sectors ──
SECTORS = ["general", "code", "medical", "legal", "finance", "science", "math"]

# ── 38 Universal Relational Primitives across all Sectors ──
DEFAULT_RELATIONS = [
    # 1. Core Ontology & Taxonomy
    "is-a", "part-of", "has-property", "same-as", "differs-from",
    
    # 2. Causality, Science & Physical Mechanics
    "causes", "prevents", "enables", "derived-from", "transforms-into",
    
    # 3. Spatial, Geography & Origins
    "located-in", "adjacent-to",
    
    # 4. Temporal & Chronology
    "precedes", "created-by",
    
    # 5. Code, Systems & Engineering
    "uses", "returns", "calls", "inherits", "implements",
    
    # 6. Medical, Pharmacology & Life Sciences
    "diagnoses", "treats", "contraindicates", "manifests-as", "synthesizes",
    "interacts-with",   # Drug-Drug interaction (symmetric)
    "metabolized-by",   # Drug → CYP450 enzyme responsible for metabolism
    "inhibits",         # Drug → Enzyme it inhibits (CYP450 inhibition → elevated co-drug levels)
    "induces",          # Drug → Enzyme it induces (CYP450 induction → reduced co-drug levels)
    "monitored-by",     # Drug/Condition → Lab parameter used for monitoring
    "has-severity",     # Interaction → Severity level (minor/moderate/major/contraindicated)
    
    # 7. Legal, Governance & Economics
    "regulates", "cites", "prohibits", "funds",
    
    # 8. Formal Logic, Mathematics & Episodic Reasoning
    "implies", "proves", "contradicts", "answered-by",

    # 9. Language Expression & Fluent Composition (Cortex)
    "expressed-as", "followed-by", "synonym-of", "antonym-of",

    # 10. Cross-Domain Bridging & Value Reasoning
    "costs", "regulated-by", "liability-if-untreated", "complies-with",
]

# ── Interaction Severity Levels ──
INTERACTION_SEVERITY = {
    "minor":            1,  # Monitor — may require dose adjustment
    "moderate":         2,  # Use with caution — monitoring required
    "major":            3,  # Avoid combination — serious risk
    "contraindicated":  4,  # Absolute contraindication — never combine
}


@dataclass
class MayonConfig:
    """Configuration for the Mayon (Scalable Multi-Domain Knowledge Graph)."""

    concept_dim: int = 384
    num_levels: int = 4

    max_nodes: int = 1_000_000
    default_sector: str = "general"
    sectors: List[str] = field(default_factory=lambda: list(SECTORS))

    # ── Retrieval ──
    ann_index_type: str = "flat"   # "flat" or "hnsw"
    retrieval_top_k: int = 12
    confidence_threshold: float = 0.3
    nprobe: int = 16

    # ── Temporal & Memory Consolidation ──
    enable_temporal: bool = True
    decay_factor: float = 0.99
    min_confidence: float = 0.05
    dedup_similarity_threshold: float = 0.92

    # ── Conflict Resolution ──
    conflict_strategy: str = "confidence_weighted"
