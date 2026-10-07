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
Sentence Templates — Semantic Relational Articulation Engine
============================================================
Phase 3 of the Brain-Like Intelligence upgrade.

Converts structured graph triples, proof chains, and reasoning nodes into
grammatically fluent, human-like sentences across multiple linguistic styles:
  - Factual / Definitional (is_a, part_of, characterized_by)
  - Causal / Explanatory (causes, leads_to, prevents, activates)
  - Comparative / Judgmental (better_than, higher_than, conflicts_with)
  - Multi-Hop Proof Chain Articulation (A → B → C)
  - Decision / Recommendation Articulation
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RelationTemplate:
    """A linguistic pattern for expressing a specific relationship between two entities."""
    relation_type: str
    templates: List[str]
    inverse_templates: List[str] = field(default_factory=list)


class SentenceTemplateEngine:
    """
    Renders structured graph triples and multi-hop paths into natural prose.
    Supports multiple discourse styles: technical, conversational, concise, formal.
    """

    TEMPLATES: Dict[str, List[str]] = {
        # Definitional & Taxonomic
        "is_a": [
            "{subject} is classified as a type of {object}.",
            "{subject} is a fundamental form of {object}.",
            "By definition, {subject} represents {object}.",
            "{subject} functions as an instance of {object}.",
        ],
        "part_of": [
            "{subject} constitutes a critical component of {object}.",
            "{subject} is an integral part of {object}.",
            "Within the structure of {object}, {subject} plays a core role.",
            "{subject} forms part of the {object} system.",
        ],
        "has_property": [
            "{subject} is characterized by {object}.",
            "{subject} exhibits key properties of {object}.",
            "A defining attribute of {subject} is {object}.",
            "{subject} demonstrates notable {object}.",
        ],
        
        # Causal & Functional
        "causes": [
            "{subject} directly leads to {object}.",
            "{subject} triggers the onset of {object}.",
            "The emergence of {subject} induces {object}.",
            "Through causal mechanisms, {subject} produces {object}.",
        ],
        "prevents": [
            "{subject} acts to prevent and mitigate {object}.",
            "{subject} inhibits the occurrence of {object}.",
            "The presence of {subject} safeguards against {object}.",
            "{subject} serves as an effective barrier to {object}.",
        ],
        "treats": [
            "{subject} is used as a therapeutic intervention for {object}.",
            "{subject} treats and alleviates {object}.",
            "Clinical protocols utilize {subject} to manage {object}.",
            "{subject} provides targeted treatment against {object}.",
        ],
        "regulates": [
            "{subject} modulates and regulates the dynamics of {object}.",
            "{subject} exerts regulatory control over {object}.",
            "The activity of {object} is finely governed by {subject}.",
        ],
        "enables": [
            "{subject} facilitates and enables {object}.",
            "{subject} provides the necessary conditions for {object}.",
            "Without {subject}, achieving {object} would not be possible.",
        ],

        # Logic & Problem Solving
        "implies": [
            "Given {subject}, it logically follows that {object}.",
            "The presence of {subject} implies {object}.",
            "From {subject}, one can deduce {object}.",
        ],
        "derived_from": [
            "{subject} is mathematically derived from {object}.",
            "{subject} originates from the foundational principle of {object}.",
            "By calculating with {object}, we obtain {subject}.",
        ],
        "conflicts_with": [
            "{subject} is in direct contradiction with {object}.",
            "There is a fundamental conflict between {subject} and {object}.",
            "{subject} mutually excludes {object}.",
        ],
        "supports": [
            "{subject} provides strong evidentiary support for {object}.",
            "Findings in {subject} corroborate {object}.",
            "{subject} reinforces the validity of {object}.",
        ],
    }

    CONNECTORS: List[str] = [
        "Furthermore,",
        "Consequently,",
        "In addition,",
        "Specifically,",
        "As a result,",
        "Moreover,",
        "Building upon this,",
        "Therefore,",
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def articulate_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        style: str = "formal",
        template_idx: Optional[int] = None,
    ) -> str:
        """Render a single (subject, relation, object) triple into a sentence."""
        rel_key = relation.lower().replace(" ", "_").replace("-", "_")
        templates = self.TEMPLATES.get(rel_key)

        if not templates:
            # Universal fallback for unknown relations
            return f"{subject.capitalize()} {relation.replace('_', ' ')} {obj}."

        if template_idx is not None and 0 <= template_idx < len(templates):
            pattern = templates[template_idx]
        else:
            pattern = random.choice(templates)

        subj_clean = subject.strip()
        obj_clean = obj.strip()

        sentence = pattern.format(subject=subj_clean, object=obj_clean)
        # Ensure capitalization and ending period
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith("."):
            sentence += "."
        return sentence

    def articulate_chain(
        self,
        chain: List[Tuple[str, str, str]],
        style: str = "formal",
    ) -> str:
        """
        Render a multi-hop proof path [(A, rel1, B), (B, rel2, C), ...] into coherent prose.
        """
        if not chain:
            return ""

        if len(chain) == 1:
            s, r, o = chain[0]
            return self.articulate_triple(s, r, o, style=style)

        sentences = []
        for i, (s, r, o) in enumerate(chain):
            sent = self.articulate_triple(s, r, o, style=style, template_idx=i % 4)
            if i > 0 and len(sentences) > 0:
                conn = self.CONNECTORS[i % len(self.CONNECTORS)]
                sent = f"{conn} {sent[0].lower() + sent[1:]}"
            sentences.append(sent)

        return " ".join(sentences)

    def articulate_decision_factor(
        self,
        factor_name: str,
        impact: float,
        evidence: str,
        recommendation: str = "",
    ) -> str:
        """Render an executive decision factor with its impact and rationale."""
        impact_label = "strongly positive" if impact > 0.5 else "positive" if impact > 0.1 else "critical risk" if impact < -0.3 else "neutral"
        text = f"Factor '{factor_name}' is assessed as {impact_label} (impact score: {impact:+.2f}). Evidence indicates that {evidence}."
        if recommendation:
            text += f" Recommended action: {recommendation}."
        return text
