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
Hyper-Relational Semantic Role & Mathematical Parser
=====================================================
Transforms natural language, conditional statements, and mathematical rules
from ANY domain into rich HyperEdge representations with:
- N-ary relation predicates (Subject, Relation, Object)
- Conditional preconditions ("if renal clearance < 30 ml/min")
- Temporal validity bounds ([t_start, t_end])
- Epistemic modality (NECESSARY, POSSIBLE, NEGATED, FACTUAL)
- Quantitative / Mathematical constraints (variables, bounds, equations)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class Modality(str, Enum):
    FACTUAL = "factual"        # Asserted fact (Box true in base world)
    NECESSARY = "necessary"    # Must hold under all conditions (Box)
    POSSIBLE = "possible"      # May hold under certain conditions (Diamond)
    CONDITIONAL = "conditional"# Holds IF condition C is satisfied
    NEGATED = "negated"        # Strictly false / prohibited (Not)


@dataclass
class MathConstraint:
    """Quantitative or algebraic constraint attached to a relation."""
    variable: str
    operator: str             # "<", "<=", ">", ">=", "==", "!=", "in"
    value: Union[float, int, str, List[Union[float, int]]]
    unit: Optional[str] = None
    raw_expression: str = ""

    def evaluate(self, env: Dict[str, Any]) -> bool:
        """Evaluate constraint against environment variable assignments."""
        if self.variable not in env:
            return True  # If variable unassigned, cannot disprove
        val = env[self.variable]
        try:
            num_val = float(val)
            if self.operator == "<":
                return num_val < float(self.value)
            elif self.operator == "<=":
                return num_val <= float(self.value)
            elif self.operator == ">":
                return num_val > float(self.value)
            elif self.operator == ">=":
                return num_val >= float(self.value)
            elif self.operator == "==":
                return np.isclose(num_val, float(self.value))
            elif self.operator == "!=":
                return not np.isclose(num_val, float(self.value))
        except (ValueError, TypeError):
            if self.operator == "==":
                return str(val).lower() == str(self.value).lower()
            elif self.operator == "!=":
                return str(val).lower() != str(self.value).lower()
            elif self.operator == "in" and isinstance(self.value, list):
                return val in self.value
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "operator": self.operator,
            "value": self.value,
            "unit": self.unit,
            "raw_expression": self.raw_expression,
        }


@dataclass
class HyperEdge:
    """
    N-ary Hyper-Relational Edge representation.
    Enables multi-domain reasoning with conditions, math constraints, and modalities.
    """
    source: str
    relation: str
    target: str
    sector: str = "general"
    confidence: float = 1.0
    modality: Modality = Modality.FACTUAL
    condition: Optional[str] = None
    math_constraints: List[MathConstraint] = field(default_factory=list)
    temporal_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    provenance: str = "hyper_parser"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid_under(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if hyper-edge is valid given context variables / conditions."""
        # 1. Modality check
        if self.modality == Modality.NEGATED:
            return False, f"Edge is negated: NOT ({self.source} {self.relation} {self.target})"

        # 2. Math constraints check
        for mc in self.math_constraints:
            if not mc.evaluate(context):
                return False, f"Constraint failed: {mc.variable} {mc.operator} {mc.value}"

        # 3. Temporal validity check
        current_time = context.get("current_time")
        if current_time is not None and self.temporal_bounds is not None:
            t_start, t_end = self.temporal_bounds
            if t_start is not None and current_time < t_start:
                return False, f"Edge not yet active (current_time < {t_start})"
            if t_end is not None and current_time > t_end:
                return False, f"Edge expired (current_time > {t_end})"

        return True, "Valid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "sector": self.sector,
            "confidence": self.confidence,
            "modality": self.modality.value,
            "condition": self.condition,
            "math_constraints": [mc.to_dict() for mc in self.math_constraints],
            "temporal_bounds": self.temporal_bounds,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }


class SemanticRoleExtractor:
    """
    High-fidelity Semantic Role Labeling & Universal Clausal Parser.
    Extracts structured HyperEdges from arbitrary cross-domain text.
    """

    # Condition regex patterns
    CONDITION_PATTERNS = [
        re.compile(r'\bif\s+(.+?)(?:,\s*|\s+then\s+|\s+causes\s+|\s+leads to\s+)(.+)', re.IGNORECASE),
        re.compile(r'\bwhen\s+(.+?)(?:,\s*|\s+then\s+)(.+)', re.IGNORECASE),
        re.compile(r'\bprovided that\s+(.+?)(?:,\s*|\s+then\s+)(.+)', re.IGNORECASE),
        re.compile(r'(.+?)\s+only if\s+(.+)', re.IGNORECASE),
        re.compile(r'(.+?)\s+in the event of\s+(.+)', re.IGNORECASE),
    ]

    # Math / Quantitative comparison regex patterns
    MATH_PATTERNS = [
        re.compile(r'([a-zA-Z_]\w*)\s*(<=|>=|<|>|==|!=|=)\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z/%\^]+)?'),
        re.compile(r'(?:temperature|speed|pressure|mass|price|clearance|dosage|age|rate|voltage|current)\s*(?:is|equals|of)?\s*(<=|>=|<|>|==|!=|=)?\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z/%\^]+)?', re.IGNORECASE),
    ]

    # Negation patterns
    NEGATION_WORDS = {"not", "never", "cannot", "does not", "is not", "prohibits", "prohibited", "contraindicates", "inhibits"}

    # Common multi-domain relational patterns
    RELATION_TEMPLATES = [
        (re.compile(r'(.+?)\s+(?:is used to treat|treats|cures|heals)\s+(.+)', re.IGNORECASE), "treats"),
        (re.compile(r'(.+?)\s+(?:causes|induces|triggers|leads to|results in)\s+(.+)', re.IGNORECASE), "causes"),
        (re.compile(r'(.+?)\s+(?:contraindicates|must not be used with|is dangerous with)\s+(.+)', re.IGNORECASE), "contraindicates"),
        (re.compile(r'(.+?)\s+(?:inhibits|blocks|suppresses|reduces)\s+(.+)', re.IGNORECASE), "inhibits"),
        (re.compile(r'(.+?)\s+(?:activates|stimulates|enhances|increases)\s+(.+)', re.IGNORECASE), "activates"),
        (re.compile(r'(.+?)\s+(?:is a type of|is an instance of|is a|is an)\s+(.+)', re.IGNORECASE), "is-a"),
        (re.compile(r'(.+?)\s+(?:consists of|is composed of|contains|has part)\s+(.+)', re.IGNORECASE), "has-part"),
        (re.compile(r'(.+?)\s+(?:is equal to|equals|=|is defined as)\s+(.+)', re.IGNORECASE), "equals"),
        (re.compile(r'(.+?)\s+(?:depends on|requires|relies on)\s+(.+)', re.IGNORECASE), "depends-on"),
        (re.compile(r'(.+?)\s+(?:calculates|computes|evaluates to)\s+(.+)', re.IGNORECASE), "computes"),
        (re.compile(r'(.+?)\s+(?:violates|breaches|contravenes)\s+(.+)', re.IGNORECASE), "violates"),
        (re.compile(r'(.+?)\s+(?:protects|guards|safeguards)\s+(.+)', re.IGNORECASE), "protects"),
    ]

    def __init__(self):
        pass

    def parse_math_constraints(self, text: str) -> List[MathConstraint]:
        """Extract quantitative and algebraic inequalities from text."""
        constraints = []
        for pattern in self.MATH_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 4:
                    var, op, val, unit = groups
                    op = "==" if op == "=" else op
                    try:
                        constraints.append(MathConstraint(
                            variable=var.strip(),
                            operator=op.strip(),
                            value=float(val),
                            unit=unit.strip() if unit else None,
                            raw_expression=match.group(0),
                        ))
                    except ValueError:
                        pass
        return constraints

    def parse_sentence(self, sentence: str, sector: str = "general", provenance: str = "input") -> List[HyperEdge]:
        """Parse arbitrary cross-domain sentence into qualified HyperEdges."""
        text = sentence.strip()
        if not text:
            return []

        hyper_edges = []
        condition_clause = None
        main_clause = text

        # 1. Detect condition structures (if/when/only if)
        for pattern in self.CONDITION_PATTERNS:
            m = pattern.search(text)
            if m:
                condition_clause = m.group(1).strip()
                main_clause = m.group(2).strip()
                break

        # 2. Extract math constraints from condition and main clause
        math_constraints = self.parse_math_constraints(text)

        # 3. Detect modality
        modality = Modality.FACTUAL
        lower_text = text.lower()
        if condition_clause or math_constraints:
            modality = Modality.CONDITIONAL
        elif any(neg in lower_text for neg in ["never", "strictly prohibited", "cannot", "do not"]):
            modality = Modality.NEGATED
        elif any(nec in lower_text for nec in ["must", "always", "strictly required", "necessarily"]):
            modality = Modality.NECESSARY
        elif any(pos in lower_text for pos in ["might", "may", "can possibly", "could"]):
            modality = Modality.POSSIBLE

        # 4. Extract relational triplets from main clause
        found_relation = False
        for pattern, rel_type in self.RELATION_TEMPLATES:
            m = pattern.search(main_clause)
            if m:
                subj = self._clean_entity(m.group(1))
                obj = self._clean_entity(m.group(2))
                if subj and obj and subj.lower() != obj.lower():
                    edge = HyperEdge(
                        source=subj,
                        relation=rel_type,
                        target=obj,
                        sector=sector,
                        confidence=0.95 if modality != Modality.POSSIBLE else 0.70,
                        modality=modality,
                        condition=condition_clause,
                        math_constraints=math_constraints,
                        provenance=provenance,
                    )
                    hyper_edges.append(edge)
                    found_relation = True
                    break

        # 5. Fallback generic parsing for subject-verb-object structures
        if not found_relation:
            generic_edge = self._parse_generic_clause(main_clause, sector, modality, condition_clause, math_constraints, provenance)
            if generic_edge:
                hyper_edges.append(generic_edge)

        return hyper_edges

    def _parse_generic_clause(
        self,
        clause: str,
        sector: str,
        modality: Modality,
        condition: Optional[str],
        math_constraints: List[MathConstraint],
        provenance: str,
    ) -> Optional[HyperEdge]:
        """Generic clause parser splitting subject and predicate."""
        tokens = [t.strip(",.!? ") for t in clause.split() if t.strip(",.!? ")]
        if len(tokens) < 3:
            return None

        # Simple S-V-O split
        mid = len(tokens) // 2
        subj = " ".join(tokens[:mid])
        obj = " ".join(tokens[mid:])
        return HyperEdge(
            source=subj,
            relation="relates-to",
            target=obj,
            sector=sector,
            confidence=0.75,
            modality=modality,
            condition=condition,
            math_constraints=math_constraints,
            provenance=provenance,
        )

    def _clean_entity(self, text: str) -> str:
        """Clean leading/trailing stop words and punctuation."""
        t = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', text)
        t = re.sub(r'^(the|a|an|that|this)\s+', '', t, flags=re.IGNORECASE)
        return t.strip()
