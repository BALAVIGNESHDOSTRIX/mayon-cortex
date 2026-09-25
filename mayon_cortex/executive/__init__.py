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
Mayon-Cortex Executive — Prefrontal Control & Metacognition
============================================================
Complexity classification, strategy planning, self-monitoring,
epistemic uncertainty gating, contradiction self-correction,
decision planning, and cognitive mode switching.
"""

from mayon_cortex.executive.executive import (
    ExecutiveController,
    ExecutivePlan,
    Complexity,
    ThinkingStrategy,
)
from mayon_cortex.executive.metacognition import MetacognitionMonitor, EpistemicState
from mayon_cortex.executive.uncertainty import UncertaintyGate
from mayon_cortex.executive.self_correction import SelfCorrector
from mayon_cortex.executive.decision_planner import DecisionPlanner, DecisionChain
from mayon_cortex.executive.cognitive_modes import CognitiveModeEngine, CognitiveMode, CognitiveParameters

__all__ = [
    "ExecutiveController",
    "ExecutivePlan",
    "Complexity",
    "ThinkingStrategy",
    "MetacognitionMonitor",
    "EpistemicState",
    "UncertaintyGate",
    "SelfCorrector",
    "DecisionPlanner",
    "DecisionChain",
    "CognitiveModeEngine",
    "CognitiveMode",
    "CognitiveParameters",
]
