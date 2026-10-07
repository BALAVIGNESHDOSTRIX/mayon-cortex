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
