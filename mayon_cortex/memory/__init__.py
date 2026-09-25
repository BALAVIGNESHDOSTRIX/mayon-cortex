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
Mayon-Cortex Memory — Associative, Episodic & HDC Memory Systems
=================================================================
Working memory buffers, case-based precedent reasoning, thought scratchpads,
and 10,000-dimensional hypervector bit algebra memory.
"""

from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.memory.case_memory import CaseMemory, PrecedentCase, CaseMatch
from mayon_cortex.memory.hyperdimensional import HyperVector, HyperdimensionalMemory
from mayon_cortex.memory.thought_scratchpad import ThoughtScratchpad, ScratchpadStep

__all__ = [
    "WorkingMemory",
    "CaseMemory",
    "PrecedentCase",
    "CaseMatch",
    "HyperVector",
    "HyperdimensionalMemory",
    "ThoughtScratchpad",
    "ScratchpadStep",
]
