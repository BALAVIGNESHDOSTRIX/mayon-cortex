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
Mayon-Cortex Memory — Associative, Episodic & HDC Memory Systems
=================================================================
Working memory buffers, case-based precedent reasoning, thought scratchpads,
10,000-dimensional hypervector bit algebra memory, and persistent error memory.
"""

from mayon_cortex.memory.working_memory import WorkingMemory
from mayon_cortex.memory.case_memory import CaseMemory, PrecedentCase, CaseMatch
from mayon_cortex.memory.hyperdimensional import HyperVector, HyperdimensionalMemory
from mayon_cortex.memory.thought_scratchpad import ThoughtScratchpad, ScratchpadStep
from mayon_cortex.memory.error_memory import ErrorMemory, ErrorRecord

__all__ = [
    "WorkingMemory",
    "CaseMemory",
    "PrecedentCase",
    "CaseMatch",
    "HyperVector",
    "HyperdimensionalMemory",
    "ThoughtScratchpad",
    "ScratchpadStep",
    "ErrorMemory",
    "ErrorRecord",
]
