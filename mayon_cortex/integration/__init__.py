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
Mayon-Cortex Integration Layer
==============================
Provides high-level adapters, bridges, and MCP (Model Context Protocol)
servers to interface Mayon-Cortex neuro-symbolic reasoning with Traditional LLMs,
Agentic frameworks, and AI Copilots.

The 5 Priority Integration Tools:
  1. cortex_think    - Multi-hop activation wave & path-ranked reasoning context
  2. cortex_recall   - Episodic precedent retrieval (CaseMemory) & working memory
  3. cortex_decide   - Structured decision planner & cross-domain conflict resolver
  4. cortex_validate - Post-generation hallucination & contradiction validation gate
  5. cortex_plan     - Strategic query complexity classifier & task decomposition
"""

from mayon_cortex.integration.cortex_llm_bridge import (
    CortexLLMBridge,
    cortex_think,
    cortex_recall,
    cortex_remember,
    cortex_decide,
    cortex_validate,
    cortex_plan,
)

__all__ = [
    "CortexLLMBridge",
    "cortex_think",
    "cortex_recall",
    "cortex_remember",
    "cortex_decide",
    "cortex_validate",
    "cortex_plan",
]
