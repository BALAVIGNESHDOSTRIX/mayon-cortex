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
Mayon-Cortex Learning — Plasticity, Local Learning & Curriculum
================================================================
Real-time Hebbian synaptic learning, Hinton Forward-Forward contrastive learning,
hierarchical predictive coding surprise minimization, curriculum teaching,
and seed bootstrapping.
"""

from mayon_cortex.learning.online_learner import OnlineLearner
from mayon_cortex.learning.forward_forward import ForwardForwardLearner, GoodnessProfile
from mayon_cortex.learning.predictive_coder import PredictiveCoder, PredictionError
from mayon_cortex.learning.teacher import Teacher
from mayon_cortex.learning.bootstrap import KnowledgeBootstrapper

__all__ = [
    "OnlineLearner",
    "ForwardForwardLearner",
    "GoodnessProfile",
    "PredictiveCoder",
    "PredictionError",
    "Teacher",
    "KnowledgeBootstrapper",
]
