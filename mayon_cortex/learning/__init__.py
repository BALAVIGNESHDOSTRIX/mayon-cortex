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
