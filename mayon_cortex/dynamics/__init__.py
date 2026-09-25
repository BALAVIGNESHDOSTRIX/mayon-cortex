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
Mayon-Cortex Dynamics — Activation Wavefronts & Energy Physics
==============================================================
Parallel spreading activation waves, salience decay, feedforward cascades,
Markov Random Field energy canvases, and morphogenetic reaction-diffusion.
"""

from mayon_cortex.dynamics.activation_wave import ActivationWave, ActivatedPath
from mayon_cortex.dynamics.salience import SalienceWeighter
from mayon_cortex.dynamics.feedforward_cascade import FeedforwardCascade, CascadeLayer, CascadeActivation
from mayon_cortex.dynamics.energy_canvas import EnergyCanvas, MRFConfig
from mayon_cortex.dynamics.morphogenetic import MorphogeneticGenerator, MorphogeneticParams

__all__ = [
    "ActivationWave",
    "ActivatedPath",
    "SalienceWeighter",
    "FeedforwardCascade",
    "CascadeLayer",
    "CascadeActivation",
    "EnergyCanvas",
    "MRFConfig",
    "MorphogeneticGenerator",
    "MorphogeneticParams",
]
