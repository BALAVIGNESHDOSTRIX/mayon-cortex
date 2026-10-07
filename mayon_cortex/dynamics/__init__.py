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
