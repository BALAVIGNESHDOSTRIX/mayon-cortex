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

import numpy as np
from PIL import Image
from pathlib import Path

# Check input images
images_dir = Path("demos/data/images")
for name in ["cat.jpg", "dog.jpg", "nature.jpg"]:
    img = Image.open(images_dir / name)
    arr = np.array(img)
    print(f"{name}: shape {arr.shape}, dtype {arr.dtype}")
