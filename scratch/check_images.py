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

import numpy as np
from PIL import Image
from pathlib import Path

# Check input images
images_dir = Path("demos/data/images")
for name in ["cat.jpg", "dog.jpg", "nature.jpg"]:
    img = Image.open(images_dir / name)
    arr = np.array(img)
    print(f"{name}: shape {arr.shape}, dtype {arr.dtype}")
