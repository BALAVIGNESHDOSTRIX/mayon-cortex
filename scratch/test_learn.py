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

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph
from mayon_cortex.perception.visual_generator import GraphImageGenerator, GenerationConfig

images_dir = Path("demos/data/images")
cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

print("Learning cat...")
codebook = VisualCodebook(codebook_size=256, patch_size=(32, 32))
codebook.learn([(cat_img, "cat", {}), (dog_img, "dog", {}), (nat_img, "nature", {})])

spatial_graph = VisualSpatialGraph(num_entries=256)
for img, label in [(cat_img, "cat"), (dog_img, "dog"), (nat_img, "nature")]:
    indices, (rows, cols) = codebook.encode_image(img)
    grid = [indices[r * cols:(r + 1) * cols] for r in range(rows)]
    spatial_graph.learn_from_grid(grid)

print(f"Codebook entries: {len(codebook.entries)}")
print(f"Cat tokens: {len(codebook.concept_to_entries['cat'])}")
print(f"Dog tokens: {len(codebook.concept_to_entries['dog'])}")
print(f"Nature tokens: {len(codebook.concept_to_entries['nature'])}")
