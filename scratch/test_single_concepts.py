import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from mayon_cortex.perception.visual_features import VisualFeatureExtractor
from mayon_cortex.perception.visual_codebook import VisualCodebook
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph

images_dir = Path("demos/data/images")
out_dir = Path("demos/demo_vision_output/test_fix")

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

print("Testing codebook learning and decoding with feathered stitching...")
