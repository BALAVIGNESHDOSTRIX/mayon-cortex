import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

sys.path.insert(0, ".")

from mayon_cortex.perception.visual_features import VisualFeatureExtractor
from mayon_cortex.perception.visual_codebook import VisualCodebook, CodebookEntry
from mayon_cortex.perception.visual_spatial_graph import VisualSpatialGraph

images_dir = Path("demos/data/images")
output_dir = Path("demos/demo_vision_output/test_fix")
output_dir.mkdir(parents=True, exist_ok=True)

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

print("Loaded images successfully.")
