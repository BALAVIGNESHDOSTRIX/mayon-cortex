import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

sys.path.insert(0, ".")

images_dir = Path("demos/data/images")
out_dir = Path("demos/demo_vision_output/test_fix")
out_dir.mkdir(parents=True, exist_ok=True)

cat_img = np.array(Image.open(images_dir / "cat.jpg"))
dog_img = np.array(Image.open(images_dir / "dog.jpg"))
nat_img = np.array(Image.open(images_dir / "nature.jpg"))

print("Testing full pipeline components...")
