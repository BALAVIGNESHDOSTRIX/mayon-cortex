import numpy as np
from PIL import Image
from pathlib import Path

# Check input images
images_dir = Path("demos/data/images")
for name in ["cat.jpg", "dog.jpg", "nature.jpg"]:
    img = Image.open(images_dir / name)
    arr = np.array(img)
    print(f"{name}: shape {arr.shape}, dtype {arr.dtype}")
