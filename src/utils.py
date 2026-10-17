import numpy as np
from PIL import Image

def load_image_as_matrix(image_path, grayscale = True):
    
    img = Image.open(image_path)

    if grayscale:
        img = img.convert("L")

    matrix = np.array(img, dtype=np.float64)

    return matrix