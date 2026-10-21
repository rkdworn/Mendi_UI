"""MEND AI inference helpers."""
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]

def preprocess(image, size=1022, augment=False):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = size / max(image.size)
    dimensions = tuple(max(1, round(v * scale)) for v in image.size)
    image = image.resize(dimensions, Image.Resampling.LANCZOS)
    if augment:
        if random.random() < .5:
            image = ImageOps.mirror(image)
        for enhancer in random.sample([ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color], 3):
            image = enhancer(image).enhance(random.uniform(.9, 1.1))
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    pixels = torch.from_numpy(np.array(canvas, dtype=np.float32)).permute(2, 0, 1) / 255
    return (pixels - MEAN) / STD
