"""Resize og synkron augmentation af billede, target og FOV-maske."""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter


def resize_image(image, size):
    """PIL RGB-billede -> float-tensor 3 x size x size i [0, 1] (bilineær)."""
    return TF.to_tensor(image.resize((size, size), Image.BILINEAR))


def resize_mask(mask, size):
    """Bool-array -> bool-tensor size x size (nearest neighbour, så masken forbliver binær)."""
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    return torch.from_numpy(np.array(resized) > 127)


class PairedAugment:
    """Geometriske transformationer anvendes ens på billede, target og FOV; farve kun på billedet.

    Der bruges kun crops, flips og 90-graders rotationer. De kræver ingen interpolation, så
    masker forbliver binære, og enkelte klik-pixels i sparse targets (del 2) går ikke tabt.
    """

    def __init__(self, crop_size=None):
        self.crop_size = crop_size
        self.color = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __call__(self, image, target, fov):
        if self.crop_size:
            h, w = target.shape
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            rows, cols = slice(top, top + self.crop_size), slice(left, left + self.crop_size)
            image, target, fov = image[:, rows, cols], target[rows, cols], fov[rows, cols]
        if random.random() < 0.5:
            image, target, fov = image.flip(-1), target.flip(-1), fov.flip(-1)
        if random.random() < 0.5:
            image, target, fov = image.flip(-2), target.flip(-2), fov.flip(-2)
        k = random.randint(0, 3)
        image, target, fov = (torch.rot90(x, k, (-2, -1)) for x in (image, target, fov))
        return self.color(image), target, fov
