from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def _autocontrast(img: Image.Image, _severity: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _severity: float) -> Image.Image:
    return ImageOps.equalize(img)


def _posterize(img: Image.Image, severity: float) -> Image.Image:
    # bits in [4, 8]; lower severity → fewer bits
    bits = max(1, int(8 - severity * 4))
    return ImageOps.posterize(img, bits)


def _solarize(img: Image.Image, severity: float) -> Image.Image:
    threshold = int(256 * (1.0 - severity * 0.5))
    return ImageOps.solarize(img, threshold)


def _rotate(img: Image.Image, severity: float) -> Image.Image:
    degrees = severity * 30.0 * random.choice([-1, 1])
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _shear_x(img: Image.Image, severity: float) -> Image.Image:
    shear = severity * 0.3 * random.choice([-1, 1])
    w, h = img.size
    matrix = (1, shear, -shear * h / 2, 0, 1, 0)
    return img.transform(img.size, Image.AFFINE, matrix,
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _shear_y(img: Image.Image, severity: float) -> Image.Image:
    shear = severity * 0.3 * random.choice([-1, 1])
    w, h = img.size
    matrix = (1, 0, 0, shear, 1, -shear * w / 2)
    return img.transform(img.size, Image.AFFINE, matrix,
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _translate_x(img: Image.Image, severity: float) -> Image.Image:
    w, _ = img.size
    tx = int(severity * w * 0.33 * random.choice([-1, 1]))
    matrix = (1, 0, tx, 0, 1, 0)
    return img.transform(img.size, Image.AFFINE, matrix,
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _translate_y(img: Image.Image, severity: float) -> Image.Image:
    _, h = img.size
    ty = int(severity * h * 0.33 * random.choice([-1, 1]))
    matrix = (1, 0, 0, 0, 1, ty)
    return img.transform(img.size, Image.AFFINE, matrix,
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _brightness(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + severity * 1.8 * random.choice([-1, 1])
    factor = max(0.1, factor)
    return ImageEnhance.Brightness(img).enhance(factor)


def _color(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + severity * 1.8 * random.choice([-1, 1])
    factor = max(0.1, factor)
    return ImageEnhance.Color(img).enhance(factor)


def _contrast(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + severity * 1.8 * random.choice([-1, 1])
    factor = max(0.1, factor)
    return ImageEnhance.Contrast(img).enhance(factor)


def _sharpness(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + severity * 1.8 * random.choice([-1, 1])
    factor = max(0.1, factor)
    return ImageEnhance.Sharpness(img).enhance(factor)


# Pool of all augmentation ops used during mixing.
_AUGMENT_OPS: list[Callable[[Image.Image, float], Image.Image]] = [
    _autocontrast,
    _equalize,
    _posterize,
    _solarize,
    _rotate,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
    _brightness,
    _color,
    _contrast,
    _sharpness,
]


######################
# AugMix transform
######################

class AugMix:
    """AugMix data augmentation as a PIL-compatible transform.

    Applies ``width`` independent augmentation chains in parallel, then
    mixes the resulting images using Dirichlet-sampled coefficients.
    Finally the mixture is blended with the original image using a
    Beta-sampled coefficient.

    Designed to be inserted into a ``torchvision.transforms.Compose``
    pipeline before ``ToTensor()``, so both the input and output are
    ``PIL.Image`` objects.

    Args:
        severity: Maximum augmentation severity in ``[1, 10]``.  Controls
            how strongly each operation distorts the image.
        width: Number of parallel augmentation chains (``k`` in the paper).
        depth: Number of ops per chain.  ``-1`` samples uniformly from
            ``[1, 3]`` for each chain independently.
        alpha: Concentration parameter for both the Dirichlet (chain
            weights) and Beta (original vs. mixture weight) distributions.
    """

    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
    ) -> None:
        if not (1 <= severity <= 10):
            raise ValueError(f"severity must be in [1, 10], got {severity}.")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}.")

        self.severity = severity / 10.0   # normalise to [0.1, 1.0]
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def _apply_chain(self, img: Image.Image) -> Image.Image:
        """Apply a single augmentation chain of random depth to *img*.

        Args:
            img: Input PIL image.

        Returns:
            Augmented PIL image.
        """
        depth = self.depth if self.depth > 0 else random.randint(1, 3)
        for op in random.choices(_AUGMENT_OPS, k=depth):
            img = op(img, self.severity)
        return img

    @staticmethod
    def _to_float_array(img: Image.Image) -> np.ndarray:
        """Convert a PIL image to a float32 numpy array in ``[0, 1]``."""
        return np.array(img, dtype=np.float32) / 255.0

    @staticmethod
    def _to_pil(arr: np.ndarray) -> Image.Image:
        """Convert a float32 numpy array in ``[0, 1]`` back to a PIL image."""
        return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply AugMix to *img*.

        Args:
            img: Input PIL image (RGB or L).

        Returns:
            AugMix-augmented PIL image of the same size and mode.
        """
        original = self._to_float_array(img)

        # Sample mixing weights from a symmetric Dirichlet distribution.
        weights = np.random.dirichlet([self.alpha] * self.width).astype(np.float32)

        # Build the mixture of augmented chains.
        mixture = np.zeros_like(original, dtype=np.float32)
        for w in weights:
            augmented = self._apply_chain(img)
            mixture += w * self._to_float_array(augmented)

        # Blend original and mixture with a Beta-sampled coefficient.
        m = float(np.random.beta(self.alpha, self.alpha))
        result = m * original + (1.0 - m) * mixture

        return self._to_pil(result)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"severity={int(self.severity * 10)}, "
            f"width={self.width}, "
            f"depth={self.depth}, "
            f"alpha={self.alpha})"
        )
