"""
Spatially-Varying PSF blurring (encryption) using a metasurface-generated PSF.

The SV-PSF is stored as (R, C, kH, kW) where R x C is the spatial grid of
PSF positions and kH x kW is each kernel's size.  Each image patch at grid
position (i, j) is convolved with PSF[i, j].
"""

import numpy as np
from scipy.signal import fftconvolve


def _crop_kernel(kernel: np.ndarray, size: int) -> np.ndarray:
    """Crop kernel to (size x size) around its center and L1-normalize."""
    kH, kW = kernel.shape
    cH, cW = kH // 2, kW // 2
    half = size // 2
    r0, r1 = max(0, cH - half), min(kH, cH + half + 1)
    c0, c1 = max(0, cW - half), min(kW, cW + half + 1)
    cropped = kernel[r0:r1, c0:c1]
    total = cropped.sum()
    if total > 0:
        cropped = cropped / total
    return cropped


def get_patch_slices(total: int, n: int):
    """Return n roughly equal slice objects covering [0, total)."""
    boundaries = [round(total * i / n) for i in range(n + 1)]
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(n)]


def apply_sv_psf(
    image: np.ndarray,
    psf: np.ndarray,
    kernel_size: int = 71,
) -> np.ndarray:
    """
    Apply spatially-varying PSF to a grayscale image.

    Parameters
    ----------
    image      : (H, W) float32 in [0, 1]
    psf        : (R, C, kH, kW) – SV-PSF grid
    kernel_size: crop size for each kernel (smaller = faster, less accurate)

    Returns
    -------
    blurred : (H, W) float32 in [0, 1]
    """
    H, W = image.shape
    R, C = psf.shape[:2]
    result = np.zeros_like(image)

    row_slices = get_patch_slices(H, R)
    col_slices = get_patch_slices(W, C)

    for i, rs in enumerate(row_slices):
        for j, cs in enumerate(col_slices):
            patch = image[rs, cs]
            kernel = _crop_kernel(psf[i, j], kernel_size)
            blurred = fftconvolve(patch, kernel, mode="same")
            result[rs, cs] = blurred

    return np.clip(result, 0.0, 1.0)
