"""
Smooth SV-PSF forward model: global convolution + bilinear blending.

The naive approach of tiling the image into 7×7 hard patches and convolving
each patch independently creates visible seams at patch boundaries, because
the PSF jumps discontinuously across them.

This module implements a smooth alternative (Method 1 from the spec):

  1. Convolve the *full* image with each of the 49 PSF kernels independently.
     This yields 49 full-resolution blurred images, one per PSF sample.

  2. For every pixel (y, x) in the output, compute a weighted sum of the 49
     blurred images.  The weights are determined by bilinear interpolation of
     the pixel's position within the 7×7 PSF anchor grid, so the effective
     kernel transitions smoothly and continuously across the field of view.

The result is a spatially-varying blur with no seam artifacts.
"""

import numpy as np
from scipy.signal import fftconvolve


def _crop_kernel(kernel: np.ndarray, size: int) -> np.ndarray:
    """Crop kernel to (size × size) around its peak centre, then L1-normalise."""
    kH, kW = kernel.shape
    cH, cW = kH // 2, kW // 2
    half = size // 2
    r0, r1 = max(0, cH - half), min(kH, cH + half + 1)
    c0, c1 = max(0, cW - half), min(kW, cW + half + 1)
    cropped = kernel[r0:r1, c0:c1]
    total = cropped.sum()
    return (cropped / total) if total > 0 else cropped


def _bilinear_weights(H: int, W: int, R: int, C: int):
    """
    Precompute bilinear blending components for a (H×W) image and (R×C) PSF grid.

    The R×C PSF anchors are placed at pixel positions:
        row_i = i * (H-1) / (R-1),   i = 0 … R-1
        col_j = j * (W-1) / (C-1),   j = 0 … C-1

    so anchor (0,0) sits at the top-left corner and anchor (R-1,C-1) at the
    bottom-right corner.

    Returns
    -------
    i0, i1 : (H,)   int   — lower/upper row bracket indices
    j0, j1 : (W,)   int   — lower/upper col bracket indices
    fy     : (H, 1) float — fractional row offset  (gy - i0)
    fx     : (1, W) float — fractional col offset  (gx - j0)
    """
    gy = np.arange(H, dtype=np.float64) * (R - 1) / (H - 1)
    gx = np.arange(W, dtype=np.float64) * (C - 1) / (W - 1)

    i0 = np.floor(gy).astype(np.int32).clip(0, R - 2)   # (H,)
    j0 = np.floor(gx).astype(np.int32).clip(0, C - 2)   # (W,)
    i1 = i0 + 1                                           # (H,)
    j1 = j0 + 1                                           # (W,)

    fy = (gy - i0).astype(np.float64)[:, None]            # (H, 1)
    fx = (gx - j0).astype(np.float64)[None, :]            # (1, W)

    return i0, i1, j0, j1, fy, fx


def _weight_map(i: int, j: int,
                i0, i1, j0, j1,
                fy, fx) -> np.ndarray:
    """
    Compute the (H, W) blending weight map for PSF anchor (i, j).

    PSF[i,j] contributes to a pixel at grid position (gy, gx) as:
      - the (i0,j0) corner  →  weight (1-fy)(1-fx)   when i0==i, j0==j
      - the (i0,j1) corner  →  weight (1-fy)fx        when i0==i, j1==j
      - the (i1,j0) corner  →  weight fy(1-fx)        when i1==i, j0==j
      - the (i1,j1) corner  →  weight fy·fx            when i1==i, j1==j
    """
    # Boolean masks, broadcast to (H, W)
    m_i0 = (i0 == i)[:, None]   # (H, 1)
    m_i1 = (i1 == i)[:, None]   # (H, 1)
    m_j0 = (j0 == j)[None, :]   # (1, W)
    m_j1 = (j1 == j)[None, :]   # (1, W)

    return (
        (m_i0 & m_j0) * ((1 - fy) * (1 - fx)) +
        (m_i0 & m_j1) * ((1 - fy) * fx      ) +
        (m_i1 & m_j0) * (fy       * (1 - fx)) +
        (m_i1 & m_j1) * (fy       * fx       )
    ).astype(np.float64)


def apply_sv_psf(
    image: np.ndarray,
    psf: np.ndarray,
    kernel_size: int = 71,
) -> np.ndarray:
    """
    Apply a spatially-varying PSF to a grayscale image without seam artifacts.

    Parameters
    ----------
    image       : (H, W) float32 in [0, 1]
    psf         : (R, C, kH, kW)  —  SV-PSF grid
    kernel_size : crop size for each kernel (71 px retains 96.6 % energy)

    Returns
    -------
    blurred : (H, W) float32 in [0, 1]
    """
    H, W    = image.shape
    R, C    = psf.shape[:2]
    img     = image.astype(np.float64)
    result  = np.zeros((H, W), dtype=np.float64)

    i0, i1, j0, j1, fy, fx = _bilinear_weights(H, W, R, C)

    for i in range(R):
        for j in range(C):
            weight = _weight_map(i, j, i0, i1, j0, j1, fy, fx)
            if weight.max() == 0.0:
                continue
            kernel  = _crop_kernel(psf[i, j], kernel_size)
            blurred = fftconvolve(img, kernel.astype(np.float64), mode='same')
            result += weight * blurred

    return np.clip(result, 0.0, 1.0).astype(np.float32)
