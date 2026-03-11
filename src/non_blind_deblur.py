"""
Non-blind (authorized) deblurring via patch-wise Wiener deconvolution.

The receiver knows the exact SV-PSF.  Each image patch is deconvolved with
its corresponding kernel using Wiener's frequency-domain formula:

    X̂(ω) = H*(ω) / (|H(ω)|² + K) · Y(ω)

where K is a regularisation constant controlling the noise-vs-ringing trade-off.
"""

import numpy as np

from .blur import _crop_kernel, get_patch_slices


def wiener_patch(
    blurred_patch: np.ndarray,
    kernel: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    """Single-patch Wiener deconvolution (frequency domain).

    The kernel must be zero-padded to patch size with its centre placed at
    array position (0, 0) — achieved here via np.roll — so that np.fft.fft2
    interprets it correctly (no phase shift in the recovered image).
    """
    pH, pW = blurred_patch.shape
    kH, kW = kernel.shape
    # Zero-pad kernel to patch size, then circular-shift centre → (0,0)
    k_pad = np.zeros((pH, pW), dtype=np.float64)
    k_pad[:kH, :kW] = kernel
    k_pad = np.roll(k_pad, -(kH // 2), axis=0)
    k_pad = np.roll(k_pad, -(kW // 2), axis=1)

    H_fft = np.fft.fft2(k_pad)
    Y_fft = np.fft.fft2(blurred_patch)
    H_conj = np.conj(H_fft)
    X_fft = (H_conj / (H_conj * H_fft + reg)) * Y_fft
    return np.real(np.fft.ifft2(X_fft))


def non_blind_deblur(
    blurred: np.ndarray,
    psf: np.ndarray,
    kernel_size: int = 71,
    reg: float = 1e-3,
) -> np.ndarray:
    """
    Patch-wise Wiener deconvolution with known SV-PSF.

    Parameters
    ----------
    blurred     : (H, W) float32 in [0, 1]
    psf         : (R, C, kH, kW)
    kernel_size : must match the size used during blurring
    reg         : Wiener regularisation constant (higher → smoother)

    Returns
    -------
    restored : (H, W) float32 in [0, 1]
    """
    H, W = blurred.shape
    R, C = psf.shape[:2]
    result = np.zeros_like(blurred)

    row_slices = get_patch_slices(H, R)
    col_slices = get_patch_slices(W, C)

    for i, rs in enumerate(row_slices):
        for j, cs in enumerate(col_slices):
            patch = blurred[rs, cs]
            kernel = _crop_kernel(psf[i, j], kernel_size)
            # Pad kernel to patch shape
            restored = wiener_patch(patch, kernel, reg=reg)
            result[rs, cs] = restored

    return np.clip(result, 0.0, 1.0)
