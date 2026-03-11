"""
Non-blind (authorized) deblurring: smooth Wiener deconvolution + bilinear blending.

Mirrors the forward model exactly.  For each of the 49 PSF anchors, we apply
Wiener deconvolution to the *full* blurred image using that anchor's kernel,
then blend the 49 recovered images using the identical bilinear weight maps
used during encryption.

In the limit of perfect Wiener inversion and spatially-varying PSFs that are
well-separated, this is equivalent to globally inverting the SV-PSF operator.
In practice, cross-terms from neighbouring PSFs add a small noise floor, but
the approximation is tight when the PSFs vary smoothly, which is exactly the
regime the forward model is designed for.
"""

import numpy as np
from .blur import _crop_kernel, _bilinear_weights, _weight_map


def wiener_patch(
    image: np.ndarray,
    kernel: np.ndarray,
    reg: float = 5e-2,
) -> np.ndarray:
    """
    Full-image Wiener deconvolution in the frequency domain.

    The kernel is zero-padded to the image size and circularly shifted
    so its centre lies at position (0, 0) before the DFT, eliminating
    any phase offset in the recovered image.

    Parameters
    ----------
    image  : (H, W) float64  — blurred image
    kernel : (kH, kW)        — PSF kernel (already cropped and normalised)
    reg    : float           — Tikhonov regularisation constant λ

    Returns
    -------
    recovered : (H, W) float64
    """
    H, W    = image.shape
    kH, kW  = kernel.shape
    # Zero-pad kernel to image size, shift centre → (0, 0)
    k_pad = np.zeros((H, W), dtype=np.float64)
    k_pad[:kH, :kW] = kernel
    k_pad = np.roll(k_pad, -(kH // 2), axis=0)
    k_pad = np.roll(k_pad, -(kW // 2), axis=1)

    H_fft  = np.fft.fft2(k_pad)
    Y_fft  = np.fft.fft2(image)
    H_conj = np.conj(H_fft)
    X_fft  = (H_conj / (H_conj * H_fft + reg)) * Y_fft
    return np.real(np.fft.ifft2(X_fft))


def non_blind_deblur(
    blurred: np.ndarray,
    psf: np.ndarray,
    kernel_size: int = 71,
    reg: float = 5e-2,
) -> np.ndarray:
    """
    Smooth non-blind deblurring via Wiener deconvolution + bilinear blending.

    Applies Wiener deconvolution of the full image with each of the R×C PSF
    kernels, then blends the results using the same bilinear weight maps as
    the forward model — the exact spatial inverse of ``apply_sv_psf``.

    Parameters
    ----------
    blurred     : (H, W) float32 in [0, 1]
    psf         : (R, C, kH, kW)
    kernel_size : must match the value used during blurring
    reg         : Wiener regularisation constant λ

    Returns
    -------
    restored : (H, W) float32 in [0, 1]
    """
    H, W    = blurred.shape
    R, C    = psf.shape[:2]
    img     = blurred.astype(np.float64)
    result  = np.zeros((H, W), dtype=np.float64)

    i0, i1, j0, j1, fy, fx = _bilinear_weights(H, W, R, C)

    for i in range(R):
        for j in range(C):
            weight = _weight_map(i, j, i0, i1, j0, j1, fy, fx)
            if weight.max() == 0.0:
                continue
            kernel    = _crop_kernel(psf[i, j], kernel_size)
            recovered = wiener_patch(img, kernel, reg=reg)
            result   += weight * recovered

    return np.clip(result, 0.0, 1.0).astype(np.float32)
