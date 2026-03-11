"""
Blind (adversary) deblurring — no knowledge of the PSF.

Two strategies are offered:

1. unsupervised_wiener  – scikit-image's joint image/PSF estimation under
   a Gaussian prior.  It assumes a *spatially uniform* blur, so the SV-PSF
   will confuse the estimator.

2. richardson_lucy_blind – Richardson-Lucy with a rough Gaussian PSF guess.
   Represents an attacker who assumes a simple isotropic blur.
"""

import numpy as np
from skimage.restoration import unsupervised_wiener, richardson_lucy
from scipy.ndimage import gaussian_filter


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_gaussian_kernel(size: int = 21, sigma: float = 5.0) -> np.ndarray:
    """Build a 2-D Gaussian kernel — the adversary's (wrong) PSF guess."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


# ── public API ────────────────────────────────────────────────────────────────

def blind_wiener(
    blurred: np.ndarray,
    psf_size: int = 71,
    n_iter: int = 30,
) -> np.ndarray:
    """
    Joint image + PSF estimation via unsupervised Wiener (Gaussian prior).

    The adversary feeds the *whole* blurred image to a spatially-uniform blind
    deconvolution algorithm.  The SV nature of the true PSF means the estimated
    kernel is an average/muddle of all 49 local kernels.

    Parameters
    ----------
    blurred  : (H, W) float32 in [0, 1]
    psf_size : initial PSF support assumed by the estimator
    n_iter   : EM iterations

    Returns
    -------
    restored : (H, W) float32 in [0, 1]
    """
    # unsupervised_wiener needs a psf initialisation; use a centred Gaussian
    psf_init = _make_gaussian_kernel(psf_size, sigma=psf_size / 6)
    user_params = {"max_iter": n_iter}
    restored, _ = unsupervised_wiener(blurred, psf_init,
                                      user_params=user_params, clip=True)
    return np.clip(restored, 0.0, 1.0)


def blind_rl(
    blurred: np.ndarray,
    assumed_sigma: float = 8.0,
    n_iter: int = 30,
) -> np.ndarray:
    """
    Richardson-Lucy with an assumed Gaussian PSF (wrong shape and uniform).

    Fast fallback blind method — the attacker guesses blur radius from the
    image statistics but cannot know the true ring-shaped metasurface kernel.

    Parameters
    ----------
    blurred       : (H, W) float32 in [0, 1]
    assumed_sigma : std-dev of the attacker's Gaussian PSF guess
    n_iter        : R-L iterations

    Returns
    -------
    restored : (H, W) float32 in [0, 1]
    """
    kernel = _make_gaussian_kernel(size=int(assumed_sigma * 6) | 1,
                                   sigma=assumed_sigma)
    restored = richardson_lucy(blurred, kernel, num_iter=n_iter,
                               clip=True)
    return np.clip(restored, 0.0, 1.0)
