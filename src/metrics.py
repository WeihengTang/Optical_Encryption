"""Image quality metrics: PSNR and SSIM."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def psnr(ref: np.ndarray, est: np.ndarray) -> float:
    return float(_psnr(ref, est, data_range=1.0))


def ssim(ref: np.ndarray, est: np.ndarray) -> float:
    return float(_ssim(ref, est, data_range=1.0))


def compute_all(ref: np.ndarray, est: np.ndarray) -> dict:
    return {"PSNR": round(psnr(ref, est), 2), "SSIM": round(ssim(ref, est), 4)}
