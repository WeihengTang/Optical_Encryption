"""
DPIR non-blind baseline — PnP-HQS with DRUNet denoiser.

DPIR (Zhang et al., TPAMI 2021) uses Half-Quadratic Splitting (HQS):

  Phase 1 (data):    z_k = Wiener(y, PSF_k, reg=ρ)
  Phase 2 (prior):   x_{k+1} = DRUNet(z_k, σ_k)

The data step uses our exact SV-PSF Wiener deblur (the authorized decoder
knows the full PSF), so this is a non-blind method.  The DRUNet denoiser
is the pretrained gray-scale model from the DPIR repo.

The noise level σ_k fed to the denoiser is annealed from σ_start → σ_end
over N_ITER iterations, following the DPIR schedule.

Reference:
  Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
  IEEE TPAMI 44(10), 2022.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'DPIR'))

import json
import numpy as np
import torch
import torch.nn.functional as F

# DPIR's DRUNet architecture
from models.network_unet import UNetRes as DRUNet

from src.non_blind_deblur import non_blind_deblur
from src.metrics import compute_all

# ── config ────────────────────────────────────────────────────────────────────
WEIGHT_PATH = Path(__file__).parent / 'weights' / 'drunet_gray.pth'
DATA_DIR    = Path(__file__).parent / 'data'
OUT_DIR     = Path(__file__).parent / 'results'
PSF_PATH    = Path(__file__).parent.parent / 'SV_PSF.npy'

N_ITER      = 8          # HQS outer iterations (8 is standard for DPIR)
SIGMA_START = 49 / 255   # noise level fed to denoiser at iter 0 (high)
SIGMA_END   = 2.55 / 255 # noise level at final iter (low)
KSIZE       = 71

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def load_drunet(weight_path: Path) -> DRUNet:
    model = DRUNet(in_nc=2, out_nc=1, nc=[64, 128, 256, 512],
                   nb=4, act_mode='R', downsample_mode='strideconv',
                   upsample_mode='convtranspose')
    state = torch.load(str(weight_path), map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


def drunet_denoise(model: DRUNet,
                   img: np.ndarray,
                   sigma: float) -> np.ndarray:
    """Run DRUNet denoiser on a (H,W) float64 image."""
    H, W = img.shape
    # DRUNet expects (1, C+1, H, W) where the extra channel is σ/255
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    noise_map = torch.full_like(x, sigma)                         # (1,1,H,W)
    inp = torch.cat([x, noise_map], dim=1).to(DEVICE)            # (1,2,H,W)
    with torch.no_grad():
        out = model(inp).squeeze().cpu().numpy()
    return np.clip(out, 0.0, 1.0)


def dpir_deblur(blurred: np.ndarray,
                psf: np.ndarray,
                model: DRUNet,
                n_iter: int = N_ITER) -> np.ndarray:
    """
    PnP-HQS deblurring with the known SV-PSF.

    The data step is our Wiener deblur; the prior step is DRUNet.
    The regularisation ρ (passed as `reg` to Wiener) is coupled to σ_k:
      ρ_k = σ_k² / σ_noise²   (following DPIR Eq. 9)
    We assume σ_noise ≈ 2.55/255 (very slight observation noise).
    """
    sigma_noise = 2.55 / 255
    sigmas = np.linspace(SIGMA_START, SIGMA_END, n_iter)

    x = blurred.astype(np.float64)
    for k, sigma_k in enumerate(sigmas):
        rho_k = float(sigma_k ** 2 / sigma_noise ** 2)
        # Data step: Wiener with reg = ρ_k (balances data fidelity vs prior)
        z = non_blind_deblur(x.astype(np.float32), psf,
                             kernel_size=KSIZE, reg=rho_k)
        z = z.astype(np.float64)
        # Prior step: DRUNet denoiser
        x = drunet_denoise(model, z, sigma_k)
        print(f'    HQS iter {k+1}/{n_iter}  σ={sigma_k*255:.1f}  ρ={rho_k:.3f}')

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def main():
    psf   = np.load(str(PSF_PATH))
    model = load_drunet(WEIGHT_PATH)

    results = {}
    names   = ['camera', 'astronaut', 'chelsea']

    for name in names:
        print(f'\nProcessing {name}...')
        orig    = np.load(str(DATA_DIR / f'orig_{name}.npy'))
        blurred = np.load(str(DATA_DIR / f'blurred_{name}.npy'))

        restored = dpir_deblur(blurred, psf, model)
        metrics  = compute_all(orig, restored)
        results[name] = metrics
        print(f'  PSNR={metrics["PSNR"]}  SSIM={metrics["SSIM"]}')

        np.save(str(OUT_DIR / f'dpir_{name}.npy'), restored)

    with open(str(OUT_DIR / 'dpir_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\nDPIR results saved to baselines/results/')


if __name__ == '__main__':
    main()
