"""
Restormer blind baseline — zero-shot motion-deblurring model.

Restormer (Zamir et al., CVPR 2022) is a transformer-based image restoration
network.  We use the pretrained motion-deblurring checkpoint (trained on
GoPro + HIDE datasets) and apply it zero-shot to our SV-PSF encrypted images.

Like NAFNet, this simulates a blind adversary who has SOTA deep-learning tools
but no knowledge of the metasurface PSF (the encryption key).

We use overlapping tile inference to handle 1008×1008 images.

Reference:
  Zamir et al., "Restormer: Efficient Transformer for High-Resolution
  Image Restoration", CVPR 2022.
"""
import sys
import types
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch the module removed in torchvision>=0.14 before basicsr tries to
# import it.  This must happen before any basicsr import, direct or indirect.
import torchvision.transforms.functional as _F
_fake = types.ModuleType('torchvision.transforms.functional_tensor')
_fake.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _fake

import json
import numpy as np
import torch

# Prioritise the Restormer repo's own basicsr over the installed one.
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Restormer'))
from basicsr.models.archs.restormer_arch import Restormer

from src.metrics import compute_all

# ── config ────────────────────────────────────────────────────────────────────
WEIGHT_PATH = Path(__file__).parent / 'weights' / 'motion_deblurring.pth'
DATA_DIR    = Path(__file__).parent / 'data'
OUT_DIR     = Path(__file__).parent / 'results'

TILE_SIZE    = 256
TILE_OVERLAP = 32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def load_restormer(weight_path: Path) -> Restormer:
    model = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
    )
    ckpt  = torch.load(str(weight_path), map_location=DEVICE)
    state = ckpt.get('params', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model


def tile_inference(model: Restormer,
                   img: np.ndarray,
                   tile: int = TILE_SIZE,
                   overlap: int = TILE_OVERLAP) -> np.ndarray:
    H, W    = img.shape
    step    = tile - overlap
    out     = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    hann1d = np.hanning(tile).astype(np.float32)
    win    = np.outer(hann1d, hann1d)

    ys = list(range(0, H - tile + 1, step)) + [H - tile]
    xs = list(range(0, W - tile + 1, step)) + [W - tile]

    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            patch = img[y0:y0+tile, x0:x0+tile]
            t = torch.from_numpy(patch).float()
            t = t.unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(t).squeeze().cpu().numpy()
            out    [y0:y0+tile, x0:x0+tile] += win * pred
            weights[y0:y0+tile, x0:x0+tile] += win

    return np.clip(out / (weights + 1e-8), 0.0, 1.0)


def main():
    model = load_restormer(WEIGHT_PATH)

    results = {}
    names   = ['camera', 'astronaut', 'chelsea']

    for name in names:
        print(f'\nProcessing {name}...')
        orig    = np.load(str(DATA_DIR / f'orig_{name}.npy'))
        blurred = np.load(str(DATA_DIR / f'blurred_{name}.npy'))

        restored = tile_inference(model, blurred)
        metrics  = compute_all(orig, restored)
        results[name] = metrics
        print(f'  PSNR={metrics["PSNR"]}  SSIM={metrics["SSIM"]}')

        np.save(str(OUT_DIR / f'restormer_{name}.npy'), restored)

    with open(str(OUT_DIR / 'restormer_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\nRestormer results saved to baselines/results/')


if __name__ == '__main__':
    main()
