"""
NAFNet blind baseline — zero-shot application of the GoPro-trained model.

NAFNet (Chen et al., ECCV 2022) is trained on motion-blurred images from the
GoPro dataset.  Here we apply it zero-shot (no fine-tuning) to our SV-PSF
encrypted images.  The model has no knowledge of the PSF.

This simulates a state-of-the-art blind adversary who has access to a
SOTA image restoration network but does not know the encryption key (PSF).

NAFNet processes the full image at once; we tile into overlapping 256×256
patches to respect VRAM limits, then stitch with linear blending.

Reference:
  Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
"""
import sys
import types
from pathlib import Path

# ── Import isolation ──────────────────────────────────────────────────────────
_nafnet_root   = Path(__file__).parent / 'third_party' / 'NAFNet'
_nafnet_basicsr = _nafnet_root / 'basicsr'

# 1. NAFNet ships without basicsr/__init__.py, so Python doesn't treat it as a
#    package and falls back to the installed (incompatible) basicsr.
#    Creating an empty __init__.py makes it a proper package.
(_nafnet_basicsr / '__init__.py').touch(exist_ok=True)

# 2. Put NAFNet's basicsr first so it wins over the installed version.
sys.path.insert(0, str(_nafnet_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

# 3. Evict any stale cached basicsr.
for _k in [k for k in sys.modules if k == 'basicsr' or k.startswith('basicsr.')]:
    del sys.modules[_k]

# 4. Patch the torchvision sub-module removed in torchvision>=0.14.
import torchvision.transforms.functional as _F
_fake = types.ModuleType('torchvision.transforms.functional_tensor')
_fake.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _fake
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import torch
from basicsr.models.archs.NAFNet_arch import NAFNet

from src.metrics import compute_all

# ── config ────────────────────────────────────────────────────────────────────
WEIGHT_PATH = Path(__file__).parent / 'weights' / 'NAFNet-GoPro-width64.pth'
DATA_DIR    = Path(__file__).parent / 'data'
OUT_DIR     = Path(__file__).parent / 'results'

TILE_SIZE   = 256   # process in 256×256 tiles (fits ~4 GB VRAM)
TILE_OVERLAP = 32   # overlap to avoid tile boundary artefacts

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def load_nafnet(weight_path: Path) -> NAFNet:
    # GoPro model is RGB (img_channel=3); correct param names from the repo
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt  = torch.load(str(weight_path), map_location=DEVICE)
    state = ckpt.get('params', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


def tile_inference(model: NAFNet,
                   img: np.ndarray,
                   tile: int = TILE_SIZE,
                   overlap: int = TILE_OVERLAP) -> np.ndarray:
    """
    Run model on (H,W) grayscale image via overlapping tiles with linear blend.
    Grayscale is broadcast to 3 channels to match the RGB-trained model;
    the 3 output channels are averaged back to grayscale.
    """
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
            # Repeat grayscale → 3 channels: (1, 3, T, T)
            t = torch.from_numpy(patch).float()
            t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(t).squeeze().mean(0).cpu().numpy()  # avg 3 ch → 1
            out    [y0:y0+tile, x0:x0+tile] += win * pred
            weights[y0:y0+tile, x0:x0+tile] += win

    return np.clip(out / (weights + 1e-8), 0.0, 1.0)


def main():
    model = load_nafnet(WEIGHT_PATH)

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

        np.save(str(OUT_DIR / f'nafnet_{name}.npy'), restored)

    with open(str(OUT_DIR / 'nafnet_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\nNAFNet results saved to baselines/results/')


if __name__ == '__main__':
    main()
