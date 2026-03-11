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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'NAFNet'))

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
    model = NAFNet(img_channel=1, width=64, middle_blk_num=12,
                   enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2])
    ckpt  = torch.load(str(weight_path), map_location=DEVICE)
    # GoPro checkpoint stores under 'params'
    state = ckpt.get('params', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model


def tile_inference(model: NAFNet,
                   img: np.ndarray,
                   tile: int = TILE_SIZE,
                   overlap: int = TILE_OVERLAP) -> np.ndarray:
    """
    Run model on (H,W) image via overlapping tiles with linear blend.
    """
    H, W    = img.shape
    step    = tile - overlap
    out     = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    # Build a 2-D Hann window for smooth blending
    hann1d = np.hanning(tile).astype(np.float32)
    win    = np.outer(hann1d, hann1d)

    ys = list(range(0, H - tile + 1, step)) + [H - tile]
    xs = list(range(0, W - tile + 1, step)) + [W - tile]

    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            patch = img[y0:y0+tile, x0:x0+tile]
            t = torch.from_numpy(patch).float()
            t = t.unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1,1,T,T)
            with torch.no_grad():
                pred = model(t).squeeze().cpu().numpy()
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
