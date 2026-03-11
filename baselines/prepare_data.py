"""
Prepare and save encrypted test images that all baselines share.

Saves to baselines/data/:
  orig_{name}.npy      — original float32 [0,1] image, (1008,1008)
  blurred_{name}.npy   — SV-PSF encrypted image, same shape
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
from pathlib import Path
from skimage import data, color, transform

from src.blur import apply_sv_psf

PSF      = np.load('SV_PSF.npy')
KSIZE    = 71
IMSZ     = 1008
OUT_DIR  = Path('baselines/data')
OUT_DIR.mkdir(exist_ok=True)

IMAGES = {
    'camera':    data.camera().astype(np.float32) / 255.0,
    'astronaut': color.rgb2gray(data.astronaut()).astype(np.float32),
    'chelsea':   color.rgb2gray(data.chelsea()).astype(np.float32),
}

for name, img in IMAGES.items():
    orig = transform.resize(img, (IMSZ, IMSZ),
                            anti_aliasing=True).astype(np.float32)
    print(f'  Encrypting {name}...')
    blurred = apply_sv_psf(orig, PSF, kernel_size=KSIZE)
    np.save(OUT_DIR / f'orig_{name}.npy',    orig)
    np.save(OUT_DIR / f'blurred_{name}.npy', blurred)
    print(f'    saved orig_{name}.npy  blurred_{name}.npy')

print('Data ready in baselines/data/')
