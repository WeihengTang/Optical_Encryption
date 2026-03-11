"""
Re-run the two classical baselines (Wiener non-blind + Unsupervised Wiener-Hunt)
and write their metrics to baselines/results/ in the same format as the deep
baseline runners, so collect_results.py can merge everything into one table.

Run this on the server alongside the deep baseline scripts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from src.non_blind_deblur import non_blind_deblur
from src.blind_deblur import blind_wiener
from src.metrics import compute_all

PSF_PATH    = Path(__file__).parent.parent / 'SV_PSF.npy'
DATA_DIR    = Path(__file__).parent / 'data'
OUT_DIR     = Path(__file__).parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

PSF   = np.load(str(PSF_PATH))
NAMES = ['camera', 'astronaut', 'chelsea']

wiener_results = {}
uwh_results    = {}

for name in NAMES:
    print(f'Processing {name}...')
    orig    = np.load(str(DATA_DIR / f'orig_{name}.npy'))
    blurred = np.load(str(DATA_DIR / f'blurred_{name}.npy'))

    nb  = non_blind_deblur(blurred, PSF, kernel_size=71, reg=5e-2)
    bl  = blind_wiener(blurred)

    np.save(str(OUT_DIR / f'wiener_nb_{name}.npy'), nb)
    np.save(str(OUT_DIR / f'uwh_{name}.npy'),       bl)

    wiener_results[name] = compute_all(orig, nb)
    uwh_results[name]    = compute_all(orig, bl)

    print(f'  Wiener NB : PSNR={wiener_results[name]["PSNR"]}  '
          f'SSIM={wiener_results[name]["SSIM"]}')
    print(f'  UWH blind : PSNR={uwh_results[name]["PSNR"]}  '
          f'SSIM={uwh_results[name]["SSIM"]}')

with open(str(OUT_DIR / 'wiener_nb_metrics.json'), 'w') as f:
    json.dump(wiener_results, f, indent=2)
with open(str(OUT_DIR / 'uwh_metrics.json'), 'w') as f:
    json.dump(uwh_results, f, indent=2)

print('\nClassical metrics written to baselines/results/')
