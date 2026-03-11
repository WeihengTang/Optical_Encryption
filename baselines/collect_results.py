"""
Collect all baseline results and print a formatted comparison table.

Run this after all run_*.py scripts have completed.
Reads JSON metric files from baselines/results/ and prints a LaTeX-ready
and a human-readable table.

Usage:
    python baselines/collect_results.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'

# Methods in display order: (key, label, role)
METHODS = [
    ('wiener_nb',   'Wiener (non-blind)',            'auth'),
    ('dpir',        'DPIR (non-blind, deep)',        'auth'),
    ('uwh',         'Unsuper. Wiener-Hunt (blind)',  'adv'),
    ('nafnet',      'NAFNet (blind, zero-shot)',     'adv'),
    ('restormer',   'Restormer (blind, zero-shot)',  'adv'),
]

IMAGES = ['camera', 'astronaut', 'chelsea']


def load_metrics(key: str) -> dict | None:
    path = RESULTS_DIR / f'{key}_metrics.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val) -> str:
    return f'{float(val):.2f}' if val is not None else '—'


def main():
    all_data = {key: load_metrics(key) for key, *_ in METHODS}

    # ── Human-readable table ──────────────────────────────────────────────────
    col_w = 26
    print('\n' + '=' * 90)
    print('BASELINE COMPARISON  —  PSNR (dB) / SSIM')
    print('=' * 90)
    header = f"{'Method':<{col_w}}" + ''.join(
        f'  {img.upper():<18}' for img in IMAGES
    )
    print(header)
    print('-' * 90)

    for key, label, role in METHODS:
        data = all_data[key]
        row  = f'{"[AUTH] " if role=="auth" else "[ADV]  "}{label:<{col_w-7}}'
        for img in IMAGES:
            if data and img in data:
                m = data[img]
                row += f'  {fmt(m["PSNR"])} / {fmt(m["SSIM"])}    '
            else:
                row += f'  {"not run":<20}'
        print(row)

    print('=' * 90)

    # ── Security gap table ────────────────────────────────────────────────────
    print('\nSECURITY GAP  (best authorized − best adversary, per image, PSNR dB)')
    print('-' * 60)

    auth_keys = [k for k, _, r in METHODS if r == 'auth']
    adv_keys  = [k for k, _, r in METHODS if r == 'adv']

    gaps = []
    for img in IMAGES:
        best_auth, best_adv = None, None
        for k in auth_keys:
            d = all_data.get(k)
            if d and img in d:
                v = float(d[img]['PSNR'])
                best_auth = max(best_auth, v) if best_auth else v
        for k in adv_keys:
            d = all_data.get(k)
            if d and img in d:
                v = float(d[img]['PSNR'])
                best_adv = max(best_adv, v) if best_adv else v
        gap = (best_auth - best_adv) if (best_auth and best_adv) else None
        gaps.append(gap)
        print(f'  {img:<12}  auth={fmt(best_auth)} dB   adv={fmt(best_adv)} dB   '
              f'gap={fmt(gap)} dB')

    avg_gap = sum(g for g in gaps if g) / sum(1 for g in gaps if g)
    print(f'\n  Average gap: {avg_gap:.2f} dB')

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print('\n\nLATEX TABLE (copy into report.tex):\n')
    print(r'\begin{tabular}{llccc}')
    print(r'  \toprule')
    print(r'  & Method & camera & astronaut & chelsea \\')
    print(r'  \midrule')

    for key, label, role in METHODS:
        data = all_data[key]
        role_str = r'\textsc{auth}' if role == 'auth' else r'\textsc{adv}'
        cells = []
        for img in IMAGES:
            if data and img in data:
                m = data[img]
                cells.append(f'{fmt(m["PSNR"])} / {fmt(m["SSIM"])}')
            else:
                cells.append('—')
        print(f'  {role_str} & {label} & {" & ".join(cells)} \\\\')

    print(r'  \bottomrule')
    print(r'\end{tabular}')


if __name__ == '__main__':
    main()
