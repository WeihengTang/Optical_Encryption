"""Generate publication-quality PDF figures for the LaTeX report."""
import sys
sys.path.insert(0, '/home/www/MetaLens')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from skimage import data, color, transform

from src.blur import apply_sv_psf
from src.non_blind_deblur import non_blind_deblur
from src.blind_deblur import blind_wiener
from src.metrics import compute_all

FIG_DIR = Path('/home/www/MetaLens/report/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

PSF   = np.load('/home/www/MetaLens/SV_PSF.npy')
KSIZE = 71
REG   = 5e-2
IMSZ  = 1008

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    9,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})

def load_images():
    raw = {
        'camera':    data.camera().astype(np.float32) / 255.0,
        'astronaut': color.rgb2gray(data.astronaut()).astype(np.float32),
        'chelsea':   color.rgb2gray(data.chelsea()).astype(np.float32),
    }
    return {k: transform.resize(v, (IMSZ, IMSZ), anti_aliasing=True).astype(np.float32)
            for k, v in raw.items()}

def run_pipeline(images):
    results = {}
    for name, img in images.items():
        print(f'  Processing {name}...')
        blurred = apply_sv_psf(img, PSF, kernel_size=KSIZE)
        nb      = non_blind_deblur(blurred, PSF, kernel_size=KSIZE, reg=REG)
        bl      = blind_wiener(blurred, psf_size=KSIZE)
        results[name] = dict(
            orig=img, blurred=blurred, nb=nb, bl=bl,
            m_nb=compute_all(img, nb),
            m_bl=compute_all(img, bl),
        )
    return results

def fig_psf_grid():
    R, C = PSF.shape[:2]
    fig, axes = plt.subplots(R, C, figsize=(6.5, 6.5))
    vmax = PSF.max()
    for i in range(R):
        for j in range(C):
            ax = axes[i][j]
            ax.imshow(np.log1p(PSF[i, j] / vmax * 1e6), cmap='inferno', origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r'Spatially-varying PSF — $7\times7$ field grid, each kernel $101\times101$ px',
                 fontsize=8, y=1.002)
    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / 'psf_grid.pdf')
    plt.close(fig)
    print('  psf_grid.pdf done')

def fig_panel(name, r):
    m_nb, m_bl = r['m_nb'], r['m_bl']
    titles = [
        'Original',
        'Encrypted (SV-PSF)',
        f"Non-blind (authorized)\nPSNR {m_nb['PSNR']} dB  SSIM {m_nb['SSIM']}",
        f"Blind (adversary)\nPSNR {m_bl['PSNR']} dB  SSIM {m_bl['SSIM']}",
    ]
    arrays = [r['orig'], r['blurred'], r['nb'], r['bl']]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.95))
    for ax, arr, ttl in zip(axes, arrays, titles):
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(ttl, fontsize=7, pad=3)
        ax.axis('off')
    plt.tight_layout(pad=0.4)
    fig.savefig(FIG_DIR / f'results_{name}.pdf')
    plt.close(fig)
    print(f'  results_{name}.pdf done')

def fig_gap_chart(results):
    names   = list(results.keys())
    nb_psnr = [results[n]['m_nb']['PSNR'] for n in names]
    bl_psnr = [results[n]['m_bl']['PSNR'] for n in names]
    gaps    = [nb - bl for nb, bl in zip(nb_psnr, bl_psnr)]
    x, w    = np.arange(len(names)), 0.30

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.bar(x - w/2, nb_psnr, w, label='Non-blind (authorized)',
           color='#6ea8c8', edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, bl_psnr, w, label='Blind (adversary)',
           color='#c8b0a0', edgecolor='white', linewidth=0.5)
    for xi, gap, nb in zip(x, gaps, nb_psnr):
        ax.annotate(f'+{gap:.1f} dB', xy=(xi, nb + 0.4),
                    ha='center', va='bottom', fontsize=7.5,
                    color='#222', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=9)
    ax.set_ylim(0, max(nb_psnr) + 3.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=5))
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#ccc')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'gap_chart.pdf')
    plt.close(fig)
    print('  gap_chart.pdf done')

if __name__ == '__main__':
    print('Loading images and running pipeline...')
    images  = load_images()
    results = run_pipeline(images)
    print('Generating figures...')
    fig_psf_grid()
    for name, r in results.items():
        fig_panel(name, r)
    fig_gap_chart(results)
    print('All figures saved to', FIG_DIR)
