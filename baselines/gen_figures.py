"""
Generate visual comparison figures for all baselines.

Produces (in baselines/figures/):
  panel_{name}.png   — one row per image: orig / encrypted / all methods
  psnr_bar.png       — grouped bar chart, all methods × all images
  ssim_bar.png       — same for SSIM
  gap_summary.png    — security gap per image (auth vs best adv)

Run on the server after all run_*.py scripts have completed, then
  git add baselines/figures && git commit && git push
so you can git pull the plots locally.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR    = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results'
FIG_DIR     = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          8,
    'axes.titlesize':     8,
    'axes.labelsize':     8,
    'figure.dpi':         150,
    'savefig.dpi':        150,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

IMAGES  = ['camera', 'astronaut', 'chelsea']

# (key, label, role)
METHODS = [
    ('wiener_nb',  'Wiener\n(non-blind)',          'auth'),
    ('dpir',       'DPIR\n(non-blind, deep)',       'auth'),
    ('uwh',        'Wiener-Hunt\n(blind)',           'adv'),
    ('nafnet',     'NAFNet\n(blind)',                'adv'),
    ('restormer',  'Restormer\n(blind)',             'adv'),
]

AUTH_COLOR = '#4e8098'
ADV_COLOR  = '#c06c5a'
GAP_COLOR  = '#5a8a5a'


# ── helpers ───────────────────────────────────────────────────────────────────

def load_npy(name: str) -> np.ndarray | None:
    p = RESULTS_DIR / f'{name}.npy'
    return np.load(str(p)) if p.exists() else None

def load_metrics(key: str) -> dict | None:
    p = RESULTS_DIR / f'{key}_metrics.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def get_psnr(metrics, img) -> float | None:
    if metrics and img in metrics:
        return float(metrics[img]['PSNR'])
    return None

def get_ssim(metrics, img) -> float | None:
    if metrics and img in metrics:
        return float(metrics[img]['SSIM'])
    return None


# ── Figure 1: per-image panel rows ───────────────────────────────────────────

def fig_panels():
    """One figure per test image: orig | encrypted | all 5 methods."""
    all_metrics = {k: load_metrics(k) for k, *_ in METHODS}

    for img_name in IMAGES:
        orig    = np.load(str(DATA_DIR / f'orig_{img_name}.npy'))
        blurred = np.load(str(DATA_DIR / f'blurred_{img_name}.npy'))

        arrays  = [orig, blurred]
        titles  = ['Original', 'Encrypted\n(SV-PSF)']

        for key, label, role in METHODS:
            arr = load_npy(f'{key}_{img_name}')
            m   = all_metrics[key]
            psnr = get_psnr(m, img_name)
            ssim = get_ssim(m, img_name)
            tag  = 'AUTH' if role == 'auth' else 'ADV'
            if arr is not None and psnr is not None:
                titles.append(
                    f'[{tag}] {label}\n'
                    f'PSNR {psnr:.1f} dB / SSIM {ssim:.3f}'
                )
            else:
                titles.append(f'[{tag}] {label}\n(not run)')
            arrays.append(arr)

        n_cols = len(arrays)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(2.2 * n_cols, 2.6))

        for ax, arr, title in zip(axes, arrays, titles):
            if arr is not None:
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1,
                          interpolation='bilinear')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_facecolor('#f0f0f0')
            color = (ADV_COLOR if '[ADV]' in title
                     else AUTH_COLOR if '[AUTH]' in title
                     else '#444')
            ax.set_title(title, fontsize=6.5, pad=3, color=color)
            ax.axis('off')

        fig.suptitle(f'Image: {img_name}', fontsize=9,
                     fontweight='bold', y=1.01)
        plt.tight_layout(pad=0.4)
        out = FIG_DIR / f'panel_{img_name}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'  panel_{img_name}.png')


# ── Figure 2: PSNR grouped bar chart ─────────────────────────────────────────

def fig_psnr_bar():
    all_metrics = {k: load_metrics(k) for k, *_ in METHODS}
    labels      = [lbl.replace('\n', ' ') for _, lbl, _ in METHODS]
    roles       = [r for _, _, r in METHODS]
    keys        = [k for k, _, _ in METHODS]
    colors      = [AUTH_COLOR if r == 'auth' else ADV_COLOR for r in roles]

    n_methods = len(METHODS)
    n_images  = len(IMAGES)
    x         = np.arange(n_images)
    w         = 0.14
    offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2,
                            n_methods) * w

    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    for i, (key, label, role) in enumerate(METHODS):
        m     = all_metrics[key]
        psnrs = [get_psnr(m, img) for img in IMAGES]
        valid = [p if p is not None else 0 for p in psnrs]
        bars  = ax.bar(x + offsets[i], valid, w,
                       label=label.replace('\n', ' '),
                       color=colors[i],
                       edgecolor='white', linewidth=0.4,
                       alpha=0.88)
        for bar, val in zip(bars, psnrs):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=5.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in IMAGES], fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=9)
    ax.set_ylim(0, 30)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)

    # Legend with role separator
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, fontsize=6.5, ncol=n_methods,
              loc='upper center', bbox_to_anchor=(0.5, 1.13),
              framealpha=0.9, edgecolor='#ccc', columnspacing=0.8)

    # Annotate role bands
    ax.axvline(x=1.5, color='#aaa', linestyle=':', linewidth=0.8)
    ax.text(-0.05, -0.12, '■ Authorized (AUTH)',
            transform=ax.transAxes, fontsize=7, color=AUTH_COLOR)
    ax.text(0.45, -0.12, '■ Adversary (ADV)',
            transform=ax.transAxes, fontsize=7, color=ADV_COLOR)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'psnr_bar.png')
    plt.close(fig)
    print('  psnr_bar.png')


# ── Figure 3: Security gap summary ───────────────────────────────────────────

def fig_gap_summary():
    all_metrics = {k: load_metrics(k) for k, *_ in METHODS}

    auth_keys = [k for k, _, r in METHODS if r == 'auth']
    adv_keys  = [k for k, _, r in METHODS if r == 'adv']

    best_auth, best_adv, gaps = [], [], []
    for img in IMAGES:
        ba = max((get_psnr(all_metrics[k], img) or -99) for k in auth_keys)
        bd = max((get_psnr(all_metrics[k], img) or -99) for k in adv_keys)
        best_auth.append(ba)
        best_adv.append(bd)
        gaps.append(ba - bd)

    x = np.arange(len(IMAGES))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - w / 2, best_auth, w, label='Best authorized (AUTH)',
           color=AUTH_COLOR, alpha=0.88, edgecolor='white')
    ax.bar(x + w / 2, best_adv,  w, label='Best adversary (ADV)',
           color=ADV_COLOR,  alpha=0.88, edgecolor='white')

    for xi, gap, ba in zip(x, gaps, best_auth):
        sign  = '+' if gap >= 0 else ''
        color = GAP_COLOR if gap > 0 else '#c04040'
        ax.annotate(f'{sign}{gap:.1f} dB',
                    xy=(xi, max(ba, best_adv[xi]) + 0.3),
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in IMAGES], fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=9)
    ax.set_ylim(0, max(best_auth + best_adv) + 4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#ccc')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'gap_summary.png')
    plt.close(fig)
    print('  gap_summary.png')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating figures...')
    fig_panels()
    fig_psnr_bar()
    fig_gap_summary()
    print(f'\nAll figures saved to {FIG_DIR}')
    print('Now run:')
    print('  git add baselines/figures && git commit -m "Add baseline comparison figures" && git push')
