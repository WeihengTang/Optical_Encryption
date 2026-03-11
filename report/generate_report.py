"""
Generates a self-contained HTML report for the MetaLens project.
All images are embedded as base64 so the file is portable.
"""

import base64
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from skimage import data, color, transform
from src.blur import apply_sv_psf
from src.non_blind_deblur import non_blind_deblur
from src.blind_deblur import blind_wiener
from src.metrics import compute_all

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMAGE_SIZE = 1008
KERNEL_SIZE = 71
WIENER_REG = 5e-2


# ── helpers ───────────────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── figure 1: PSF grid ────────────────────────────────────────────────────────

def make_psf_figure(psf):
    R, C = psf.shape[:2]
    fig, axes = plt.subplots(R, C, figsize=(10, 10))
    fig.suptitle("Metasurface SV-PSF Grid (7 × 7 positions, each 101 × 101 px)",
                 fontsize=11)
    vmax = psf.max()
    for i in range(R):
        for j in range(C):
            ax = axes[i][j]
            ax.imshow(np.log1p(psf[i, j] / vmax * 1e6), cmap="inferno",
                      origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"({i},{j})", fontsize=6, pad=1)
    plt.tight_layout()
    return fig


# ── figure 2: optimization pipeline diagram ──────────────────────────────────

def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    BOX = dict(boxstyle="round,pad=0.5", linewidth=1.5)
    TXT = dict(ha="center", va="center", fontsize=9, fontweight="bold")
    ARR = dict(arrowstyle="-|>", color="#cdd6f4", lw=1.5,
               connectionstyle="arc3,rad=0.0")

    def box(x, y, w, h, label, sublabel="", color="#313244", tcolor="#cdd6f4"):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color, edgecolor=tcolor,
                                       linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + (0.18 if sublabel else 0), label,
                color=tcolor, fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=3)
        if sublabel:
            ax.text(x, y - 0.28, sublabel,
                    color="#a6adc8", fontsize=7, ha="center", va="center",
                    zorder=3, style="italic")

    def arrow(x0, y0, x1, y1, label="", color="#cdd6f4"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.5, mutation_scale=12), zorder=2)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my + 0.25, label, color="#a6adc8", fontsize=7,
                    ha="center", va="center", zorder=3)

    # ── Phase banner ──────────────────────────────────────────────────────────
    for px, ph, ptxt, pc in [
        (4.0, 6.6, "PHASE 1 — Train decoders (PSF fixed)", "#45475a"),
        (12.0, 6.6, "PHASE 2 — Optimise PSF (decoders fixed)", "#45475a"),
    ]:
        ax.text(px, ph, ptxt, color="#a6adc8", fontsize=8, ha="center",
                va="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#313244",
                          ec="#6c7086", lw=1))

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    box(1.0, 5.0, 1.6, 0.8,  "Clean Image x",  "training set",   "#1e66f5", "#cdd6f4")
    box(2.8, 5.0, 1.5, 0.8,  "Blur  Bθ(x)",    "SV-PSF encrypt", "#8839ef", "#cdd6f4")
    box(4.8, 6.0, 1.8, 0.8,  "D_nb (authorized)", "knows PSF θ",  "#40a02b", "#cdd6f4")
    box(4.8, 4.0, 1.8, 0.8,  "D_b  (adversary)",  "no PSF",       "#d20f39", "#cdd6f4")
    box(6.8, 6.0, 1.4, 0.8,  "L_nb",  "−PSNR(x̂,x)",              "#fe640b", "#cdd6f4")
    box(6.8, 4.0, 1.4, 0.8,  "L_b",   "+PSNR(x̃,x)",              "#fe640b", "#cdd6f4")

    arrow(1.8, 5.0, 2.05, 5.0)
    arrow(3.55, 5.0, 3.9, 5.4, "y = Bθ(x)")
    arrow(3.55, 5.0, 3.9, 4.6)
    arrow(5.7, 6.0, 6.1, 6.0, "x̂")
    arrow(5.7, 4.0, 6.1, 4.0, "x̃")

    # Phase 1 loss combine
    box(8.0, 5.0, 1.4, 0.8, "Total Loss",
        "L = L_nb + λ·L_b", "#df8e1d", "#cdd6f4")
    arrow(7.5, 6.0, 7.7, 5.3)
    arrow(7.5, 4.0, 7.7, 4.7)
    ax.text(8.0, 3.2, "Update D_nb & D_b\n(PSF θ frozen)", color="#a6adc8",
            fontsize=7.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#313244", ec="#6c7086"))
    arrow(8.0, 4.6, 8.0, 3.5)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    box(10.2, 5.0, 1.7, 0.8, "PSF Generator",
        "metasurface params φ", "#1e66f5", "#cdd6f4")
    box(12.1, 5.0, 1.4, 0.8, "Blur  Bφ(x)",
        "new candidate PSF",    "#8839ef", "#cdd6f4")
    box(13.8, 6.0, 1.6, 0.8, "D_nb (frozen)",
        "↑ PSNR goal",          "#40a02b", "#cdd6f4")
    box(13.8, 4.0, 1.6, 0.8, "D_b  (frozen)",
        "↓ PSNR goal",          "#d20f39", "#cdd6f4")
    box(15.4, 5.0, 1.0, 0.8, "ΔL",
        "gap ↑",                "#df8e1d", "#cdd6f4")

    arrow(10.95, 5.0, 11.4, 5.0)
    arrow(12.8, 5.0, 13.0, 5.4)
    arrow(12.8, 5.0, 13.0, 4.6)
    arrow(14.6, 6.0, 14.9, 5.3)
    arrow(14.6, 4.0, 14.9, 4.7)

    ax.text(12.8, 3.2, "Gradient ∂ΔL/∂φ → update metasurface params",
            color="#a6adc8", fontsize=7.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#313244", ec="#6c7086"))
    arrow(12.8, 4.6, 12.8, 3.5)

    # ── Loop arrow ────────────────────────────────────────────────────────────
    ax.annotate("", xy=(9.3, 5.3), xytext=(8.7, 5.3),
                arrowprops=dict(arrowstyle="-|>", color="#f5c2e7",
                                lw=2, mutation_scale=14,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(9.0, 5.65, "Iterate", color="#f5c2e7", fontsize=8,
            ha="center", va="center", fontweight="bold")

    # ── Physical constraint note ──────────────────────────────────────────────
    ax.text(12.0, 1.6,
            "Physical constraint: φ ∈ realizable metasurface parameter space\n"
            "(differentiable PSF generator from co-worker's model)",
            color="#a6adc8", fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e2e",
                      ec="#6c7086", lw=1.2, linestyle="--"))

    ax.set_title("MetaLens PSF Encryption — Adversarial Optimization Loop",
                 color="#cdd6f4", fontsize=12, pad=10)
    return fig


# ── figure 3: results panel ───────────────────────────────────────────────────

def make_results_summary(results_paths, metrics):
    n = len(results_paths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, results_paths.items()):
        img_data = plt.imread(path)
        ax.imshow(img_data)
        ax.axis("off")
        m = metrics[name]
        gap = round(m["non_blind"]["PSNR"] - m["blind"]["PSNR"], 2)
        ax.set_title(f"{name}  |  gap = {gap} dB", fontsize=10)
    plt.tight_layout()
    return fig


# ── run experiments ───────────────────────────────────────────────────────────

def run_experiments():
    psf = np.load(ROOT / "SV_PSF.npy")
    test_images_raw = {
        "camera":    data.camera().astype(np.float32) / 255.0,
        "astronaut": color.rgb2gray(data.astronaut()).astype(np.float32),
        "chelsea":   color.rgb2gray(data.chelsea()).astype(np.float32),
    }
    images = {
        name: transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                               anti_aliasing=True).astype(np.float32)
        for name, img in test_images_raw.items()
    }

    metrics = {}
    panels = {}

    for name, img in images.items():
        blurred = apply_sv_psf(img, psf, kernel_size=KERNEL_SIZE)
        nb_rec  = non_blind_deblur(blurred, psf,
                                   kernel_size=KERNEL_SIZE, reg=WIENER_REG)
        b_rec   = blind_wiener(blurred, psf_size=KERNEL_SIZE)

        metrics[name] = {
            "blurred":   compute_all(img, blurred),
            "non_blind": compute_all(img, nb_rec),
            "blind":     compute_all(img, b_rec),
        }

        path = RESULTS / f"{name}_panel.png"
        panels[name] = path

        if not path.exists():
            fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
            fig.suptitle(f"MetaLens Demo — '{name}'", fontsize=12, y=1.01)
            for ax, (arr, ttl) in zip(axs, [
                (img,     "Original"),
                (blurred, "Encrypted (SV-PSF)"),
                (nb_rec,  f"Non-blind (auth.)\n{metrics[name]['non_blind']['PSNR']} dB"),
                (b_rec,   f"Blind (adversary)\n{metrics[name]['blind']['PSNR']} dB"),
            ]):
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
                ax.set_title(ttl, fontsize=10)
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

    return psf, metrics, panels


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MetaLens — Optical Encryption via SV-PSF</title>
<style>
  :root {{
    --bg: #1e1e2e; --surface: #313244; --text: #cdd6f4;
    --sub: #a6adc8; --accent: #1e66f5; --green: #40a02b;
    --red: #d20f39; --yellow: #df8e1d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: "Segoe UI", Arial, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: 0.3rem; }}
  h2 {{ font-size: 1.3rem; color: var(--yellow); margin: 2rem 0 0.7rem; }}
  h3 {{ font-size: 1.05rem; color: var(--green); margin: 1rem 0 0.4rem; }}
  p  {{ color: var(--sub); line-height: 1.7; margin-bottom: 0.8rem; }}
  .meta {{ color: var(--sub); font-size: 0.85rem; margin-bottom: 2rem; }}
  .badge {{ display:inline-block; padding:0.2rem 0.6rem;
            border-radius:9999px; font-size:0.75rem; font-weight:600;
            margin-right:0.3rem; }}
  .b-blue  {{ background:#1e66f5; color:#fff; }}
  .b-green {{ background:#40a02b; color:#fff; }}
  .b-red   {{ background:#d20f39; color:#fff; }}
  .b-purp  {{ background:#8839ef; color:#fff; }}
  figure {{ margin: 1.5rem 0; text-align:center; }}
  figure img {{ max-width:100%; border-radius:8px;
                border:1px solid var(--surface); }}
  figcaption {{ color:var(--sub); font-size:0.82rem; margin-top:0.5rem; }}
  table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
  th,td {{ padding:0.55rem 0.9rem; text-align:center;
           border:1px solid var(--surface); }}
  th    {{ background:var(--surface); color:var(--accent); }}
  td    {{ color:var(--sub); }}
  .gap  {{ color:var(--yellow); font-weight:700; }}
  .nb   {{ color:var(--green); }}
  .bl   {{ color:var(--red); }}
  code  {{ background:var(--surface); padding:0.15rem 0.4rem;
           border-radius:4px; font-size:0.85rem; }}
  .card {{ background:var(--surface); border-radius:10px;
           padding:1.2rem 1.6rem; margin:1rem 0; }}
  .roadmap li {{ color:var(--sub); line-height:1.9; margin-left:1.2rem; }}
  .roadmap li::marker {{ color:var(--accent); }}
</style>
</head>
<body>

<h1>MetaLens</h1>
<p class="meta">
  <span class="badge b-blue">Optical Encryption</span>
  <span class="badge b-purp">Metasurface PSF</span>
  <span class="badge b-green">Non-blind Deblurring</span>
  <span class="badge b-red">Blind Deblurring</span>
  &nbsp; Generated {date}
</p>

<h2>1. Motivation &amp; Overview</h2>
<p>
  We propose using a physically fabricated <strong>metasurface</strong>
  as an optical encryption device.  The metasurface imposes a complex,
  spatially-varying point-spread function (SV-PSF) on any scene it images.
  An <em>authorized receiver</em> who knows the exact PSF can apply
  non-blind deconvolution to recover the original image.
  An <em>adversary</em> without PSF knowledge — even using state-of-the-art
  blind deblurring — cannot.
</p>
<p>
  The long-term goal is to <strong>optimize the SV-PSF parameters</strong>
  (via a differentiable metasurface generator) to maximise this
  <em>security gap</em> while keeping authorized decoding high quality.
</p>

<h2>2. The Metasurface PSF</h2>
<p>
  The provided PSF <code>SV_PSF.npy</code> is shaped
  <code>(7, 7, 101, 101)</code> — a 7 × 7 spatial grid, each position
  carrying a 101 × 101 kernel.  The kernels exhibit a characteristic
  <strong>ring/donut structure</strong> that rotates and gains asymmetry
  towards the field corners (off-axis optical aberrations typical of
  meta-optics).
</p>
<figure>
  <img src="data:image/png;base64,{psf_b64}" alt="PSF grid">
  <figcaption>Fig. 1 — SV-PSF grid (log-scale intensity, inferno colormap).
  Each sub-image is one of the 49 spatially distinct PSFs.</figcaption>
</figure>

<h2>3. Methods</h2>

<div class="card">
<h3>3.1  Encryption — SV-PSF Blurring</h3>
<p>
  The image is divided into a 7 × 7 grid of patches.
  Each patch is convolved with its corresponding PSF kernel
  (cropped to 71 × 71, capturing 96.6 % of the total kernel energy)
  via fast FFT-based convolution (<code>scipy.signal.fftconvolve</code>).
  The patches are reassembled to produce the encrypted image.
</p>
</div>

<div class="card">
<h3>3.2  Authorized Decoder — Non-blind Wiener Deconvolution</h3>
<p>
  The receiver applies patch-wise <strong>Wiener deconvolution</strong>
  with the <em>known</em> SV-PSF:
</p>
<p style="font-family:monospace; color:#f5c2e7; margin:0.5rem 0 0.5rem 1.5rem;">
  X̂(ω) = H*(ω) / (|H(ω)|² + λ) · Y(ω)
</p>
<p>
  where <code>λ = 5 × 10⁻²</code> is the regularisation constant,
  and the kernel is zero-padded and circular-shifted to place its
  centre at array position (0,0) before the DFT — ensuring no phase offset
  in the recovered image.
</p>
</div>

<div class="card">
<h3>3.3  Adversary — Blind Wiener Deconvolution</h3>
<p>
  The adversary applies scikit-image's
  <strong>Unsupervised Wiener-Hunt</strong> deconvolution
  (Gibbs sampler under a Gaussian image prior) to the <em>entire</em>
  blurred image, assuming a single spatially-uniform blur.
  The true SV-PSF is unknown; the algorithm initialises with a small
  Gaussian and iteratively refines both the PSF estimate and the image.
  Because the real blur is <em>spatially varying</em> and ring-shaped,
  the blind estimator converges to a wrong, smeared kernel.
</p>
</div>

<h2>4. Preliminary Results</h2>
<p>
  Experiments on three standard test images (all resized to 1008 × 1008).
  Metrics: PSNR (dB) and SSIM against the clean original.
</p>

<table>
  <tr>
    <th>Image</th>
    <th>Encrypted PSNR</th>
    <th class="nb">Non-blind PSNR</th>
    <th class="bl">Blind PSNR</th>
    <th class="gap">Gap (dB) ↑</th>
    <th class="nb">Non-blind SSIM</th>
    <th class="bl">Blind SSIM</th>
  </tr>
  {table_rows}
</table>

{panels_html}

<h2>5. Proposed Optimization Framework</h2>
<p>
  Once the co-worker's <strong>differentiable PSF generator</strong>
  is available, we will run the following adversarial training loop
  to find a PSF that maximises the authorized vs. adversary performance gap.
</p>
<figure>
  <img src="data:image/png;base64,{pipeline_b64}" alt="Optimization pipeline">
  <figcaption>Fig. — Adversarial PSF optimization loop.
  Phase 1 trains the two decoders on the current PSF.
  Phase 2 back-propagates through both decoders to update the metasurface
  parameters φ, maximising the gap ΔL = L_nb − λ·L_b.</figcaption>
</figure>

<div class="card">
<h3>Optimization Objective</h3>
<p style="font-family:monospace; color:#f5c2e7; margin:0.4rem 0;">
  max<sub>φ</sub>  PSNR(D_nb(B_φ(x), φ), x) − λ · PSNR(D_b(B_φ(x)), x)
</p>
<p>
  subject to φ being physically realizable by the metasurface model.
  The gradient ∂ΔL/∂φ is computed end-to-end through the differentiable
  blur operator and both frozen decoder networks.
</p>
</div>

<h2>6. Roadmap</h2>
<ol class="roadmap">
  <li><strong>Current (today)</strong> — Baseline pipeline on existing PSF;
      non-blind Wiener vs. blind Wiener-Hunt. ✅</li>
  <li><strong>This week</strong> — Integrate co-worker's differentiable
      PSF generator; establish training set (DIV2K).</li>
  <li><strong>Next week</strong> — Replace Wiener with deep non-blind decoder
      (DPIR / USRNet); replace blind baseline with NAFNet / Restormer.</li>
  <li><strong>Month 2</strong> — Full adversarial PSF optimization loop;
      perceptual loss (LPIPS) in addition to PSNR.</li>
  <li><strong>Month 3</strong> — Ablation study; physical fabrication
      candidate PSF; write-up.</li>
</ol>

<h2>7. Dependencies</h2>
<p>
  <code>numpy scipy scikit-image matplotlib torch</code>
  — see <code>requirements.txt</code>.
  No pretrained weights required at this stage.
</p>

</body>
</html>
"""


def build_report():
    psf, metrics, panels = run_experiments()

    # PSF figure
    psf_fig = make_psf_figure(psf)
    psf_b64 = fig_to_b64(psf_fig)

    # Pipeline diagram
    pipe_fig = make_pipeline_diagram()
    pipe_b64 = fig_to_b64(pipe_fig)

    # Table rows
    rows = ""
    for name, m in metrics.items():
        gap = round(m["non_blind"]["PSNR"] - m["blind"]["PSNR"], 2)
        rows += (
            f"<tr><td>{name}</td>"
            f"<td>{m['blurred']['PSNR']}</td>"
            f"<td class='nb'>{m['non_blind']['PSNR']}</td>"
            f"<td class='bl'>{m['blind']['PSNR']}</td>"
            f"<td class='gap'>+{gap}</td>"
            f"<td class='nb'>{m['non_blind']['SSIM']}</td>"
            f"<td class='bl'>{m['blind']['SSIM']}</td></tr>\n"
        )

    # Panel figures
    panels_html = ""
    for name, path in panels.items():
        b64 = img_to_b64(path)
        gap = round(metrics[name]["non_blind"]["PSNR"] - metrics[name]["blind"]["PSNR"], 2)
        panels_html += (
            f'<figure><img src="data:image/png;base64,{b64}" alt="{name}">'
            f"<figcaption>Fig. — '{name}' | "
            f"Non-blind {metrics[name]['non_blind']['PSNR']} dB vs "
            f"Blind {metrics[name]['blind']['PSNR']} dB | "
            f"Gap = <strong>+{gap} dB</strong></figcaption></figure>\n"
        )

    html = HTML.format(
        date=time.strftime("%Y-%m-%d"),
        psf_b64=psf_b64,
        pipeline_b64=pipe_b64,
        table_rows=rows,
        panels_html=panels_html,
    )

    out = ROOT / "report" / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report saved to {out}")
    return out


if __name__ == "__main__":
    build_report()
