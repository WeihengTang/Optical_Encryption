"""
MetaLens Demo — Optical Encryption via Spatially-Varying PSF

Runs three test images through:
  1. SV-PSF blurring (encryption)
  2. Non-blind Wiener deconvolution (authorized decoder, knows PSF)
  3. Blind Wiener deconvolution  (adversary, no PSF knowledge)

Saves per-image panels to results/ and prints a metrics table.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data, color, transform

from src.blur import apply_sv_psf
from src.non_blind_deblur import non_blind_deblur
from src.blind_deblur import blind_wiener, blind_rl
from src.metrics import compute_all

# ── config ────────────────────────────────────────────────────────────────────

PSF_PATH = "SV_PSF.npy"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

KERNEL_SIZE = 71      # PSF crop (96.6 % energy)
WIENER_REG = 5e-2     # non-blind regularisation
IMAGE_SIZE = 1008     # 7×144 px → each PSF patch is 144×144 (≫ 71 kernel)


# ── test images ───────────────────────────────────────────────────────────────

def load_test_images():
    imgs = {
        "camera":    data.camera().astype(np.float32) / 255.0,
        "astronaut": color.rgb2gray(data.astronaut()).astype(np.float32),
        "chelsea":   color.rgb2gray(data.chelsea()).astype(np.float32),
    }
    return {
        name: transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                               anti_aliasing=True).astype(np.float32)
        for name, img in imgs.items()
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def save_panel(name, orig, blurred, nb_rec, b_rec, metrics):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"MetaLens Encryption Demo  —  '{name}'", fontsize=13, y=1.01)

    panels = [
        (orig,    "Original"),
        (blurred, "Encrypted\n(SV-PSF blur)"),
        (nb_rec,  f"Non-blind (authorized)\nPSNR {metrics['non_blind']['PSNR']} dB"
                  f"  SSIM {metrics['non_blind']['SSIM']}"),
        (b_rec,   f"Blind (adversary)\nPSNR {metrics['blind']['PSNR']} dB"
                  f"  SSIM {metrics['blind']['SSIM']}"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out = RESULTS_DIR / f"{name}_panel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def run(blind_method: str = "wiener"):
    psf = np.load(PSF_PATH)
    images = load_test_images()
    all_metrics = {}

    print(f"\n{'Image':<12} {'Method':<20} {'PSNR (dB)':<12} {'SSIM'}")
    print("-" * 55)

    for name, img in images.items():
        # ── encrypt ──────────────────────────────────────────────────────────
        t0 = time.time()
        blurred = apply_sv_psf(img, psf, kernel_size=KERNEL_SIZE)
        t_blur = time.time() - t0

        # ── authorized decode ─────────────────────────────────────────────────
        t0 = time.time()
        nb_rec = non_blind_deblur(blurred, psf,
                                  kernel_size=KERNEL_SIZE, reg=WIENER_REG)
        t_nb = time.time() - t0

        # ── adversary attempt ─────────────────────────────────────────────────
        t0 = time.time()
        if blind_method == "rl":
            b_rec = blind_rl(blurred)
        else:
            b_rec = blind_wiener(blurred, psf_size=KERNEL_SIZE)
        t_b = time.time() - t0

        # ── metrics ───────────────────────────────────────────────────────────
        blurred_m  = compute_all(img, blurred)
        nb_m       = compute_all(img, nb_rec)
        b_m        = compute_all(img, b_rec)

        all_metrics[name] = {
            "blurred":   blurred_m,
            "non_blind": nb_m,
            "blind":     b_m,
            "times": {"blur": t_blur, "non_blind": t_nb, "blind": t_b},
        }

        for label, m in [("blurred", blurred_m), ("non-blind", nb_m), ("blind", b_m)]:
            print(f"  {name:<12} {label:<20} {m['PSNR']:<12} {m['SSIM']}")

        gap = round(nb_m["PSNR"] - b_m["PSNR"], 2)
        print(f"  {'':12} {'→ gap':<20} {gap:<12} dB advantage for authorized\n")

        save_panel(name, img, blurred, nb_rec, b_rec,
                   {"non_blind": nb_m, "blind": b_m})

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", choices=["wiener", "rl"], default="wiener",
                        help="Blind deblurring method (default: wiener)")
    args = parser.parse_args()
    run(blind_method=args.blind)
