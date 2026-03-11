#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Setup script — run once on the server before any baseline evaluation.
# Expected: Python 3.9+, pip, git, wget/curl, CUDA GPU recommended.
# ──────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

echo "=== [1/4] Installing Python dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install basicsr einops timm scipy scikit-image matplotlib numpy tqdm

echo "=== [2/4] Cloning third-party repos ==="
mkdir -p third_party

# DPIR — Deep Plug-and-Play Image Restoration (non-blind, knows PSF)
if [ ! -d third_party/DPIR ]; then
    git clone --depth 1 https://github.com/cszn/DPIR third_party/DPIR
fi

# NAFNet — Simple Baselines for Image Restoration (blind adversary)
if [ ! -d third_party/NAFNet ]; then
    git clone --depth 1 https://github.com/megvii-research/NAFNet third_party/NAFNet
fi

# Restormer — Efficient Transformer for High-Res Restoration (blind adversary)
if [ ! -d third_party/Restormer ]; then
    git clone --depth 1 https://github.com/swz30/Restormer third_party/Restormer
fi

echo "=== [3/4] Downloading pretrained weights ==="
mkdir -p weights

# DPIR: DRUNet denoiser (grayscale, ~33 MB)
if [ ! -f weights/drunet_gray.pth ]; then
    wget -q --show-progress \
        "https://github.com/cszn/DPIR/releases/download/v1.0/drunet_gray.pth" \
        -O weights/drunet_gray.pth
fi

# NAFNet: GoPro deblurring checkpoint (~100 MB)
if [ ! -f weights/NAFNet-GoPro-width64.pth ]; then
    wget -q --show-progress \
        "https://github.com/megvii-research/NAFNet/releases/download/v0.0/NAFNet-GoPro-width64.pth" \
        -O weights/NAFNet-GoPro-width64.pth
fi

# Restormer: motion deblurring checkpoint (~165 MB)
if [ ! -f weights/motion_deblurring.pth ]; then
    wget -q --show-progress \
        "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth" \
        -O weights/motion_deblurring.pth
fi

echo "=== [4/4] Preparing test data ==="
cd ..
python baselines/prepare_data.py
cd baselines

echo ""
echo "Setup complete. Run the baselines with:"
echo "  python baselines/run_dpir.py"
echo "  python baselines/run_nafnet.py"
echo "  python baselines/run_restormer.py"
echo "  python baselines/collect_results.py"
