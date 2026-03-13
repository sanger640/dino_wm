#!/bin/bash

echo "🌐 Starting pre-trained weights download on the login node..."

# 1. Define and create the shared cache directory
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"
echo "📁 Cache directory ready: $CACHE_DIR"

# 2. Download LPIPS VGG weights
echo "⬇️ Downloading LPIPS VGG weights..."
if [ ! -f "$CACHE_DIR/vgg.pth" ]; then
    wget -O "$CACHE_DIR/vgg.pth" https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth
else
    echo "✅ LPIPS VGG already exists."
fi

# 3. Download Torchvision VGG16 weights
echo "⬇️ Downloading Torchvision VGG16 weights..."
if [ ! -f "$CACHE_DIR/vgg16-397923af.pth" ]; then
    wget -O "$CACHE_DIR/vgg16-397923af.pth" https://download.pytorch.org/models/vgg16-397923af.pth
else
    echo "✅ Torchvision VGG16 already exists."
fi

# 4. Download DinoV2 weights & repository code using PyTorch Hub
# (Using Python here ensures the GitHub repo code is also downloaded to ~/.cache/torch/hub)
echo "⬇️ Downloading DinoV2 weights and codebase..."
python -c "
import torch
import warnings
warnings.filterwarnings('ignore') # Suppress xFormers warnings
try:
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    print('✅ DinoV2 download successful.')
except Exception as e:
    print(f'❌ Error downloading DinoV2: {e}')
"

echo "------------------------------------------------"
echo "🔍 Verification: Checking files in $CACHE_DIR..."
ls -lh "$CACHE_DIR"
echo "------------------------------------------------"
echo "🎉 All done! You can now request your salloc session and run the training script."