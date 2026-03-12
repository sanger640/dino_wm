
# Load Modules
module purge
module load StdEnv/2023 

module load python/3.11.5 cuda/12.2 cudnn/8.9.5.29 opencv/4.9.0 mujoco/3.3.0


# Activate venv
source vdino/bin/activate

## Create venv
virtualenv vdino


## Install (only first time)
pip install --no-index -r req_env_nibi.txt

# SAlloc Cmd:
salloc --account=def-soojeon_gpu        --nodes=1        --gres=gpu:h100:4        --cpus-per-task=32        --mem=480G        --time=07:00:00


# Inside Salloc
unset CUDA_VISIBLE_DEVICES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_API_KEY="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"

export WANDB_MODE=offline

srun --unbuffered \
    --ntasks=4 \
    --gpu-bind=closest \
    python train_dual.py \
    --config-name train_dual_sall \
    env=jenga \
    env.num_workers=4 \
    num_hist=3 \
    training.batch_size=128 \
    hydra.launcher.timeout_min=120

srun --unbuffered \
    --ntasks=4 \
    --gpu-bind=closest \
    python train_dual.py \
    --config-name train_dual_sall \
    env=jenga \
    env.num_workers=4 \
    env.dataset.data_path = $SLURM_TMPDIR/data \
    num_hist=3 \
    training.batch_size=128 \
    hydra.launcher.timeout_min=120

## For roq (and narval)
srun --unbuffered \
     --ntasks=4 \
     --cpus-per-task=8 \
     python train_dual.py \
     --config-name train_dual_sall \
     env=jenga \
     env.num_workers=4 \
     env.dataset.data_path=$SLURM_TMPDIR/jenga_mujoco_noise \
     num_hist=3 \
     training.batch_size=32 \
     ckpt_base_path=$HOME/links/scratch \
     hydra.launcher.timeout_min=120

    env.dataset.data_path = $SLURM_TMPDIR/jenga_mujoco_noise \


### TO-DO (convert full lmdb for images and pkl for act/procep data, and move to tmp drive)
env.dataset.data_path=$SLURM_TMPDIR/data \


- gpu bind can have problem on narval and rorqual
- node number, cpu per task issue
- move everything slurm_tmpdir, still seeing if its good
- checkpoint in scratch tho?
- download flipping transformers and vgg before moving to tmp dir in narval/ror
        export TORCH_HUB_OFFLINE=1
        export OMP_NUM_THREADS=8
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8


- thread issues mate
export OMP_NUM_THREADS=8
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8


### Roq

#### Login script
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


### GPU Node Script
#!/bin/bash

# ==========================================
# 0. DEFINE YOUR PATHS (Edit these if needed)
# ==========================================
# Assuming your code and tar file are in your scratch or home dir
SOURCE_CODE_DIR="$HOME/links/scratch/dino_wm"
DATA_TAR_PATH="$HOME/links/scratch/jenga_mujoco_noise.tar"
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"

echo "🚀 Starting full setup pipeline..."

# ==========================================
# 1. PRE-FETCH WEIGHTS (Offline Cache Setup)
# ==========================================
echo "📦 Checking local weight cache in $CACHE_DIR..."
mkdir -p $CACHE_DIR

# Download LPIPS VGG if missing
if [ ! -f "$CACHE_DIR/vgg.pth" ]; then
    echo "Downloading LPIPS vgg.pth..."
    wget -q --timeout=10 -O $CACHE_DIR/vgg.pth https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth
fi

# Download Torchvision VGG16 if missing
if [ ! -f "$CACHE_DIR/vgg16-397923af.pth" ]; then
    echo "Downloading VGG16..."
    wget -q --timeout=10 -O $CACHE_DIR/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
fi

# Note: dinov2_vits14_pretrain.pth should already be there from your python -c command earlier.

# ==========================================
# 2. MOVE TO FAST LOCAL STORAGE ($SLURM_TMPDIR)
# ==========================================
echo "⚡ Copying data and code to local scratch ($SLURM_TMPDIR)..."
cp -r $SOURCE_CODE_DIR $SLURM_TMPDIR/
cp $DATA_TAR_PATH $SLURM_TMPDIR/

echo "🗜️ Extracting dataset..."
cd $SLURM_TMPDIR
tar -xf jenga_mujoco_noise.tar

# Move into the working directory
cd $SLURM_TMPDIR/dino_wm

# ==========================================
# 3. ENVIRONMENT & MODULE SETUP
# ==========================================
echo "🧹 Purging modules and loading dependencies..."
module purge
module load StdEnv/2023 
module load python/3.11.5 cuda/12.2 cudnn/8.9.5.29 opencv/4.9.0 mujoco/3.3.0

echo "🐍 Creating and activating virtual environment..."
virtualenv vdino
source vdino/bin/activate

echo "📦 Installing packages..."
pip install --no-index -r req_env_nibi.txt

# ==========================================
# 4. DATA PREPARATION
# ==========================================
echo "🗄️ Building LMDB database..."
# Assuming create_lmdb_full.py is configured to read from $SLURM_TMPDIR/jenga_mujoco_noise
python create_lmdb_full.py

# ==========================================
# 5. DISTRIBUTED TRAINING VARIABLES
# ==========================================
echo "🌐 Configuring distributed environment..."
unset CUDA_VISIBLE_DEVICES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# NCCL Error handling (including the new PyTorch 2.4+ variable to avoid the warning)
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1

# ==========================================
# 6. OFFLINE & THREADING FIXES (Crucial!)
# ==========================================
# Prevent "calling home" hangs
export TORCH_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Prevent "libblis: A different number of threads" crashes
export OMP_NUM_THREADS=8
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8



# ==========================================
# 7. LAUNCH TRAINING
# ==========================================
echo "🔥 Launching 4-GPU Training via srun..."
srun --unbuffered \
     --ntasks=4 \
     --cpus-per-task=8 \
     python train_dual.py \
     --config-name train_dual_sall \
     env=jenga \
     env.num_workers=4 \
     env.dataset.data_path=$SLURM_TMPDIR/jenga_mujoco_noise \
     num_hist=3 \
     training.batch_size=32 \
     ckpt_base_path=$HOME/links/scratch \
     hydra.launcher.timeout_min=120

echo "✅ Script complete."