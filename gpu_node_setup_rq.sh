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
# have to add this chief
# ==========================================
# 2. MOVE TO FAST LOCAL STORAGE ($SLURM_TMPDIR)
# ==========================================
echo "⚡ Copying data and code to local scratch ($SLURM_TMPDIR)..."
# cp -r $SOURCE_CODE_DIR $SLURM_TMPDIR/
# cp $DATA_TAR_PATH $SLURM_TMPDIR/
rsync -ah --progress "$SOURCE_CODE_DIR" "$SLURM_TMPDIR/"
# Add --exclude='wandb' and --exclude='outputs' to skip the junk!
# rsync -av \
#     --exclude='.git' \
#     --exclude='__pycache__' \
#     --exclude='wandb' \
#     --exclude='outputs' \
#     "$SOURCE_CODE_DIR/" "$SLURM_TMPDIR/dino_wm/"
#     # Add --exclude='wandb' and --exclude='outputs' to skip the junk!
# rsync -ah --progress "$DATA_TAR_PATH" "$SLURM_TMPDIR/"
rsync -ah --progress "$DATA_TAR_PATH" "$SLURM_TMPDIR/"
# cat "$DATA_TAR_PATH" > "$SLURM_TMPDIR/jenga_mujoco_noise.tar"
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
python create_lmdb_full.py --data_path $SLURM_TMPDIR/jenga_mujoco_noise
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
     num_action_repeat=1 \
     num_proprio_repeat=1 \
     frameskip=5 \
     ckpt_base_path=$HOME/links/scratch
    #  hydra.run.dir=$HOME/links/scratch/outputs/2026-03-10/18-04-17 \
    #  hydra.launcher.timeout_min=120
echo "✅ Script complete."