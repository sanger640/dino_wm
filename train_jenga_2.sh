#!/bin/bash
#SBATCH --job-name=jenga_h3_b32
#SBATCH --account=def-soojeon          # Your compute account
#SBATCH --nodes=1                      # 1 Node
#SBATCH --ntasks-per-node=1         # 2 Tasks (1 per GPU)
#SBATCH --gres=gpu:h100:1           # 2 H100 GPUs
#SBATCH --cpus-per-task=8           # 8 CPUs per GPU
#SBATCH --mem=32G                     # Total Memory
#SBATCH --time=2:00:00                # Max run time (24 hours)
#SBATCH --output=%j.out                # Log file named by Job ID
#SBATCH --mail-user=a2budhwa@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.2 cudnn/8.9.5.29 opencv/4.9.0 mujoco/3.3.0
source /home/ali313/scratch/dino_wm/vdino/bin/activate

# --- DDP Networking ---
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# --- Performance Flags ---
export NCCL_ASYNC_ERROR_HANDLING=1
export OPENCV_FORK_HANDLING_IS_DANGEROUS=1
export WANDB_API_KEY="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"
# --- Launch Training ---
# Note: training.batch_size=64 means 32 per GPU
# srun --unbuffered python train_dual.py \
#     --config-name train_dual_sall \
#     num_hist=3 \
#     training.batch_size=64 \
#     env.num_workers=4 \
#     training.mixed_precision=bf16

# srun --unbuffered \
#     --ntasks=4 \
#     --gpu-bind=closest \
#     python train_dual.py \
#     --config-name train_dual_sall \
#     env=jenga \
#     env.num_workers=4 \
#     num_hist=3 \
#     training.batch_size=64

# srun --unbuffered \
#     --ntasks=2 \
#     --gpu-bind=closest \
#     python train_dual.py \
#     --config-name train_dual_sall \
#     env=jenga \
#     env.num_workers=4 \
#     num_hist=2 \
#     training.batch_size=64
    # concat_dim=0 \
    # action_emb_dim=384 \
    # proprio_emb_dim=384

srun --unbuffered \
    --ntasks=1 \
    --gpu-bind=closest \
    python train_dual.py \
    --config-name train_dual_sall \
    env=jenga \
    env.num_workers=4 \
    num_hist=3 \
    hydra.run.dir="/home/ali313/scratch/dino_wm/outputs/2026-03-09/05-11-07" \
    training.batch_size=8

## add in it to start from last run