
# Load Modules
module purge
module load StdEnv/2023 

module load python/3.11.5 cuda/12.2 cudnn/8.9.5.29 opencv/4.9.0 mujoco/3.3.0


# Activate venv
source vdino/bin/activate

## Install (only first time)
pip install --no-index -r req_env_nibi.txt


# Inside Salloc
unset CUDA_VISIBLE_DEVICES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_API_KEY="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"

srun --unbuffered \
    --ntasks=2 \
    --gpu-bind=closest \
    python train_dual.py \
    --config-name train_dual_sall \
    env=jenga \
    env.num_workers=4 \
    num_hist=3 \
    training.batch_size=64 \
    hydra.launcher.timeout_min=120


### TO-DO (convert full lmdb for images and pkl for act/procep data, and move to tmp drive)
env.dataset.data_path=$SLURM_TMPDIR/data \
