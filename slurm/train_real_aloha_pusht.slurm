#!/bin/bash
#SBATCH --job-name="dino_real_pusht"
#SBATCH --output="/projects/bcyd/ywang41/dino_wm/slurm/outputs/%x/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcyd-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 1:00:00

export WANDB_API_KEY=65a3cb43423c6ede326b59f04363f333b5ef5f87
export WANDB_DIR='/projects/bcyd/ywang41/dino_wm/wandb'
export WANDB_DATA_DIR='/projects/bcyd/ywang41/dino_wm/wandb/share'
export HYDRA_FULL_ERROR=1
export DATASET_DIR='/work/hdd/bcyd/ywang41/diffusion-forcing/real_aloha/pusht_1000_1101/'

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate df_il_2
cd /projects/bcyd/ywang41/dino_wm
python train.py \
    --config-name \
    train.yaml \
    training.batch_size=32 \
    env=real_aloha \
    env.dataset.dataset_dir='/work/hdd/bcyd/ywang41/diffusion-forcing/data/real_aloha/pusht_1000_1101/' \
    frameskip=1 \
    num_hist=9
