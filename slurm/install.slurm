#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --output="/projects/bcyd/ywang41/dino_wm/slurm/outputs/%x/%j.out"
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --account=bcyd-delta-cpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

export WANDB_API_KEY=65a3cb43423c6ede326b59f04363f333b5ef5f87
export WANDB_DIR='/projects/bcyd/ywang41/dino_wm/wandb'
export WANDB_DATA_DIR='/projects/bcyd/ywang41/dino_wm/wandb/share'
export HYDRA_FULL_ERROR=1
export DATASET_DIR='/scratch/bcyd/ywang41/diffusion-forcing/data/sim_aloha/pusht_0407/'

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
pip cache purge
cd /projects/bcyd/ywang41/diffusion-forcing
conda env remove --name diffusion-forcing -y
mamba env create -f conda_env.yaml
conda activate diffusion-forcing
pip install -y pytorch3d==0.7.8 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
pip uninstall -y av
