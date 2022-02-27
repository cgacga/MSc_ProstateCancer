#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=pca_slurm_setup
#SBATCH --output=pca_slurm_setup.out
 
# Set up environment
uenv avail
uenv list
uenv
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda create -f conda_env.yaml -y
