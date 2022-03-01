#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=05:15:00
#SBATCH --job-name=pca_slurm_setup
#SBATCH --output=pca_slurm_setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda env create -f conda_env.yaml
