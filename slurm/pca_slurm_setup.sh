#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=05:15:00
#SBATCH --job-name=slurm_setup
#SBATCH --output=../log/%x.%j.out
#SBATCH --error=../log/%x.%j.err
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda env create -f conda_env.yaml