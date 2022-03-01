#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=05:15:00
#SBATCH --job-name=setup_env
#SBATCH --output=../log/%x_%j.out
#SBATCH --error=../log/%x_%j.err
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda env create -f conda_env.yaml