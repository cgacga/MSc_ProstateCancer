#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=15:15:00
#SBATCH --job-name=job_bs2_tf
#SBATCH --output=../logs/%x_%j.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate pca_env_tf
# Run the Python script that uses the GPU
python3 -u ../code/main.py