#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuA100 
#SBATCH --time=15:15:00
#SBATCH --job-name=job_bs16_tf_args
#SBATCH --output=../logs/%j_%x.out
# test #SBATCH --output=../logs/job_%x/%j_%x.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate pca_env_tf
# Run the Python script that uses the GPU
# kwargs = {device:"foo",
#           format:1,
#           test:123,
#           string:"str"}

# declare -A kwargs
# kwargs[dog]="Bark"
# kwargs[wolf]="Howl"

python3 -u ../code/main.py 
#kwargs
#device=foo format=1

