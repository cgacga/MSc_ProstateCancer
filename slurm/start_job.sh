#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=15:15:00
#SBATCH --job-name=job_bs2_500_u3_vgg
#SBATCH --output=../logs/jobs/%j_%x.out
# test #SBATCH --output=../logs/job_%x/%j_%x.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate pca_env
#pca_env_piptf
#pca_env_tf
# pca_env_piptf
# Run the Python script that uses the GPU

# to_param_list () {
#     declare -n outlist=$1
#     declare -n inhash=$2

#     for param in "${!inhash[@]}"; do
#         #outlist+=( "--$param=${inhash[$param]}" )
#         outlist+=( "$param=${inhash[$param]}" )
#     done
# }

# declare -A options
# options[dog]="Bark"
# options[wolf]=123

# to_param_list kwargs options

python3 -u ../code/main.py
#"${kwargs[@]}"
#kwargs
#device=foo format=1

