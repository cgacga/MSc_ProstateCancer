#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=20:15:00
#SBATCH --job-name=job_a
#SBATCH --array=0-24%2
#SBATCH --output=../logs/jobs/%j_%A_%a.out

 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate pca_env


python3 -u ../code/main.py 

# to_param_list () {
#     declare -n outlist=$1
#     declare -n inhash=$2

#     for param in "${!inhash[@]}"; do
#         #outlist+=( "--$param=${inhash[$param]}" )
#         outlist+=( "$param:${inhash[$param]}" )
#     done
# }


# to_param_list2 () {
#     declare -n outlist=$1
#     declare -n inhash=$2

#     for param in "${!inhash[@]}"; do
#         #outlist+=( "--$param=${inhash[$param]}" )
#         outlist+=( "$param=${inhash[$param]}" )
#     done
# }

# declare -A ADC
# ADC[dim]="(32,128,96)"
# ADC[batchsize]=32

# declare -A t2tsetra
# t2tsetra[tags]=None
# t2tsetra[batchsize]=2

# to_param_list adc_dict ADC
# to_param_list t2tsetra_dict t2tsetra

# declare -A params
# params[adc_dict]="${adc_dict[@]}"
# params[t2tsetra_dict]="${t2tsetra_dict[@]}"


# declare -A params2
# params2[adc_dict]=${adc_dict[@]}
# params2[t2tsetra_dict]=${t2tsetra_dict[@]}

# to_param_list2 kwargs params

#python3 -u ../code/main.py ${params[@]} "${params[@]}" ${params2[@]} "${params2[@]}" ${kwargs[@]} "${kwargs[@]}" 
#python3 -u ../code/main.py ${kwargs[@]}
#"${kwargs[@]}"
