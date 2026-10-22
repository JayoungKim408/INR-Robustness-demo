#!/usr/bin/env bash
set -x

device=$1
dataset=$2


for data in ${dataset}
do
    for hidden_layers in 0 3 
    do
        mkdir -p src/inr/logs/make_inrs/d${hidden_layers}_w32_s30
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers ${hidden_layers} \
        --hidden_features 32 \
        --first_omega_0 30 \
        --hidden_omega_0 30 \
        --init_mode fixed_init > src/inr/logs/make_inrs/d${hidden_layers}_w32_s30/${data}_fixed_init.log  
    done
    for hidden_features in 16 64
    do
        mkdir -p src/inr/logs/make_inrs/d1_w${hidden_features}_s30
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers 1 \
        --hidden_features ${hidden_features} \
        --first_omega_0 30 \
        --hidden_omega_0 30 \
        --init_mode fixed_init > src/inr/logs/make_inrs/d1_w${hidden_features}_s30/${data}_fixed_init.log  
    done

    for hidden_omega_0 in 10 60 
    do
        mkdir -p src/inr/logs/make_inrs/d1_w32_s${hidden_omega_0}
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers 1 \
        --hidden_features 32 \
        --first_omega_0 ${hidden_omega_0} \
        --hidden_omega_0 ${hidden_omega_0} \
        --init_mode fixed_init > src/inr/logs/make_inrs/d1_w32_s${hidden_omega_0}/${data}_fixed_init.log  
    done
done

for data in ${dataset}
do
    for hidden_layers in 0 3 
    do
        mkdir -p src/inr/logs/make_inrs/d${hidden_layers}_w32_s30
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers ${hidden_layers} \
        --hidden_features 32 \
        --first_omega_0 30 \
        --hidden_omega_0 30 \
        --init_mode random_init \
        --bias_zero_init True \
        --is_first False > src/inr/logs/make_inrs/d${hidden_layers}_w32_s30/${data}_random_init_bias_zero_no_is_first.log  
    done

    for hidden_features in 16 64
    do
        mkdir -p src/inr/logs/make_inrs/d1_w${hidden_features}_s30
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers 1 \
        --hidden_features ${hidden_features} \
        --first_omega_0 30 \
        --hidden_omega_0 30 \
        --init_mode random_init \
        --bias_zero_init True \
        --is_first False > src/inr/logs/make_inrs/d1_w${hidden_features}_s30/${data}_random_init_bias_zero_no_is_first.log  
    done

    for hidden_omega_0 in 10 60 
    do
        mkdir -p src/inr/logs/make_inrs/d1_w32_s${hidden_omega_0}
        CUDA_VISIBLE_DEVICES=$device python src/inr/make_inr_init.py \
        --dataset_name ${data} \
        --hidden_layers 1 \
        --hidden_features 32 \
        --first_omega_0 ${hidden_omega_0} \
        --hidden_omega_0 ${hidden_omega_0} \
        --init_mode random_init \
        --bias_zero_init True \
        --is_first False > src/inr/logs/make_inrs/d1_w32_s${hidden_omega_0}/${data}_random_init_bias_zero_no_is_first.log  
    done
done
