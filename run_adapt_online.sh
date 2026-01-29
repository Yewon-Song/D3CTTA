#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate d3ctta

# Run online adaptation for D3CTTA method for SemanticSTF
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-stf/nusc2stf.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-stf/sk2stf.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-stf/synth2stf.yaml



# Run online adaptation for D3CTTA method for SemanticKITTI-C
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-kitti-c/nusc2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-kitti-c/sk2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file configs/adaptation/test-kitti-c/synth2kitti.yaml
echo "All adaptation and evaluation processes are completed."
