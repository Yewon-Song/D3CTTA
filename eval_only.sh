#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate d3ctta

# Run eval_only for no adaptation baseline for SemanticSTF
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-stf/nusc2stf.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-stf/sk2stf.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-stf/synth2stf.yaml

# # Run eval_only for no adaptation baseline for SemanticKITTI-C
# CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-kitti-c/nusc2kitti.yaml
# CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-kitti-c/sk2kitti.yaml
# CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file configs/adaptation/test-kitti-c/synth2kitti.yaml

# echo "All evaluation processes are completed."
