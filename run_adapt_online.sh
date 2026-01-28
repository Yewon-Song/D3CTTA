#!/bin/bash

# Run online adaptation for D3CTTA method for SemanticSTF
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/nusc2stf.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/sk2stf.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/synth2stf.yaml



# Run online adaptation for D3CTTA method for SemanticKITTI-C
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/nusc2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/sk2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python adapt_online.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/synth2kitti.yaml
echo "All adaptation and evaluation processes are completed."