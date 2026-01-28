#!/bin/bash


# Run eval_only for no adaptation baseline for SemanticSTF
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/nusc2stf.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/sk2stf.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-stf/synth2stf.yaml

# Run eval_only for no adaptation baseline for SemanticKITTI-C
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/nusc2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/sk2kitti.yaml
CUDA_VISIBLE_DEVICES=0 python eval_only.py --config_file /home/yewon/project/tta/D3CTTA/configs/adaptation/test-kitti/synth2kitti.yaml

"