@echo off
cd /d %~dp0

set CUDA_VISIBLE_DEVICES=0

python infer.py ^
    --config ./configs/infer.yaml ^
    --model_path ./ckpt_models/ckpts ^
    --input_path ./example/123.png ^
    --lmk_path ./inference_temple/test_temple.npy ^
    --output_path ./data/out ^
    --model_step 0

pause