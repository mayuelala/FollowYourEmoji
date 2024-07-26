CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $LOCAL_IP \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --input_path ./data/images \
    --lmk_path ./data/mplmks \
    --output_path ./data/out