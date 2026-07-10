#!/bin/bash
gpus=0
cfg_path=./configs/finetune_s2.yaml
maxn_view=24

dst_name=thorax

for n_view in 6 8 10; do
    min_view=$n_view
    name=${dst_name}+${n_view}v+s2
    mkdir -p ./logs/$name
    echo "========================================="
    echo "Training: $name, Finetune Stage 2"
    echo "Dataset: $dst_name, Views: $n_view"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
        --master_port 2037 \
        --nproc_per_node 1 \
        code/train.py \
            --name $name \
            --batch_size 2 \
            --epoch 400 \
            --dst_name $dst_name \
            --num_views $maxn_view \
            --min_views $min_view \
            --random_views \
            --cfg_path $cfg_path \
            --dist \
            --vq_w 1.0 \
            --safely_load \
            --freeze_ft \
            --resume_path "./logs/${dst_name}+${n_view}v+s1/ep_400.pth" \
            >> ./logs/$name/train.log 2>&1 &
    # --safely_load（容错加载开关）。加上这个参数，脚本会忽略缺失或多余的层，只加载能匹配上的部分。
    # --resume_path（S2 用的）：仅用于迁移加载。它只提取模型参数（state_dict），优化器的状态会重新初始化（学习率重置），Epoch 计数从 0 或给定的 --epoch 400 重新开始计算。这意味着 S2 是一次全新的“微调旅程”，而不是接着 S1 的时间线。
    # 固定住网络底层或编码器（Encoder）的权重，在 S2 训练过程中不再更新这些层的参数，只更新高层（解码器 Decoder 或输出层）
    PID=$!
    echo "Process started with PID: $PID. Waiting..."
    wait $PID
done

echo "All thorax Stage 2 training jobs completed!"
