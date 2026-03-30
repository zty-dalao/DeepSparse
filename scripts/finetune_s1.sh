gpus=0
cfg_path=./configs/finetune_s1.yaml
maxn_view=24
pretrain_ckpt=./logs/pretrain/ep_700.pth

for dst_name in abdomen pelvis tooth luna; do
    for n_view in 6 8 10; do
        min_view=$n_view
        name=${dst_name}+${n_view}v+s1
        mkdir -p ./logs/$name
        echo "========================================="
        echo "Training: $name, Finetune Stage 1"
        echo "Dataset: $dst_name, Views: $n_view"
        echo "========================================="
        CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
            --master_port 2036 \
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
                --vq_w 0.1 \
                --resume_path $pretrain_ckpt \
                >> ./logs/$name/train.log 2>&1 &
        PID=$!
        echo "Process started with PID: $PID. Waiting..."
        wait $PID
    done
done

echo "All training jobs completed!"

