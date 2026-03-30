gpus=0
cfg_path=./configs/finetune_s2.yaml
maxn_view=24

for dst_name in abdomen pelvis tooth luna; do
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
        PID=$!
        echo "Process started with PID: $PID. Waiting..."
        wait $PID
    done
done

echo "All training jobs completed!"
