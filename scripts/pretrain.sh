gpus=0
cfg_path=./configs/pretrain.yaml
maxn_view=24
dst_name=atlas-mini
name=pretrain

mkdir -p ./logs/$name

CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
    --master_port 7007 \
    --nproc_per_node 1 \
    code/train.py \
        --name $name \
        --batch_size 2 \
        --epoch 800 \
        --dst_name $dst_name \
        --num_views $maxn_view \
        --min_views 6 \
        --random_views \
        --cfg_path $cfg_path \
        --dist \
        --vq_w 0.1 \
        >> ./logs/$name/train.log 2>&1 &
