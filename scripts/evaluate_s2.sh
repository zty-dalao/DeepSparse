gpus=0
n_epoch=400

for dst_name in abdomen pelvis tooth luna; do
    for n_view in 6 8 10; do
        name=${dst_name}+${n_view}v+s2
        echo "========================================="
        echo "Evaluating: $name"
        echo "Dataset: $dst_name, Views: $n_view"
        echo "========================================="
        CUDA_VISIBLE_DEVICES=$gpus python code/evaluate.py \
            --name $name \
            --epoch $n_epoch \
            --dst_name $dst_name \
            --split test \
            --num_views $n_view \
            --out_res_scale 1.0 \
            --test_time \
            --save_results
    done
done
