gpu=0
data_dir=./resampled
save_dir=./projections
vis_dir=./projections_vis
config=../config.yaml

mkdir -p $save_dir $vis_dir

for name in $(ls $data_dir); do
    echo $name
    CUDA_VISIBLE_DEVICES=$gpu python project.py \
        -n $name \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --vis_dir $vis_dir \
        --config $config
done
