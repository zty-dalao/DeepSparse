#!/bin/bash

gpus=0                                  # 指定使用 GPU 0 号卡
cfg_path=./configs/finetune_s1.yaml     # 读入之前我们讨论过的 YAML 配置文件
maxn_view=24                            # 最大视角数（这里当作“全采样”或基础对比基准）
                                        # 注意：虽然 maxn_view=24，但下面循环里实际只用了 6、8、10。这里的 24 可能代表完整采样的地面真值（Ground Truth），用于在训练时计算损失（模型输入稀疏图，输出与 24 视图的全图做对比）。
dst_name=thorax                         # 数据集名称：胸部 CT

for n_view in 6 8 10; do                # 训练6view,8view，10view
    min_view=$n_view                    # 以6 view为例，min_view=6，maxn_view=24，表示训练时输入的投影视角数为 6，输出的 Ground Truth 投影视角数为 24。
    name=${dst_name}+${n_view}v+s1      # name=thorax+6v+s1, thorax+8v+s1, thorax+10v+s1，表示胸部数据集，6/8/10 视角，Stage 1 训练
    log_dir=./logs/$name                # 日志目录：./logs/thorax+6v+s1, ./logs/thorax+8v+s1, ./logs/thorax+10v+s1
    mkdir -p "$log_dir"                 # 创建对应的日志目录

    # 自动识别该目录下最新的 checkpoint
    latest_ckpt=$(find "$log_dir" -maxdepth 1 -name 'ep_*.pth' | sort -V | tail -n 1)   # find "$log_dir"：去该实验的日志文件夹里找。
                                                                                        # =maxdepth 1：只在当前目录下找，不进入子目录。
                                                                                        # -name 'ep_*.pth'：找以 ep_ 开头、.pth 结尾的模型权重文件（PyTorch 标准格式）。
                                                                                        # sort -V：按版本号排序（比如 ep_10.pth 会排在 ep_9.pth 后面）。
                                                                                        # tail -n 1：取最后一个，即最新的 Epoch 权重。


    if [ -z "$latest_ckpt" ]; then                                                      # 如果没找到（-z "$latest_ckpt"），脚本直接报错退出，提醒你先跑 Stage-1（第一阶段完整训练）。这保证了续训必须基于已有的成果，不会从零开始瞎跑。
        echo "[ERROR] No checkpoint found under $log_dir"
        echo "Please run the normal stage-1 script first."
        exit 1
    fi

    latest_epoch=$(basename "$latest_ckpt" | sed -n 's/^ep_\([0-9]\+\)\.pth$/\1/p')     # 假设找到的权重文件是 ep_120.pth，那么 latest_epoch=120。
                                                                                        # basename:去掉文件路径中的目录前缀，只保留最后的文件名（基础名称）。如 /path/to/file.txt -> file.txt

    if [ -z "$latest_epoch" ]; then
        echo "[ERROR] Failed to parse checkpoint epoch from: $latest_ckpt"
        exit 1
    fi

    # 继续训练 400 个 epoch（从最新 checkpoint 开始）
    resume_epoch=$latest_epoch
    target_epoch=$((resume_epoch + 400))

    echo "========================================="
    echo "Continue training: $name"
    echo "Dataset: $dst_name, Views: $n_view"
    echo "Latest checkpoint: $latest_ckpt"
    echo "Resume epoch: $resume_epoch"
    echo "Target epoch: $target_epoch"
    echo "========================================="

    # nohup 让进程在后台运行且不受终端断开影响。
    # python -m：告诉 Python 解释器，把后面的 torch.distributed.launch 当作一个可执行的模块（脚本）来运行
    # torch.distributed.launch：这是 PyTorch 内置的一个多进程启动器（Launcher）。它的核心职责是：
    #   1.读取环境变量（比如当前是世界里的第几号进程 RANK）。
    #   2.自动复制并启动你的 train.py 脚本 N 次（N 由后面的 nproc_per_node 决定）。
    #   3.为每次启动的脚本分配不同的 RANK 和 LOCAL_RANK，让它们知道谁是“老大”（Master），谁是“小弟”。
    # torch.distributed.launch 在 PyTorch 1.10 之后已被标记为弃用（Deprecated），官方强烈建议改用 torchrun。但在很多老项目和兼容性代码中，你依然会大量见到 launch。如果写成 torchrun --nproc_per_node=1 train.py ...，效果完全等价。
    # 启动器在初始化分布式后端（NCCL/GLOO）时，多个进程（即使在同一台机器上）需要通过TCP 网络端口来互相同步梯度、交换信息。
    # --nproc_per_node 1 表示只用1张卡的分布式训练

    CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
        --master_port 2038 \
        --nproc_per_node 1 \
        code/train.py \
            --name $name \
            --batch_size 2 \
            --epoch $target_epoch \
            --resume $resume_epoch \
            --dst_name $dst_name \
            --num_views $maxn_view \
            --min_views $min_view \
            --random_views \
            --cfg_path $cfg_path \
            --dist \
            --vq_w 0.1 \
            >> "$log_dir/train.log" 2>&1 &  # & 将进程放入后台。
            # --num_views = 24 和 --min_views = 6，8，10 表示模型接收的输入是从 24 个完整视角中，随机抽取 6 个（或 8、10 个）角度的投影数据（正弦图），但损失函数监督的 Ground Truth 依然是 24 视角重建出的高清图。
            # --random_views 标志开启后，每个 Epoch 抽取的具体角度编号都不同（比如这次抽第 1, 5, 9...，下次抽 2, 6, 10...）。这极大地增强了模型对任意稀疏角度的泛化能力，避免了过拟合于固定的 6 个角度。
            # 分布式训练（--dist）与单卡限制（--nproc_per_node 1）
            # 它用了 torch.distributed.launch（分布式启动器），但 nproc_per_node=1 意味着只开了一张卡（因为 gpus=0）。
            # 这里用分布式启动器可能是为了代码兼容（比如统一采用 DistributedSampler 管理数据加载），或者实际只用了单卡但保留了多卡扩展接口。
    PID=$!  # wait $PID 强制当前 Shell 阻塞等待这个后台进程结束。虽然用了后台命令，但因为 wait 的存在，这 3 个视角（6、8、10）是按顺序串行执行的（跑完 6 视角的 400 轮，才开始跑 8 视角），而不是一起抢显存。这避免了显存溢出（OOM）。
    echo "Process started with PID: $PID. Waiting..."
    wait $PID

done

echo "All thorax Stage 1 continuation jobs completed!"
