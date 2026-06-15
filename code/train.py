import os
import shutil
import argparse
import importlib
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import convert_cuda, load_config
from evaluate import eval_one_epoch


def load_ckpt_safe(model, ckpt):
    model_dict = model.state_dict()
    ckpt = {k : v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)
    return model


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--out_res_scale', type=float, default=1.)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--dst_subset', type=float, default=1.)

    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--random_views', action='store_true', default=False)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--resume_path', type=str, default=None)

    parser.add_argument('--vq_w', type=float, default=1.)
    parser.add_argument('--safely_load', action='store_true', default=False)
    parser.add_argument('--freeze_ft', action='store_true', default=False)
    
    parser.add_argument('--min_views', type=int, default=6)

    args = parser.parse_args()

    if args.dist:
        args.local_rank = int(os.environ["LOCAL_RANK"]) # Make it compatible with different versions of DDP
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.cfg_path)
    if args.local_rank == 0:    # 只有当前进程的 local_rank 为 0 时，才执行后面的代码块。
                                # 在分布式训练（DDP）里，local_rank==0 通常是主进程/主 GPU。
                                # 这段判断的目的是让只有主进程打印参数、保存配置等“只做一次”的工作，避免多个进程重复输出或重复写文件。
        print(args)
        print(cfg)

        # save config
        save_dir = f'./logs/{args.name}'                                        # 构造日志保存目录路径。例如 args.name = 'pretrain' → save_dir = './logs/pretrain'
        os.makedirs(save_dir, exist_ok=True)                                    # 创建日志保存目录，如果目录已存在则不报错。
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):               # 检查 config.yaml 是否已经存在（即之前是否已经训练过）。
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')             # 生成当前时间字符串，格式如 26-05-2026_14-30-00，用作备份文件的时间戳。
            shutil.copyfile(                                                    # 把旧的 config.yaml 复制为 config_26-05-2026_14-30-00.yaml，作为历史备份。这样每次重新训练时，之前的配置不会被覆盖。
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
        shutil.copyfile(args.cfg_path, os.path.join(save_dir, 'config.yaml'))   # 把本次训练使用的配置文件（args.cfg_path，如 pretrain.yaml）复制到日志目录下，命名为 config.yaml。
                                                                                # 这样日志目录里始终保存着本次训练的配置，方便之后复现。

    # Set min_views via args
    cfg.dataset.min_views = args.min_views

    # -- initialize training dataset/loader
    '''
    getattr() 函数用于获取对象的属性值。它的语法是 getattr(object, name[, default])，其中 object 是对象，name 是属性名字符串，default 是可选参数，当属性不存在时返回的默认值。
    在这里，getattr() 用来根据配置文件中的模型名称动态导入数据集类。具体来说：
1. cfg.dataset.name 是配置文件中指定的数据集类的全路径字符串，例如 'datasets.atlas_mini' '。
2. importlib.import_module() 用来导入模块，这里导入的是 'datasets.atlas_mini' 模块（去掉最后的类名部分）。
3. cfg.dataset.name.split('.')[-1] 则提取出类名部分，例如 'AtlasMini'。
4. 最终，getattr() 会返回数据集类对象，例如 datasets.atlas_mini.AtlasMini，这样就可以实例化这个数据集了。
    这种方式的好处是代码更加灵活，不需要在 train.py 中写死具体的数据集类名，用户只需要在配置文件中指定即可，方便扩展新的数据集。
    '''
    DST_CLASS = getattr(    # 根据配置文件动态选择数据集类，不需要写死 import
        importlib.import_module('.'.join(['datasets'] + cfg.dataset.name.split('.')[:-1])), 
        cfg.dataset.name.split('.')[-1]
    )
    train_dst = DST_CLASS(
        dst_name=args.dst_name,             # --dst_name	数据集名称，如 'atlas-mini'
        cfg=cfg.dataset,                    # 配置 YAML 中 dataset 段	数据集根目录、分辨率等
        split='train',                      # 固定 'train'	训练集划分
        num_views=args.num_views,           # --num_views	最大投影视角数，如 24
        npoint=args.num_points,             # --num_points	每次随机采样的 3D 点数，如 10000
        out_res_scale=args.out_res_scale,   # --out_res_scale	输出分辨率缩放，默认 1.0
        random_views=args.random_views,     # --random_views	是否随机偏移视角起始位置
        subset=args.dst_subset              # --dst_subset	数据集子采样比例，如 1.0=全部--dst_subset	数据集子采样比例，如 1.0=全部
    )
    train_sampler = None                    # 默认不使用分布式采样器。
    if args.dist:                           # 如果启用分布式训练（DDP），则使用 DistributedSampler 来划分数据集，使每个进程只处理数据集的一部分，避免重复计算。
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        sampler=train_sampler,              # 分布式采样器（单卡时为 None）
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,                    # 锁页内存，加速 CPU→GPU 数据传输
        worker_init_fn=worker_init_fn       # 每个 worker 初始化时设置不同的随机种子，保证数据增强多样性
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        DST_CLASS(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='eval',
            num_views=args.num_views,
            out_res_scale=0.5, # low-res for faster evaluation,低分辨率评估，只有训练分辨率的一半，用于加速
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- initialize model
    METHOD_CLASS = getattr(     # 根据配置文件动态选择模型类，不需要写死 import。与数据集导入同理
        importlib.import_module('.'.join(['models',] + cfg.model.name.split('.')[:-1])), 
        cfg.model.name.split('.')[-1]
    )
    model = METHOD_CLASS(cfg.model) # 用配置中的 model 段实例化模型。例如传入 cfg.model 这个 EasyDict，其中包含 encoder、decoder、codebook 等子配置。
    if args.resume_path:                                    # --resume_path：从外部 checkpoint 加载模型。例如 Stage 1 微调时加载预训练权重 ep_700.pth。
        print(f'resume model from {args.resume_path}')      # 打印日志，告知从哪个路径恢复。
        ckpt = torch.load(                                  # 加载 checkpoint 到 CPU（map_location='cpu'），避免 GPU 内存溢出
            args.resume_path,
            map_location=torch.device('cpu')
        )
        if args.safely_load:                                # --safely_load：安全加载模式。
            model = load_ckpt_safe(model, ckpt)             # 安全加载（load_ckpt_safe）：只加载 checkpoint 中与模型匹配的键，忽略多余的键。
        else:
            model.load_state_dict(ckpt)                     # 普通加载（load_state_dict）：要求 checkpoint 的键与模型完全匹配，否则报错。
    
    if args.resume:                                         # --resume：从训练中断处恢复。例如 --resume 400 加载 ./logs/pretrain/ep_400.pth。
        print(f'resume model from epoch {args.resume}')
        ckpt = torch.load(
            os.path.join(save_dir, f'ep_{args.resume}.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckpt)

    if args.freeze_ft:                                      # --freeze_ft：冻结模型的特征提取部分（encoder），只微调 decoder 和 codebook。                         
        model.freeze_ft()                                   # 调用模型的 freeze_ft() 方法，冻结 encoder 的权重，使其在训练过程中不更新，只训练 decoder 和 codebook。这通常用于微调阶段，利用预训练的特征提取能力，同时适应新的数据分布。
    
    model = model.cuda()
    if args.dist:                                           # 如果启用了分布式训练，用 DDP 包装模型。
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=False,                   # 不检测未使用参数，提升效率。
            device_ids=[args.local_rank]                    # 绑定到当前 GPU。
        )
    
    # -- initialize optimizer, lr scheduler, and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = None
    loss_func = nn.MSELoss()

    start_epoch = 0
    if args.resume:                                         # 如果使用了 --resume（断点续训）
        start_epoch = args.resume + 1                       # start_epoch = args.resume + 1：从下一个 epoch 继续。例如 --resume 400，则从 epoch 401 开始。
        if lr_scheduler is not None:                        # 如果有 LR scheduler，则同步其内部状态到恢复的 epoch（但本项目 lr_scheduler = None，所以这行实际不执行）。
            lr_scheduler.step(epoch=args.resume)

    if args.mixed_precision:                                # --mixed_precision：启用混合精度训练（FP16/BF16）。
        from torch.cuda.amp import GradScaler, autocast     
        amp_grad_scaler = GradScaler()                      # GradScaler：自动缩放梯度，防止 FP16 下梯度下溢。

    # -- training starts
    for epoch in range(start_epoch, args.epoch + 1):
        if args.dist:                                       # 分布式训练中，每个 epoch 调用 set_epoch。
            train_loader.sampler.set_epoch(epoch)           # 作用：让每个 epoch 的数据 shuffle 不同，保证各 GPU 的随机性。
                                                            # 三个列表，分别记录当前 epoch 中每个 batch 的：
        loss_list = []                                      # 总损失
        loss_task_list = []                                 # 任务损失（MSE on CT values）
        loss_vq_list = []                                   # codebook 量化损失
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            item = convert_cuda(item)                       # 把数据字典中的所有张量移到 GPU

            if args.mixed_precision:
                with autocast(dtype=torch.bfloat16):                                # 上下文管理器，自动把部分计算转为 BF16 精度
                    pred = model(item)                                              # 前向传播在混合精度下进行
                    loss_task = loss_func(pred['points_pred'], item['points_gt'])   # MSE(预测, GT)
                    loss_vq = pred.get('loss_vq', torch.tensor(0.).float().cuda())  # codebook 量化 loss（beta 已在内部乘过），如果模型没返回则默认 0
                    loss = loss_task + args.vq_w * loss_vq                          # 总损失，vq_w 控制量化 loss 权重

                amp_grad_scaler.scale(loss).backward()                              # 用 GradScaler 缩放 loss 后再反向传播，防止 FP16 梯度下溢。
                amp_grad_scaler.unscale_(optimizer)                                 # 反向传播后，把梯度还原为原始尺度。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)             # 梯度裁剪，最大范数 = 1.0。防止梯度爆炸。
                amp_grad_scaler.step(optimizer)                                     # 更新参数（GradScaler 会自动跳过无效梯度）。
                amp_grad_scaler.update()                                            # 更新 GradScaler 内部的缩放因子。
                optimizer.zero_grad()                                               # 清空梯度，准备下一个 batch。
            else:                                                                   # 与混合精度逻辑完全相同，但没有 autocast 和 GradScaler。
                                                                                    # 标准流程：前向 → 算 loss → 反向 → 梯度裁剪 → 更新 → 清零。
                pred = model(item)
                loss_task = loss_func(pred['points_pred'], item['points_gt'])
                loss_vq = pred.get('loss_vq', torch.tensor(0.).float().cuda())
                loss = loss_task + args.vq_w * loss_vq

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_task_list.append(loss_task.item())
            loss_vq_list.append(loss_vq.item())
            loss_list.append(loss.item())

        if args.local_rank == 0:    # 分布式训练时，只有 rank 0（主进程）执行打印、保存、评估。
            if epoch % 10 == 0:     # 每 10 epoch 打印 loss
                loss = np.mean(loss_list)
                loss_task = np.mean(loss_task_list)
                loss_vq = np.mean(loss_vq_list)
                print('epoch: {}, loss: {:.4}, loss_task: {:.4}, loss_vq: {:.4}'.format(epoch, loss, loss_task, loss_vq))
            
            if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):   # 每 100 epoch 保存一次（100, 200, 300, ...）。最后 100 epoch 每 20 epoch 保存一次（最后阶段更密集）
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save(                                     # 保存到 ./logs/pretrain/ep_100.pth 等。
                    model_state,
                    os.path.join(save_dir, f'ep_{epoch}.pth')
                )

            if epoch % 50 == 0:                                 # 每 50 epoch 在评估集上跑一次全量评估，计算 PSNR 和 SSIM。
                metrics, _ = eval_one_epoch(
                    model, 
                    eval_loader, 
                    args.eval_npoint,
                    ignore_msg=True,                            # 不逐样本打印，只汇总。
                    mixed_precision=args.mixed_precision
                )
                msg = f' --- epoch {epoch}'
                for dst_name in metrics.keys():
                    msg += f', {dst_name}'
                    met = metrics[dst_name]
                    for key, val in met.items():
                        msg += ', {}: {:.4}'.format(key, val)
                print(msg)                                      # 拼接评估指标并打印。例如：epoch 50, atlas-mini, psnr: 32.15, ssim: 0.9214
        
        if lr_scheduler is not None:                            # 如果配置了 LR scheduler，每个 epoch 调用一次。但本项目 lr_scheduler = None，所以跳过
            lr_scheduler.step()                                 
