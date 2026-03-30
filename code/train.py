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
    if args.local_rank == 0:
        print(args)
        print(cfg)

        # save config
        save_dir = f'./logs/{args.name}'
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            shutil.copyfile(
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
        shutil.copyfile(args.cfg_path, os.path.join(save_dir, 'config.yaml'))

    # Set min_views via args
    cfg.dataset.min_views = args.min_views
    # -- initialize training dataset/loader
    DST_CLASS = getattr(
        importlib.import_module('.'.join(['datasets'] + cfg.dataset.name.split('.')[:-1])), 
        cfg.dataset.name.split('.')[-1]
    )
    train_dst = DST_CLASS(
        dst_name=args.dst_name,
        cfg=cfg.dataset,
        split='train', 
        num_views=args.num_views, 
        npoint=args.num_points,
        out_res_scale=args.out_res_scale,
        random_views=args.random_views,
        subset=args.dst_subset
    )
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        DST_CLASS(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='eval',
            num_views=args.num_views,
            out_res_scale=0.5, # low-res for faster evaluation,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- initialize model
    METHOD_CLASS = getattr(
        importlib.import_module('.'.join(['models',] + cfg.model.name.split('.')[:-1])), 
        cfg.model.name.split('.')[-1]
    )
    model = METHOD_CLASS(cfg.model)
    if args.resume_path:
        print(f'resume model from {args.resume_path}')
        ckpt = torch.load(
            args.resume_path,
            map_location=torch.device('cpu')
        )
        if args.safely_load:
            model = load_ckpt_safe(model, ckpt)
        else:
            model.load_state_dict(ckpt)
    
    if args.resume:
        print(f'resume model from epoch {args.resume}')
        ckpt = torch.load(
            os.path.join(save_dir, f'ep_{args.resume}.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckpt)

    if args.freeze_ft:
        model.freeze_ft()
    
    model = model.cuda()
    if args.dist:
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=False,
            device_ids=[args.local_rank]
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
    if args.resume:
        start_epoch = args.resume + 1
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=args.resume)

    if args.mixed_precision:
        from torch.cuda.amp import GradScaler, autocast
        amp_grad_scaler = GradScaler()

    # -- training starts
    for epoch in range(start_epoch, args.epoch + 1):
        if args.dist:
            train_loader.sampler.set_epoch(epoch)

        loss_list = []
        loss_task_list = []
        loss_vq_list = []
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            item = convert_cuda(item)

            if args.mixed_precision:
                with autocast(dtype=torch.bfloat16):
                    pred = model(item)
                    loss_task = loss_func(pred['points_pred'], item['points_gt'])
                    loss_vq = pred.get('loss_vq', torch.tensor(0.).float().cuda())
                    loss = loss_task + args.vq_w * loss_vq

                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
                optimizer.zero_grad()
            else:
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

        if args.local_rank == 0:
            if epoch % 10 == 0:
                loss = np.mean(loss_list)
                loss_task = np.mean(loss_task_list)
                loss_vq = np.mean(loss_vq_list)
                print('epoch: {}, loss: {:.4}, loss_task: {:.4}, loss_vq: {:.4}'.format(epoch, loss, loss_task, loss_vq))
            
            if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save(
                    model_state,
                    os.path.join(save_dir, f'ep_{epoch}.pth')
                )

            if epoch % 50 == 0:
                metrics, _ = eval_one_epoch(
                    model, 
                    eval_loader, 
                    args.eval_npoint,
                    ignore_msg=True,
                    mixed_precision=args.mixed_precision
                )
                msg = f' --- epoch {epoch}'
                for dst_name in metrics.keys():
                    msg += f', {dst_name}'
                    met = metrics[dst_name]
                    for key, val in met.items():
                        msg += ', {}: {:.4}'.format(key, val)
                print(msg)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
