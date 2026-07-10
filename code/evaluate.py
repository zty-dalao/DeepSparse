import os
import csv
import json
import time
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import convert_cuda, load_config, sitk_save



def count_parameters(model):
    '''返回模型总参数量'''
    return sum(p.numel() for p in model.parameters())


def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False, mixed_precision=False, test_time=False):
    '''对一个数据集跑一轮评估，计算 PSNR / SSIM'''
    model.eval()                                        # 关闭 dropout / batchnorm 训练行为
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim']}  # 得到{'psnr': [], 'ssim': []}
    if use_tqdm:                                        # 如果 use_tqdm=True，显示进度条
        loader = tqdm(loader, ncols=50)                 # ncols=50 设置进度条宽度为 50 个字符
    
    with torch.no_grad():
        for item in loader:                             # 遍历 loader 中每个 batch / 样本
            item = convert_cuda(item)                   # 将 item 中除 'name' 和 'dst_name' 外的张量转为 CUDA 浮点张量

            dst_name = item['dst_name'][0]              # 获取当前样本的目标数据集名称
            name = item['name'][0]                      # 获取当前样本的名称，用于保存结果和打印信息
            image = item['points_gt'].cpu().numpy()     # 获取当前样本的 ground truth 点云数据，并转为 NumPy 数组 
            image = image[0]                            # 获得当前样本的 ground truth 点云数据的形状为 (W, H, D)                  

            if mixed_precision:                         # 如果 mixed_precision=True，使用混合精度推理
                with autocast(dtype=torch.bfloat16):    # 使用 bfloat16 数据类型进行混合精度推理，autocast 会自动处理张量类型转换
                    if test_time:                       # 如果 test_time=True，记录推理时间
                        t_start = time.time()
                    pred = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N。得到模型预测结果 pred，包含 'points_pred' 键，对应预测的点云数据，点云数据的形状为 (B, 1, N)，其中 B 是 batch size，N 是点云数量
                    if test_time:
                        t_end = time.time()
                        print('inference time:', t_end - t_start)
                output = pred['points_pred']            # 获取模型预测的点云数据
                output = output[0, 0].data.cpu().float().numpy()    # 将预测结果转为 NumPy 数组，形状为 (N,)，并转换为 CPU 浮点张量。[0,0]的第一个索引是 batch size，第二个索引是通道数
            else:
                if test_time:
                    t_start = time.time()
                pred = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
                if test_time:
                    t_end = time.time()
                    print('inference time:', t_end - t_start)
                output = pred['points_pred']
                output = output[0, 0].data.cpu().numpy()
            
            output = output.reshape(image.shape)    # 将预测结果 output 重塑为与 ground truth 图像 image 相同的形状，以便进行 PSNR 和 SSIM 计算
            output = np.clip(output, 0, 1)          # 将预测结果 output 的值限制在 [0, 1] 范围内，避免出现负值或大于 1 的值，确保计算 PSNR 和 SSIM 时的数值稳定性

            psnr = peak_signal_noise_ratio(image, output, data_range=1.)    # 计算预测结果 output 与 ground truth 图像 image 之间的峰值信噪比（PSNR），data_range=1. 表示图像的像素值范围为 [0, 1]。
            ssim = structural_similarity(image, output, data_range=1.)      # 计算预测结果 output 与 ground truth 图像 image 之间的结构相似性指数（SSIM），data_range=1. 表示图像的像素值范围为 [0, 1]。

            if not ignore_msg:                      # 如果 ignore_msg=False，打印当前样本的名称、PSNR 和 SSIM 值
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])                             # 获取当前目标数据集 dst_name 的评估结果列表，如果不存在则返回空列表
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))          # 获取当前目标数据集 dst_name 的评估指标字典，如果不存在则返回 metrics_tmp 的深拷贝，metrics_tmp 包含 'psnr' 和 'ssim' 两个键，对应的值为一个空列表

            dst_res.append({                                                # 将当前样本的评估结果添加到 dst_res 列表中，包含 'name'、'psnr' 和 'ssim' 三个键
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
            })
            for key in dst_met.keys():                                      # 将当前样本的 PSNR 和 SSIM 值分别添加到 dst_met 字典中对应的列表中
                dst_met[key].append(dst_res[-1][key])                       # -1 表示获取 dst_res 列表中的最后一个元素，即当前样本的评估结果字典，然后根据 key 获取对应的值（PSNR 或 SSIM），并将其添加到 dst_met 字典中对应的列表中
            
            results[dst_name] = dst_res                                     # 将更新后的 dst_res 列表保存回 results 字典中，键为当前目标数据集 dst_name
            metrics[dst_name] = dst_met                                     # 将更新后的 dst_met 字典保存回 metrics 字典中，键为当前目标数据集 dst_name

            if save_dir is not None:                                        # 如果 save_dir 不为 None，表示需要保存预测结果
                spacing = item['spacing'][0].cpu().numpy()                  # 获取当前样本的体素间距 spacing，并将其转为 NumPy 数组
                origin = item['origin'][0].cpu().numpy()                    # 获取当前样本的物理坐标原点 origin，并将其转为 NumPy 数组
                save_path = os.path.join(save_dir, f'{name}.nii.gz')        # 构建保存预测结果的文件路径，文件名为当前样本的名称 name，后缀为 .nii.gz
                sitk_save(save_path, output, spacing=spacing, origin=origin, uint8=True)    # 调用 sitk_save 函数将预测结果 output 保存为 NIfTI 格式的医学图像文件，使用指定的 spacing 和 origin，并将像素值转换为 uint8 类型。

    for dst_name in metrics.keys():                                         # 遍历 metrics 字典中的每个目标数据集 dst_name
        dst_met = metrics[dst_name]                                         # 获取当前目标数据集 dst_name 的评估指标字典 dst_met，包含 'psnr' 和 'ssim' 两个键，对应的值为一个列表，存储了该数据集所有样本的 PSNR 和 SSIM 值
        m = {key:np.mean(val) for key, val in dst_met.items()}              # 计算当前目标数据集 dst_name 的平均 PSNR 和 SSIM 值，使用 NumPy 的 mean 函数对 dst_met 字典中的每个键对应的列表进行求平均，得到一个新的字典 m，包含 'psnr' 和 'ssim' 两个键，对应的值为该数据集所有样本的平均 PSNR 和 SSIM 值
        metrics[dst_name] = m                                               # 将计算得到的平均评估指标字典 m 保存回 metrics 字典中，键为当前目标数据集 dst_name

    return metrics, results                                                 # 返回 metrics 字典和 results 字典，metrics 字典包含每个目标数据集的平均 PSNR 和 SSIM 值，results 字典包含每个目标数据集的所有样本的评估结果


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')                        # 创建一个 ArgumentParser 对象，用于解析命令行参数，description='eval' 表示该脚本的描述信息为 'eval'

    parser.add_argument('--name', type=str, default='baseline')                 # 设置命令行参数 --name，类型为字符串，默认值为 'baseline'，用于指定实验名称
    parser.add_argument('--dst_name', type=str, default='LUNA16')               # 设置命令行参数 --dst_name，类型为字符串，默认值为 'LUNA16'，用于指定目标数据集名称
    parser.add_argument('--epoch', type=int, default=400)                   
    parser.add_argument('--num_views', type=int, default=10)            
    parser.add_argument('--cfg_path', type=str, default=None)                   # 设置命令行参数 --cfg_path，类型为字符串，默认值为 None，用于指定配置文件路径
    parser.add_argument('--split', type=str, default='test')                    # 设置命令行参数 --split，类型为字符串，默认值为 'test'，用于指定数据集划分
    parser.add_argument('--view_offset', type=int, default=0)                   # 设置命令行参数 --view_offset，类型为整数，默认值为 0，用于指定视图偏移量
    parser.add_argument('--out_res_scale', type=float, default=1.0)             # 设置命令行参数 --out_res_scale，类型为浮点数，默认值为 1.0，用于指定输出分辨率缩放比例
    parser.add_argument('--eval_npoint', type=int, default=100000)              # 设置命令行参数 --eval_npoint，类型为整数，默认值为 100000，用于指定评估时的点云数量
    parser.add_argument('--save_results', action='store_true', default=False)   # 设置命令行参数 --save_results，类型为布尔值，默认值为 False，用于指定是否保存评估结果
    parser.add_argument('--test_time', action='store_true', default=False)      # 设置命令行参数 --test_time，类型为布尔值，默认值为 False，用于指定是否在评估时记录推理时间

    args = parser.parse_args()
    if args.cfg_path is None:   # 如果命令行参数 --cfg_path 为 None，则根据实验名称 args.name 构建默认的配置文件路径 ./logs/{args.name}/config.yaml
        args.cfg_path = f'./logs/{args.name}/config.yaml'
    
    print(args)             

    cfg = load_config(args.cfg_path)

    # -- dataloader
    DST_CLASS = getattr(    # 动态获取数据集类)
        importlib.import_module('.'.join(['datasets'] + cfg.dataset.name.split('.')[:-1])),     # 根据 cfg.dataset.name 动态导入数据集模块，
        # 这行代码在拼接一个文件路径字符串，并去加载这个路径对应的 .py 文件（模块）。
        # 拆解步骤（假设 cfg.dataset.name = "ct_sparse.AblationNet_Dataset"）：
        #   1.cfg.dataset.name.split('.') -> 按点切分，得到 ['ct_sparse', 'AblationNet_Dataset']。
        #   2.[:-1] -> 去掉最后一个元素（因为是类名，不属于路径），剩下 ['ct_sparse']。
        #   3.['datasets'] + ['ct_sparse'] -> 得到 ['datasets', 'ct_sparse']。
        #   4. '.'.join(...) -> 最终拼成字符串 "datasets.ct_sparse"。
        #   5.importlib.import_module("datasets.ct_sparse") -> 去硬盘里找 datasets/ct_sparse.py 这个文件，并把里面的所有代码加载到内存中（相当于执行了 import datasets.ct_sparse）

        cfg.dataset.name.split('.')[-1]                                                         # 根据 cfg.dataset.name 动态获取数据集类名，为 cfg.dataset.name 的最后一个点分隔的部分
        # 这行在干什么？ 它从刚刚加载的模块文件里，把名叫 AblationNet_Dataset 的那个类给“抓”出来。
        # 拆解步骤：
        #   1.cfg.dataset.name.split('.')[-1] -> 还是切分，但取最后一个，得到字符串 "AblationNet_Dataset"。
        #   2.getattr(模块, "AblationNet_Dataset") -> Python 内置函数，相当于执行 模块.AblationNet_Dataset，把这个类的内存地址返回给你。
    )
    eval_loader = DataLoader(
        DST_CLASS(
            dst_name=args.dst_name,             # 指定目标数据集名称
            cfg=cfg.dataset,                    # 传入数据集配置参数
            split=args.split,                   # 指定数据集划分（train/val/test）
            num_views=args.num_views,           # 指定视图数量
            out_res_scale=args.out_res_scale,   # 指定输出分辨率缩放比例
            view_offset=args.view_offset,       # 指定视图偏移量。作用是让投影视角序列从不同位置开始选取，增加视角组合的变化性。训练时如果开启 random_views，它会成为一种增强式的随机视角变化。验证/测试时它则类似于“固定的起始角度设置”。
                                                # 例如，假设 num_views=10，view_offset=0，则选取的视角为 [0,1,2,...,9]；如果 view_offset=5，则选取的视角为 [5,6,7,...,14]。通过调整 view_offset，可以让模型在不同的视角组合下进行评估，从而更全面地测试模型的泛化能力。
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True                         # 让数据加载器在将数据传输到 GPU 时使用固定内存页，从而提高数据传输效率，减少 CPU 和 GPU 之间的数据传输延迟。
    )

    # -- model, load ckpt
    ckpt_path = f'./logs/{args.name}/ep_{args.epoch}.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))  # 加载检查点
    print('load ckpt from', ckpt_path)
    
    METHOD_CLASS = getattr(                     # 动态获取模型类
        importlib.import_module('.'.join(['models',] + cfg.model.name.split('.')[:-1])), 
        # 这行代码在拼接一个文件路径字符串，并去加载这个路径对应的 .py 文件（模块）。
        # 假设你的配置是：cfg.model.name = "ddpm.DiffusionUNet"
        # 拆解步骤：
        #   1. cfg.model.name.split('.') -> 切分为 ['ddpm', 'DiffusionUNet']
        #   2. [:-1] -> 去掉最后一个，保留文件夹名 ['ddpm']
        #   3. ['models',] + ['ddpm'] -> 拼成 ['models', 'ddpm']
        #   4. '.'.join(...) -> 得到字符串 "models.ddpm"
        #   5. importlib.import_module("models.ddpm") -> 去硬盘加载 models/ddpm.py 这个文件（里面定义了各种去噪网络变体）
        
        cfg.model.name.split('.')[-1]
        #   6. cfg.model.name.split('.')[-1] -> 取出最后一段，得到字符串 "DiffusionUNet"

        #   7. getattr(模块, "DiffusionUNet") -> 从 ddpm.py 文件中把 DiffusionUNet 这个类给掏出来，赋值给 METHOD_CLASS
    )
    model = METHOD_CLASS(cfg.model)
    '''
    实例化模型对象（model = METHOD_CLASS(cfg.model)）

    类拿到了，现在要真正在显存里把网络搭起来（分配内存、初始化权重）。
    cfg.model：传入的是模型的专属配置子字典。里面通常包含：
        网络骨架参数：in_channel=1（单通道CT）、out_channel=1、base_channels=64、num_res_blocks=2。
        扩散过程专属参数（如果是扩散模型）：num_timesteps=1000（去噪总步数）、beta_schedule="cosine"（噪声衰减策略）。
        迭代展开专属参数（如果是你的稀疏重建模型）：num_iterations=8（展开迭代次数）、data_consistency_weight=0.1（数据保真项权重）。
    执行效果：此时 model 已经是一个完整的 torch.nn.Module 子类实例，内部包含了几百万甚至上亿个待训练的参数张量。你可以随时调用 model(noisy_image, t) 进行前向推理了。
    '''

    print('model parameters:', count_parameters(model))
    model.load_state_dict(ckpt) # 加载模型权重
    model = model.cuda()        

    # -- output dir
    tag = '{:.1f}x'.format(args.out_res_scale)                                      # 构建输出结果的标签字符串，例如 "1.0x"、"0.5x"，用于区分不同分辨率缩放比例的评估结果
    save_dir = None                                             
    if args.save_results:
        save_dir = f'./logs/{args.name}/results/ep_{args.epoch}/predictions_{tag}'
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(  # 返回2个字典，第一个字典 metrics 记录每个数据集的平均 PSNR 和 SSIM，第二个字典 results 记录每个数据集的所有样本的评估结果
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        use_tqdm=True,
        ignore_msg=False,
        test_time=args.test_time
    )
    print(metrics)

    # -- save results [csv]
    pred_dir = f'./logs/{args.name}/results/ep_{args.epoch}'
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, f'results_{tag}.csv'), 'w', newline='')      # 创建一个 CSV 文件，用于保存评估结果，文件名为 results_{tag}.csv，其中 {tag} 是输出分辨率缩放比例的标签字符串
    csv_writer = csv.writer(csv_file)                                                   # 创建一个 CSV 写入器对象，用于将数据写入 CSV 文件
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])                          # 写入 CSV 文件的表头，包含 'dataset'（数据集名称）、'obj_id'（样本名称）、'psnr'（峰值信噪比）和 'ssim'（结构相似性指数）四列

    for dst_name in results.keys():                                                     # 遍历 results 字典中的每个目标数据集 dst_name
        dst_res = results[dst_name]
        for res in dst_res:                                                             # 遍历当前目标数据集 dst_name 的所有样本的评估结果 res
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])      # 将当前样本的评估结果写入 CSV 文件，包含数据集名称、样本名称、PSNR 和 SSIM 四列

        dst_avg = metrics[dst_name]                                                     # 获取当前目标数据集 dst_name 的平均评估指标字典 dst_avg，包含 'psnr' 和 'ssim' 两个键，对应的值为该数据集所有样本的平均 PSNR 和 SSIM 值
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])    # 将当前目标数据集 dst_name 的平均评估指标写入 CSV 文件，包含数据集名称、'average' 字符串、平均 PSNR 和平均 SSIM 四列
    
    csv_file.close()
    
    # -- save config [args]
    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:                           # 创建一个 JSON 文件，用于保存当前评估的命令行参数，文件名为 args.json
        args = vars(args)                                                               # 将 argparse.Namespace 对象 args 转换为字典形式，方便后续保存为 JSON 格式
        json.dump(args, f, indent=4)                                                    # 将命令行参数字典 args 保存为 JSON 格式到文件中，使用缩进为 4 个空格的格式，使得 JSON 文件更易读
