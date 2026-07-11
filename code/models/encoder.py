import torch
import torch.nn as nn
import numpy as np

from models.conv_utils import Pooling
from models.utils import query_view_feats
from models.image_encoders.base import Encoder_base
from models.image_encoders.mednextv1.MedNextV1 import MedNeXt



class EncoderV_mv(nn.Module):
    '''
    EncoderV_mv 就是一个 “通过多张 2D 照片反推 3D 结构的测绘师”。
    它的输入是多视角投影图像（projs），输出是对应的 3D 特征图。
    它的工作流程是：
    1. 先用一个 2D 编码器（Encoder_base 或 MedNeXt）提取每张投影图像的 2D 特征。
    2. 再通过 query_view_feats 函数，将每张投影图像的 2D 特征映射到 3D 空间，得到对应的 3D 特征图。
    3. 最后返回每个视图掩码对应的 3D 特征图列表。
    '''
    def __init__(self, cfg):
        super().__init__()
        if cfg.encoder_type == 'base':
            self.encoder = Encoder_base(    # 配置可以见configs文件夹，这里以finetune_s1.yaml为例，base_ch=16, n_layer=3, ch_up=2
                in_ch=1, 
                base_ch=cfg.base_ch,    
                n_layer=cfg.n_layer,
                ch_up=cfg.ch_up,
                n_dim='2d'
            )
        elif cfg.encoder_type == 'mednextv1':
            self.encoder = MedNeXt(
                in_ch=1, 
                base_ch=cfg.base_ch,
                n_layer=cfg.n_layer,
                ch_up=cfg.ch_up,
                n_blocks=cfg.n_blocks,
                exp_r=cfg.exp_r,
            )
        else:
            raise NotImplementedError
    
    @property
    def chs(self):
        return self.encoder.chs

    def forward(self, data, require_2d=False, view_masks=[None]):
        # [B, M, C, W, H]，分别指的是batch_size、view_num、channel、width、height
        x = data['projs']                                           # 从train.py的pred = model(item) 传入
                                                                    # data参数是从data_loader中获取的一个batch数据，这里取出projs，即投影图像。     通过 p data.keys() 查看
                                                                    # dst_name
                                                                    # 含义：数据集名称
                                                                    # 取值示例：thorax、atlas-mini、luna
                                                                    # 作用：告诉数据集类去哪个子目录加载数据，比如 thorax 会映射到 ./data/Thorax Fast
                                                                    
                                                                    # name
                                                                    # 含义：样本标识符
                                                                    # 取值示例：某个体/扫描的名字（从 meta_info.json 中加载）
                                                                    # 作用：用于定位该样本对应的投影文件、体素文件、块文件等
                                                                    
                                                                    # view_mask
                                                                    # 含义：视角有效掩码
                                                                    # 形状：[M]
                                                                    # 内容：布尔值或 0/1，前 n_views 为 True，剩余 max_views-n_views 为 False
                                                                    # 作用：
                                                                    #   1.标记哪些视图是真实采样的
                                                                    #   2.在 query_view_feats 中屏蔽无效视图，避免填充视图影响最大池化融合
                                                                    
                                                                    # angles
                                                                    # 含义：投影视角角度
                                                                    # 形状：[M]
                                                                    # 作用：用于将 3D 点坐标投影到每个视图平面，生成投影坐标 points_proj
                                                                    # 说明：每个视图的 angle 可能是弧度或角度，具体由 datasets.geometry.Geometry_nonuniform 解释
                                                                    
                                                                    # projs
                                                                    # 含义：投影图像
                                                                    # 形状：[B， M, 1, W, H]
                                                                    # 内容：每个视角的 2D 投影图像，1 通道通常是单通道灰度
                                                                    # 作用：
                                                                    #   1.EncoderV_mv 的输入
                                                                    #   2.通过 2D 编码器提取视图特征
                                                                    
                                                                    # points
                                                                    # 含义：训练时采样的 3D 点位置
                                                                    # 形状：[N, 3]
                                                                    # 内容：体素中心坐标，范围大致是 [0, 1]
                                                                    # 作用：
                                                                    #   1.这些点会被投影到 2D 视图，得到 points_proj
                                                                    #   2.预测目标是这些点对应的体素值
                                                                    
                                                                    # points_gt
                                                                    # 含义：点的真实标量值
                                                                    # 形状：训练时为 [1, N]
                                                                    # 内容：对应 points 的真实体素值（归一化到 [0,1]）
                                                                    # 作用：
                                                                    #   1.训练损失目标
                                                                    #   2.在 train.py 中与 points_pred 做 MSE
                                                                    
                                                                    # points_proj
                                                                    # 含义：采样点 points 在每个视图上的投影坐标
                                                                    # 形状：[M, N, 2]
                                                                    # 内容：每个点在每个视角的图像平面坐标
                                                                    # 作用：
                                                                    #   1.forward_points() 中用于从 2D 特征里采样点级特征
                                                                    #   2.通过 query_view_feats(..., points_proj=data['points_proj']) 聚合多视角点特征
                                                                    
                                                                    # points_lr_proj
                                                                    # 含义：低分辨率固定点网格在每个视图上的投影坐标
                                                                    # 形状：[M, N_lr, 2]
                                                                    # 来源：CBCT_dataset_LR 里基于 lr_res 生成固定稠密网格
                                                                    # 作用：
                                                                    #   1.EncoderV_mv 用它将低分辨率点在视图上投影，生成多尺度 3D 初始特征
                                                                    #   2.这是一种“粗粒度 3D 特征”构建方式
                                                                    
                                                                    # points_ct
                                                                    # 含义：低分辨率点网格坐标经过归一化后的 3D 坐标
                                                                    # 形状：[N_lr, 3]
                                                                    # 来源：CBCT_dataset_LR 继承 CBCT_dataset，在 __getitem__ 中把 points 复制、归一化为 [-1,1]
                                                                    # 作用：
                                                                    #   1.forward_points() 在 feats_dict['feats_3d'] 上用 index_3d(..., data['points_ct']) 采样点级特征
                                                                    #   2.这里 points_ct 是实际用于从 3D 特征体中采点的坐标
                                                                    
        b, m = x.shape[:2]                                          # b: batch_size, m: view_num
        x = x.reshape(b * m, *x.shape[2:])                          # 将projs张量的前两维（batch_size和view_num）合并为一维，得到一个新的张量x，shape为[B*M, C, W, H]，其中B*M表示所有视图的总数，C表示通道数，W和H分别表示宽度和高度。

        feats_3d_lists = [[] for _ in range(len(view_masks))]       # 创建一个列表feats_3d_lists，长度为view_masks的长度，每个元素都是一个空列表，用于存储每个视图掩码对应的3D特征图。
        if require_2d:                                              # 如果require_2d为True，则创建一个空列表feats_2d_list，用于存储每个视图的2D特征图。
            feats_2d_list = []
        # ============================================================
        # [GPU计时] 2D CNN编码器: 24张投影→4尺度2D特征图
        # ============================================================
        gpu_timer_cnn = torch.cuda.Event(enable_timing=True)
        gpu_timer_cnn_end = torch.cuda.Event(enable_timing=True)
        gpu_timer_cnn.record()

        self_encoder_x = self.encoder(x)

        gpu_timer_cnn_end.record()
        torch.cuda.synchronize()
        print(f"[GPU计时] 2D CNN编码(24投影→4尺度2D特征) 耗时 {gpu_timer_cnn.elapsed_time(gpu_timer_cnn_end):.3f} ms")

        # ============================================================
        # [GPU计时] query_view_feats ×4: 4尺度2D多视角→3D体素反投影
        # ============================================================
        gpu_timer_qvf_all = torch.cuda.Event(enable_timing=True)
        gpu_timer_qvf_all_end = torch.cuda.Event(enable_timing=True)
        gpu_timer_qvf_all.record()

        for feats in self_encoder_x:                               # 通过调用编码器的forward方法，将输入张量x传入编码器，得到编码器输出的特征图feats。这里的feats是一个张量，shape为[B*M, C', W', H']，其中C'表示编码器输出的通道数，W'和H'分别表示编码器输出的宽度和高度。 这里再finetunes1中会直接跳转到base.py的Encoder_base类的forward函数当中。self.encoder(x)的数据是一组多通道的二维投影，以finetune_s1.yaml为例，一个列表：[[48, 128, 32, 32]，[48, 64, 64, 64]，[48, 32, 128, 128]，[48, 16, 256, 256]]
            # [B, M, C', W', H']
            feats = feats.reshape(b, m, *feats.shape[1:])           # 将编码器输出的特征图feats的前两维（batch_size和view_num）重新分开，得到一个新的张量feats，shape为[B, M, C', W', H']，其中B表示batch_size，M表示view_num，C'表示编码器输出的通道数，W'和H'分别表示编码器输出的宽度和高度。
            if require_2d:                                          # 如果require_2d为True，则将当前视图的2D特征图feats添加到feats_2d_list中。
                feats_2d_list.append(feats)
            
            for i, view_mask in enumerate(view_masks):              # 遍历view_masks列表中的每个视图掩码view_mask，并获取其索引i。

                feats_3d = query_view_feats(                        # 调用query_view_feats函数，将当前视图的2D特征图feats、投影点坐标data['points_lr_proj']、聚合策略fusion='max'和视图掩码view_mask作为参数，得到对应的3D特征图feats_3d。
                    view_feats=feats,
                    points_proj=data['points_lr_proj'],             # data['points_lr_proj'].shape=torch.Size([2, 24, 32768, 2])。 第一个为batch size，当前一个 batch 中有 2 个样本；第二个view 数量，每个样本有 24 个投影视角；第3个每个视角上投影点的数量，这是因为 lr_res=32，低分辨率网格点数为 32^3 = 32768；第4个，2表示每个点的 2D 投影坐标，即有（x，y）两个分量
                    fusion='max',                                   # 聚合策略。多个视角观察同一个 3D 点时，取特征通道的最大值
                    view_mask=view_mask                             # 再finetune_s1中，view_mask是None
                )
                n_res = np.power(feats_3d.shape[-1], 1 / 3)         # 计算3D特征图feats_3d的最后一维的立方根，得到n_res，即3D特征图的分辨率。
                n_res = int(np.round(n_res))                        # 将n_res四舍五入为整数，确保其为一个有效的分辨率值。
                feats_3d = feats_3d.reshape(*feats_3d.shape[:2], n_res, n_res, n_res)   # 将3D特征图feats_3d的最后一维重新reshape为一个立方体，得到一个新的张量feats_3d，shape为[B, C', n_res, n_res, n_res]，其中B表示batch_size，C'表示编码器输出的通道数，n_res表示3D特征图的分辨率。finetune_s1下为32×32×32
                feats_3d_lists[i].append(feats_3d)                  # 将当前视图的3D特征图feats_3d添加到feats_3d_lists列表中对应索引i的子列表中，以便后续处理。如此反复，获得32×32×32到（中间有64，128）256×256×256的3D体素信息.因为没有mask掩码，所以这些都存在feats_3d_lists索引为0的列表元素中，即[[32×32×32,...,2562×256×256]]

        gpu_timer_qvf_all_end.record()
        torch.cuda.synchronize()
        print(f"[GPU计时] 4尺度2D→3D反投影(query_view_feats) 总耗时 {gpu_timer_qvf_all.elapsed_time(gpu_timer_qvf_all_end):.3f} ms")

        if len(feats_3d_lists) == 1:                                # 如果feats_3d_lists的长度为1，说明只有一个视图掩码，则将feats_3d_lists的第一个元素（即唯一的子列表）赋值给feats_3d_lists，以便后续处理。
            feats_3d_lists = feats_3d_lists[0]
        
        if require_2d:                                              # 如果require_2d为True，则返回feats_3d_lists和feats_2d_list两个列表，分别包含每个视图掩码对应的3D特征图和每个视图的2D特征图。否则，只返回feats_3d_lists列表，包含每个视图掩码对应的3D特征图。
            # return feats_2d_list
            return feats_3d_lists, feats_2d_list                    # model_v7的Model_mv类的forward函数的第一个else，传入的 require_2d=True。返回的数据类型如下，feats_3d_lists的长度为4，第一个元素shape为torch.Size([2, 128, 32, 32, 32])，最后一个为torch.Size([2, 16, 32, 32, 32]).feats_2d_lists也是4长度，元素的shape从torch.Size([2, 24, 128, 32, 32])到torch.Size([2, 24, 16, 256, 256])
        else:                                                       # 如果require_2d为False，则只返回feats_3d_lists列表，包含每个视图掩码对应的3D特征图。
            return feats_3d_lists


class EncoderV_ct(nn.Module):
    def __init__(self, cfg):
        '''
        它直接在 3D 空间中对低分辨率体积特征进行编码。与多视角版（MV）不同，它不依赖投影几何，而是纯数据驱动。
        它将输入的 32³ 体积，通过 3D 残差块提炼通道数，再通过暴力最大池化，强行压出 16³、8³、4³ 的多尺度金字塔。
        虽然显存友好，但它在空间细节的保留上比 UNet 经典下采样更粗糙，更适合作为粗粒度全局特征的提取器。
        '''
        super().__init__()
        self.encoder = Encoder_base(
            in_ch=1, 
            base_ch=cfg.base_ch,
            n_layer=cfg.n_layer,
            ch_up=cfg.ch_up,
            n_dim='3d'
        )
        self.down = nn.ModuleList()
        for i in range(cfg.n_layer):        # i = 0, 1, 2。池化层堆叠
            self.down.append(Pooling(
                kernel_size=2 ** (i + 1),   # 核大小依次为 2, 4, 8
                mode='max', 
                n_dim='3d'
            ))
    
    @property
    def chs(self):
        return self.encoder.chs

    def forward(self, data):
        x = data['ct']                                          # 假设 data['ct'] 的形状为 [1, 1, 32, 32, 32]（即你配置中的 lr_res=32）。lr指low resolution
        feats_list = self.encoder(x)
        for i in range(len(feats_list)):
            if i > 0:                                           # i=0时不进行下采样
                feats_list[i] = self.down[i - 1](feats_list[i])
        return feats_list                                       # 一个4层金字塔：[16@32³, 32@16³, 64@8³, 128@4³]


if __name__ == '__main__':
    import torch

    model = Encoder_base()
    im = torch.randn(1, 1, 128, 128)
    feats = model(im)
    import pdb; pdb.set_trace()
