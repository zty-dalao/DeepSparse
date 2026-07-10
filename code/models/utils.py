import torch
from torch import nn
import torch.nn.functional as F



def index_3d(feat, uv):
    '''根据给定的浮点坐标，从 3D 特征体积（如 [B, C, D, H, W]）中通过三线性插值“抠”出对应位置的特征向量。
    :param feat: [B, C, H, W, D] image features                             体素信息：长度为4，第一个元素shape为torch.Size([2, 128, 32, 32, 32])，最后一个为torch.Size([2, 16, 32, 32, 32])。
    :param uv: [B, N, 3] uv coordinates in the image plane, range [-1, 1]   实际是data['points_ct']，shape为torch.Size([2, 10000, 3])
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]; 5-d case  。形状变化：[2, 10000, 3] → [2, 10000, 1, 1, 3]。适配grid_sample 的网格格式。原因：grid_sample（3D 版本）要求 grid 必须是 5 维：[B, D_out, H_out, W_out, 3]。这里将 10000 视为输出体积的 深度（D_out），后面的两个 1 视为高度和宽度。相当于构建了一个 10000×1×1 的“点状”采样网格。
    feat = feat.transpose(2, 4)         # 交换“深度”和“高度”的语义（关键步骤）。执行 transpose(2, 4) 后：[B, C, D, W, H]。为什么要交换？grid_sample 对坐标的映射有严格要求：grid 的最后一维 (x, y, z) 必须对应输入张量的 (宽度 W, 高度 H, 深度 D)。在转置后的 [B, C, D, W, H] 中，最后一个维度是 H，倒数第二个维度是 W，第三个维度是 D。这样，(x, y, z) 就能精准命中输入特征的 (W, H, D) 顺序。
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1, 1]。输出形状：[2, 128, 10000, 1, 1]。物理动作：拿着这 10000 个 (x, y, z) 坐标，在 128 个通道的 32³ 特征体积上进行三线性插值，为每个点抠出一个 128 维的向量。
    return samples[:, :, :, 0, 0]                                           # [B, C, N] 。 压缩多余维度。形状变化：[2, 128, 10000, 1, 1] → [2, 128, 10000]。去掉那俩没用的 1 维度，返回干净的“点特征矩阵”。


def index_2d(feat, uv): 
    # https://zhuanlan.zhihu.com/p/137271718.从 2D 特征图 feat 中，按照点集 uv 指定的二维坐标，采样出这些点对应的特征向量。
    # feat: [B, C, H, W]。以finetune_s1 + 第一个视角为例,feat.shape = torch.Size([2, 128, 32, 32])
    # uv: [B, N, 2]       uv.size = torch.Size([2, 32768, 2])                 
    uv = uv.unsqueeze(2) # [B, N, 1, 2] = torch.Size([2, 32768, 1, 2])。PyTorch 的 grid_sample 要求 grid 输入必须是 4 维：[B, H_out, W_out, 2]（2D 情况）。这里将 N（32768）视为输出特征图的高度，将 1 视为宽度。相当于我们构建了一个 32768 行、1 列 的超长网格，每一行对应一个 3D 点的 (x, y) 坐标。
    feat = feat.transpose(2, 3) # [W, H]。size不变，但轴的信息改变了。为什么必须交换？grid_sample 有一个死规矩：grid 的最后一维 (x, y) 必须分别对应输入张量的最后一个维度（宽度 W） 和倒数第二个维度（高度 H）。原始 feat 是 [C, H, W]，如果不转置，x 会错误地映射到 H，y 映射到 W。通过 transpose(2,3)，我们将原始的第 2 轴（H）和第 3 轴（W）互换，强制让 W 成为倒数第一维。这就使得 (x, y) 能精准命中图像的 (列坐标, 行坐标)。
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]。输出形状：输出形状：torch.Size([2, 128, 32768, 1])。发生了什么？grid_sample 拿着这 32768 个 (x, y) 浮点坐标（范围需在 [-1, 1] 内），在 32×32 的特征图上做双线性插值。因为 uv 的 “高度” 是 32768，“宽度” 是 1，所以输出的高度也是 32768，宽度是 1。结果就是：对于每一个 3D 点，我们都从 128 个通道的特征图中各抠出了一个插值后的响应值。
    return samples[:, :, :, 0] # [B, C, N]。samples[:, :, :, 0]压缩多余的宽度，[2, 128, 32768, 1] → [2, 128, 32768]。


def query_view_feats(view_feats, points_proj, fusion='max', view_mask=None):
    '''
    将多张 2D 特征图上的信息，按照 3D 点的投影坐标“反投”并“聚合”到三维空间中，生成 3D 体素特征。
    
    view_feats: [B, M, C, H, W]   在你的场景中，B=1, M=24, C=16/32/64/128, H/W=128/64/32/16。这是 2D 编码器生成的 24 个视角的特征图（其中后 14 个是无效填充视图）。
    points_proj: [B, M, N, 2]     N = lr_res^3 = 32768。这是 32³ 个 3D 体素点在 24 个 2D 图像平面上的投影像素坐标 (u, v)。
    fusion='max'：聚合策略。多个视角观察同一个 3D 点时，取特征通道的最大值
    view_mask：形状 [B, M]。例如 [True]*10 + [False]*14，用于屏蔽无效的填充视图。
    view_length (optional): [B, M]
    output: [B, C, N, M] for fusion = None, or [B, C, N] for fusion = 'max'
    '''
    n_view = view_feats.shape[1]                                                # finetune_s1中，view_feats.shape = torch.Size([2, 24, 128, 32, 32])。n_view = 24，表示有 24 个视角的特征图。   [Batch, Views, Channels, Height, Width]
    p_feats_list = []                                                           # 创建一个空列表 p_feats_list，用于存储每个视角的特征图在投影点上的特征向量
    for i in range(n_view):
        feat = view_feats[:, i, ...]                                            # B, C, W, H。 feat 是第 i 个视角的特征图，形状为 [B, C, W, H]，其中 B=1, C=16/32/64/128, W/H=128/64/32/16。finetune_s1来说，feat.shape = torch.Size([2, 128, 32, 32])
        p = points_proj[:, i, ...]                                              # B, N, 2。 p 是第 i 个视角的投影点坐标，形状为 [B, N, 2]，其中 N=lr_res^3=32768。points_proj.shape = torch.Size([2, 24, 32768, 2])。p.size = torch.Size([2, 32768, 2])
        p_feats = index_2d(feat, p)                                             # B, C, N。 p 是第 i 个视角的特征图在投影点上的特征向量，形状为 [B, C, N]，其中 B=1, C=16/32/64/128, N=lr_res^3=32768。p_feats.shape = [2, 128, 32768],可解释为：从单个视角上得到3D低分辨率空间的一个点对应一个128维的特征向量。于是我们得到了32768个向量。这个128维的向量将会用于码本。
        p_feats_list.append(p_feats)                                            # 将第 i 个视角的特征向量 p_feats 添加到列表 p_feats_list 中，最终 p_feats_list 的长度为 n_view=24。 len(p_feats_list)=24
    p_feats = torch.stack(p_feats_list, dim=-1)                                 # B, C, N, M。torch.stack它将一个张量列表，沿着一个全新创建的维度“叠”在一起，形成一个维度更高的张量。输出：形状变为 [2, 128, 32768, 24]。
    if fusion == 'max':                                                         # 如果 fusion='max'，则对所有视角的特征向量取最大值，得到最终的 3D 特征向量
        if view_mask is not None:                                               # 如果 view_mask 不为 None，则将无效视角的特征向量置为负无穷大，以便在取最大值时被忽略
            view_mask = view_mask.unsqueeze(1).unsqueeze(1).expand_as(p_feats)  # B, C, N, M。2个 unsqueeze 是为了把 view_mask 的维度扩展到与 p_feats 相同，方便后续的 masked_fill 操作。expand_as(p_feats) 是为了把 view_mask 的维度扩展到与 p_feats 相同，方便后续的 masked_fill 操作。
            view_mask = view_mask.bool()                                        # 将 view_mask 转换为布尔类型，以便在 masked_fill 操作中使用
            p_feats = p_feats.masked_fill(~view_mask, float('-inf'))            # 将无效视角的特征向量置为负无穷大，以便在取最大值时被忽略
            
        p_feats = F.max_pool2d(p_feats, (1, p_feats.shape[-1]))                 # B, C, N, 1。对 p_feats 的最后一个维度（即视角维度）进行最大池化，得到最终的 3D 特征向量，形状为 [B, C, N, 1]，其中 B=1, C=16/32/64/128, N=lr_res^3=32768。第一个维度 1：表示在“点数维（N）”上，窗口高度为 1（不跨点）。第二个维度 p_feats.shape[-1]：即 24，表示在“视角维（M）”上，窗口宽度覆盖整个 24 个视角。默认 stride = kernel_size：因此滑动步长也是 (1, 24)。最终输出为p_feats.shape：[2, 128, 32768, 1]
        p_feats = p_feats.squeeze(-1)                                           # [B, C, N]。squeeze(-1) 是为了去掉最后一个维度，使得最终的 3D 特征向量的形状为 [B, C, N]，其中 B=1, C=16/32/64/128, N=lr_res^3=32768。p_feats.shape = [2, 128, 32768]
    elif fusion is not None:
        raise NotImplementedError
    return p_feats                                                              # 返回的p_feats.shape = [2, 128, 32768]


class MLP_1d(nn.Module):
    def __init__(self, mlp_list, use_bn=False, last_bn=True, last_act=True):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv1d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn and (last_bn or i < len(mlp_list) - 2):
                layers += [nn.BatchNorm1d(mlp_list[i + 1])]
            if last_act or i < len(mlp_list) - 2:
                layers += [nn.LeakyReLU(inplace=True)]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
