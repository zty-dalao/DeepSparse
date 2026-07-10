import torch
import torch.nn as nn



class PointDecoder(nn.Module):
    def __init__(self, channels, residual=True, use_bn=True):
        '''
        这段代码定义了一个名为 PointDecoder 的点特征解码器。
        在你的 CT 重建项目 中，它是 “从抽象特征到具体体素值” 的最后一公里——负责将 [B, C, N] 的点云特征映射为 [B, 1, N] 的密度值（或 CT 值），最终重塑回 3D 体积（如 32³）。
        
        channels：列表，定义了每一层的通道数。这里表示：输入 128 维特征 -> 隐藏层 64 -> 隐藏层 16 -> 输出 1 维（体素值）。
        residual=True：启用特殊的“拼接式残差连接”（不是常见的相加，而是拼接）。
        use_bn=True：在中间层使用 BatchNorm 加速收敛
        '''
        super().__init__()

        self.residual = residual
        self.mlps = nn.ModuleList()

        for i in range(len(channels) - 1):
            modules = []
            if i == 0 or not self.residual:
                modules.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1))              # 这里第一层的输入是和p_feats.shape中的368维度一致的
            else:
                modules.append(nn.Conv1d(channels[i] + channels[0], channels[i + 1], kernel_size=1))    # 这里使用了密集残差链接形式，即拼接式残差连接（concatenation residual），需要再forward中持续与原始输入再维度上拼接

            if i != len(channels) - 1:  # 最后一层是纯线性映射
                if use_bn:
                    modules.append(nn.BatchNorm1d(channels[i + 1]))
                modules.append(nn.LeakyReLU(inplace=True))

            self.mlps.append(nn.Sequential(*modules))

    def forward(self, x):
        x_ = x
        for i, m in enumerate(self.mlps):
            if i != 0 and self.residual:
                x_ = torch.cat([x_, x], dim=1)  # 见本文件的26行，用于拼接式残差链接
            x_ = m(x_)
        return x_   # x_.shape = torch.Size([2, 1, 10000])

'''
self.mlp:

ModuleList(
  (0): Sequential(
    (0): Conv1d(368, 184, kernel_size=(1,), stride=(1,))
    (1): BatchNorm1d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01, inplace=True)
  )
  (1): Sequential(
    (0): Conv1d(552, 128, kernel_size=(1,), stride=(1,))
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01, inplace=True)
  )
  (2): Sequential(
    (0): Conv1d(496, 64, kernel_size=(1,), stride=(1,))
    (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01, inplace=True)
  )
  (3): Sequential(
    (0): Conv1d(432, 16, kernel_size=(1,), stride=(1,))
    (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01, inplace=True)
  )
  (4): Sequential(
    (0): Conv1d(384, 1, kernel_size=(1,), stride=(1,))
    (1): BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01, inplace=True)
  )
)
'''