import torch.nn as nn



class Pooling(nn.Module):
    def __init__(self, in_ch=None, kernel_size=2, mode='max', n_dim='2d'):
        '''
        池化层：
        如果是max最大池化，则使用使用最大池化层
        如果是avg池化，则使用平均池化层
        如果是conv卷积池化，则使用卷积层+批归一化+激活函数
        '''
        super().__init__()

        assert n_dim in ['2d', '3d'], f'Invalid n_dim: {n_dim}'

        if mode == 'max':
            self.layer = (nn.MaxPool2d if n_dim == '2d' else nn.MaxPool3d)(kernel_size)
        elif mode == 'avg':
            self.layer = (nn.AvgPool2d if n_dim == '2d' else nn.AvgPool3d)(kernel_size)
        elif mode == 'conv':
            self.layer = nn.Sequential(
                (nn.Conv2d if n_dim == '2d' else nn.Conv3d)(in_ch, in_ch, kernel_size, stride=kernel_size),
                (nn.BatchNorm2d if n_dim == '2d' else nn.BatchNorm3d)(in_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            raise ValueError(f'Invalid pooling mode: {mode}')
        
    def forward(self, x):
        return self.layer(x)


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, n_dim='2d'):
        '''
        基础卷积块：
        2d卷积/3d卷积 + 2d批归一化。
        '''
        super().__init__()

        assert n_dim in ['2d', '3d'], f'Invalid n_dim: {n_dim}'

        self.conv = nn.Sequential(
            (nn.Conv2d if n_dim == '2d' else nn.Conv3d)(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.BatchNorm2d if n_dim == '2d' else nn.BatchNorm3d)(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class ResConv(nn.Module):
    def __init__(self, in_ch, n_dim='2d'):
        '''
        残差卷积块
        默认情况下是2维的
        输入维度等于输出维度，输入尺寸等于输出尺寸（通过padding=1 ； kernel_size=3）
        conv2d/conv3d + batchnorm2d/batchnorm3d + leakyrelu
        '''
        super().__init__()

        assert n_dim in ['2d', '3d'], f'Invalid n_dim: {n_dim}'

        self.conv = nn.Sequential(
            (nn.Conv2d if n_dim == '2d' else nn.Conv3d)(in_ch, in_ch, kernel_size=3, padding=1),
            (nn.BatchNorm2d if n_dim == '2d' else nn.BatchNorm3d)(in_ch)
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x_ = self.conv(x)
        return self.relu(x_ + x)


class StackedResConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, n_layer=1, n_dim='2d'):
        '''
        如果in_ch!=out_ch，则先用一个卷积层把输入通道数变成输出通道数，再堆叠 n_layer 个残差卷积块（ResConv）。否则只堆叠 n_layer 个残差卷积块。

        in_ch：输入特征图的通道数（比如输入是单通道 CT 灰度图，in_ch=1）。
        out_ch=None：输出通道数。如果为 None，则默认等于 in_ch（即不改变通道数）。
        n_layer=1：内部堆叠的残差块（ResConv）的数量。设为 3 就代表连续过 3 个残差块，用于增加网络深度和感受野。
        n_dim='2d'：医学影像的灵魂参数。指定处理的是 2D 切片还是 3D 立体（CT 体素）
        '''
        super().__init__()

        if out_ch is None:  # 默认输入输出维度相等
            out_ch = in_ch

        assert n_dim in ['2d', '3d'], f'Invalid n_dim: {n_dim}'

        if out_ch != in_ch:                                                                             # 如果不等于，则先用一个卷积层把输入通道数变成输出通道数
            self.pre_conv = nn.Sequential(                                                              # 2d的是 2d卷积+2d批归一化+激活，3d的是 3d卷积+3d批归一化+激活
                (nn.Conv2d if n_dim == '2d' else nn.Conv3d)(in_ch, out_ch, kernel_size=3, padding=1),   
                (nn.BatchNorm2d if n_dim == '2d' else nn.BatchNorm3d)(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else: self.pre_conv = None                                                                      # 相等则不需要预卷积层，直接用残差块堆叠即可

        self.layers = nn.Sequential(*[                                                                  # 堆叠 n_layer 个残差块，输入输出通道数都是 out_ch
            ResConv(out_ch, n_dim=n_dim) for _ in range(n_layer)                                        # ResConv残差卷积输出尺寸=输出尺寸不变
        ])
    
    def forward(self, x):
        if self.pre_conv:
            x = self.pre_conv(x)
        return self.layers(x)

