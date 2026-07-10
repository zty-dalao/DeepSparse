from torch import nn
from models.conv_utils import BasicConv, Pooling, StackedResConv



class Encoder_base(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, n_layer=3, ch_up=1.7, n_dim='2d'):
        '''
        提取特征的编码器：
        1. stem：基础卷积块，输入通道数 in_ch，输出通道数 base_ch。
        2. layers：堆叠 n_layer 个卷积层，每一层包括池化和残差卷积块。每一层的输出通道数会根据 ch_up 进行膨胀。
        3. 输出：
            - feats_list：每一层的输出特征图列表，分辨率从低到高排列。
            - chs：每一层的输出通道数列表，
        '''
        super().__init__()

        self.stem = BasicConv(in_ch, base_ch, n_dim=n_dim)

        chs = [base_ch]
        
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            mid_ch = int(chs[-1] * ch_up)                                       # 通道膨胀，每次取最后一个维度，以finetune_s1.yaml为例，第二个参数为channel,[48, 16, 256, 256]->[48, 32, 128, 128]->[48, 64, 64, 64]->[48, 128, 32, 32]
            layer = nn.Sequential(
                Pooling(in_ch=chs[-1], kernel_size=2, mode='conv', n_dim=n_dim),# 池化层，输入维度in_ch为最后一个得到的维度，输出维度为in_ch。但是size变为输入的一半
                StackedResConv(chs[-1], mid_ch, n_dim=n_dim)                    # 没有指明n_layer，默认是1层残差卷积块
            )
            self.layers.append(layer)                                           # 通过nn.ModuleList()把每一层的layer都加入到self.layers中，方便后续forward时迭代调用
            chs.append(mid_ch)                                                  # 向chs列表中加入每一层的输出通道数，方便后续使用

        self._chs = chs
    
    @property                       # @property 是一个内置装饰器，用于将类的方法转换为属性访问形式，从而实现数据封装、参数检查以及简洁调用的统一。它常用于替代传统的 get_xxx() 和 set_xxx() 方法，使代码更优雅且安全。
    def out_ch(self):
        return self._chs[-1]        # 返回最后一层的输出通道数，即编码器的最终输出通道数。

    @property
    def chs(self):
        '''返回编码器每一层的输出通道数列表，并将其反转，使得从最后一层到第一层的通道数顺序排列，方便后续使用。'''
        return self._chs[::-1]      # 返回编码器每一层的输出通道数列表，并将其反转，使得从最后一层到第一层的通道数顺序排列，方便后续使用。

    def forward(self, x):
        x = self.stem(x)            # 先来一个基础卷积块，得到初始特征图。通过BasicConv将channel扩展到16

        feats_list = [x]
        for layer in self.layers:   # 再迭代每一层的layer，依次进行池化和残差卷积块操作
            x = layer(x)
            feats_list.append(x)

        return feats_list[::-1]     # 返回编码器每一层的输出特征图列表，并将其反转。最终特征图列表的分辨率是从低到高排列的，方便后续使用。
                                    # 原本feats_list的参数的shape如下：[48, 16, 256, 256]->[48, 32, 128, 128]->[48, 64, 64, 64]->[48, 128, 32, 32]
                                    # 反转后为：[48, 128, 32, 32]->[48, 64, 64, 64]->[48, 32, 128, 128]->[48, 16, 256, 256]