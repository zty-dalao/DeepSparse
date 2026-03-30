import torch.nn as nn



class Pooling(nn.Module):
    def __init__(self, in_ch=None, kernel_size=2, mode='max', n_dim='2d'):
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
        super().__init__()

        if out_ch is None:
            out_ch = in_ch

        assert n_dim in ['2d', '3d'], f'Invalid n_dim: {n_dim}'

        if out_ch != in_ch:
            self.pre_conv = nn.Sequential(
                (nn.Conv2d if n_dim == '2d' else nn.Conv3d)(in_ch, out_ch, kernel_size=3, padding=1),
                (nn.BatchNorm2d if n_dim == '2d' else nn.BatchNorm3d)(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else: self.pre_conv = None

        self.layers = nn.Sequential(*[
            ResConv(out_ch, n_dim=n_dim) for _ in range(n_layer)
        ])
    
    def forward(self, x):
        if self.pre_conv:
            x = self.pre_conv(x)
        return self.layers(x)

