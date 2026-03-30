from torch import nn
from models.conv_utils import BasicConv, Pooling, StackedResConv



class Encoder_base(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, n_layer=3, ch_up=1.7, n_dim='2d'):
        super().__init__()

        self.stem = BasicConv(in_ch, base_ch, n_dim=n_dim)

        chs = [base_ch]
        
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            mid_ch = int(chs[-1] * ch_up)
            layer = nn.Sequential(
                Pooling(in_ch=chs[-1], kernel_size=2, mode='conv', n_dim=n_dim),
                StackedResConv(chs[-1], mid_ch, n_dim=n_dim)
            )
            self.layers.append(layer)
            chs.append(mid_ch)

        self._chs = chs
    
    @property
    def out_ch(self):
        return self._chs[-1]

    @property
    def chs(self):
        return self._chs[::-1]

    def forward(self, x):
        x = self.stem(x)

        feats_list = [x]
        for layer in self.layers:
            x = layer(x)
            feats_list.append(x)

        return feats_list[::-1]
