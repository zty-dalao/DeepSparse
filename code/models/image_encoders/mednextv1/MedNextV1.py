import torch.nn as nn
from models.image_encoders.mednextv1.blocks import MedNeXtBlock, MedNeXtDownBlock



class MedNeXt(nn.Module):
    def __init__(self, 
        in_ch=1, 
        base_ch=16, 
        n_layer=3, 
        ch_up=1.7,
        n_blocks=2, 
        exp_r: int = 4,              # Expansion ratio as in Swin Transformers
        kernel_size: int = 3,
        do_res: bool = True,         # Can be used to individually test residual connection
        do_res_up_down: bool = True, # Additional 'res' connection on up and down convs
    ):
        super().__init__()

        dim = '2d'
        conv = nn.Conv2d
        norm_type = 'group'
        grn = False
            
        self.stem = nn.Sequential(
            conv(in_ch, base_ch, kernel_size=1),
            nn.Sequential(*[
                MedNeXtBlock(
                    in_channels=base_ch,
                    out_channels=base_ch,
                    exp_r=exp_r,
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn
                )
                for _ in range(n_blocks)]
            )
        )
        
        chs = [base_ch]
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            mid_ch = int(chs[-1] * ch_up)
            layer = nn.Sequential(
                MedNeXtDownBlock(
                    in_channels=chs[-1],
                    out_channels=mid_ch,
                    exp_r=exp_r,
                    kernel_size=kernel_size,
                    do_res=do_res_up_down,
                    norm_type=norm_type,
                    dim=dim
                ),
                nn.Sequential(*[
                    MedNeXtBlock(
                        in_channels=mid_ch,
                        out_channels=mid_ch,
                        exp_r=exp_r,
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        dim=dim,
                        grn=grn
                    )
                    for _ in range(n_blocks)]
                )
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
