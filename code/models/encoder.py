import torch.nn as nn
import numpy as np

from models.conv_utils import Pooling
from models.utils import query_view_feats
from models.image_encoders.base import Encoder_base
from models.image_encoders.mednextv1.MedNextV1 import MedNeXt



class EncoderV_mv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.encoder_type == 'base':
            self.encoder = Encoder_base(
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
        # [B, M, C, W, H]
        x = data['projs']
        b, m = x.shape[:2]
        x = x.reshape(b * m, *x.shape[2:])

        feats_3d_lists = [[] for _ in range(len(view_masks))]
        if require_2d:
            feats_2d_list = []
        for feats in self.encoder(x):
            # [B, M, C', W', H']
            feats = feats.reshape(b, m, *feats.shape[1:])
            if require_2d:
                feats_2d_list.append(feats)
            
            for i, view_mask in enumerate(view_masks):
                # if not require_2d:
                feats_3d = query_view_feats(
                    view_feats=feats,
                    points_proj=data['points_lr_proj'],
                    fusion='max',
                    view_mask=view_mask
                )
                n_res = np.power(feats_3d.shape[-1], 1 / 3)
                n_res = int(np.round(n_res))
                feats_3d = feats_3d.reshape(*feats_3d.shape[:2], n_res, n_res, n_res)
                feats_3d_lists[i].append(feats_3d)

        if len(feats_3d_lists) == 1:
            feats_3d_lists = feats_3d_lists[0]
        
        if require_2d:
            # return feats_2d_list
            return feats_3d_lists, feats_2d_list
        else:
            return feats_3d_lists


class EncoderV_ct(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder_base(
            in_ch=1, 
            base_ch=cfg.base_ch,
            n_layer=cfg.n_layer,
            ch_up=cfg.ch_up,
            n_dim='3d'
        )
        self.down = nn.ModuleList()
        for i in range(cfg.n_layer):
            self.down.append(Pooling(
                kernel_size=2 ** (i + 1), 
                mode='max', 
                n_dim='3d'
            ))
    
    @property
    def chs(self):
        return self.encoder.chs

    def forward(self, data):
        x = data['ct']
        feats_list = self.encoder(x)
        for i in range(len(feats_list)):
            if i > 0:
                feats_list[i] = self.down[i - 1](feats_list[i])
        return feats_list


if __name__ == '__main__':
    import torch

    model = Encoder_base()
    im = torch.randn(1, 1, 128, 128)
    feats = model(im)
    import pdb; pdb.set_trace()
