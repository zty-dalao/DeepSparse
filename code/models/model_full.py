import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.utils import index_3d, index_2d, query_view_feats
from models.conv_utils import StackedResConv
from models.encoder import EncoderV_ct, EncoderV_mv
from models.point_decoder import PointDecoder
from models.ema_codebook import WrappedEMAVQ3d



class Model_mv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = EncoderV_mv(cfg.encoder)

        chs = self.encoder.chs
        out_ch = cfg.decoder.out_ch

        self.cb = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(cfg.encoder.n_layer + 1):
            self.cb.append(WrappedEMAVQ3d(
                n=cfg.codebook.n_embed, 
                dim=chs[i]
            ))
            self.decoders.append(StackedResConv(
                in_ch=chs[i] if i == 0 else chs[i] + out_ch,
                out_ch=out_ch,
                n_layer=cfg.decoder.n_conv3d,
                n_dim='3d'
            ))

        self.point_decoder = PointDecoder(
            channels=[out_ch + sum(chs)] + cfg.point_decoder.mlp_chs,
            residual=True,
            use_bn=True
        )

        self.registered_point_keys = ['points_proj', 'points_ct']

    def encode(self, data):
        loss_vq_all = 0.
        n_layer = 0

        feats_3d, feats_2d = self.encoder(data, require_2d=True)
        for i, feats in enumerate(feats_3d):
            n_layer += 1
            feats, loss_vq = self.cb[i](feats)
            loss_vq_all += loss_vq

            if i > 0:
                feats = torch.cat([feats, feats_out], dim=1)
            
            feats_out = self.decoders[i](feats)
        
        return {
            'feats_3d': feats_out,
            'feats_2d': feats_2d,
            'loss_vq': loss_vq_all / n_layer
        }
    
    def forward_points(self, feats_dict, data):
        feats_3d = feats_dict['feats_3d']
        p_feats = index_3d(feats_3d, data['points_ct'])
        
        for feats_2d in feats_dict['feats_2d']:
            p_feats_2d = query_view_feats(
                view_feats=feats_2d, 
                points_proj=data['points_proj'],
                fusion='max'
            )
            p_feats = torch.cat([p_feats_2d, p_feats], dim=1)

        p_pred = self.point_decoder(p_feats)
        return p_pred

    def forward(self, data, is_eval=False, eval_npoint=100000):
        feats_dict = self.encode(data) # these features are shared for any sampled 3D points

        if not is_eval:
            return {
                'points_pred': self.forward_points(feats_dict, data),
                'loss_vq': feats_dict['loss_vq']
            }
        else:
            total_npoint = data['points_ct'].shape[1]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                
                tmp_data = {}
                for key in data.keys():
                    if key in self.registered_point_keys:
                        tmp_data[key] = data[key][..., left:right, :]
                    else: 
                        tmp_data[key] = data[key]
                
                points_pred = self.forward_points(feats_dict, tmp_data) # B, C, N
                pred_list.append(points_pred)

            return {
                'points_pred': torch.cat(pred_list, dim=2)
            }