import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.utils import index_3d, query_view_feats
from models.conv_utils import StackedResConv, ResConv
from models.encoder import EncoderV_ct, EncoderV_mv
from models.point_decoder import PointDecoder
from models.ema_codebook import WrappedEMAVQ3d



class Model_mv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.init_encoder(cfg.encoder)

        chs = self.encoder.chs
        out_ch = cfg.decoder.out_ch
        self.is_finetune = cfg.get('finetune', False)

        self.cb = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if self.is_finetune:
            self.denoise_layers = nn.ModuleList()

        for i in range(cfg.encoder.n_layer + 1):
            self.cb.append(WrappedEMAVQ3d(
                n=cfg.codebook.n_embed, 
                dim=chs[i]
            ))
            if self.is_finetune:
                self.denoise_layers.append(StackedResConv(in_ch=chs[i], n_layer=2, n_dim='3d'))
            self.decoders.append(StackedResConv(
                in_ch=chs[i] if i == 0 else chs[i] + out_ch,
                out_ch=out_ch,
                n_layer=cfg.decoder.n_conv3d,
                n_dim='3d'
            ))

        in_ch = out_ch + sum(chs)
        self.point_decoder = PointDecoder(
            channels=[in_ch, in_ch // 2] + cfg.point_decoder.mlp_chs,
            residual=True,
            use_bn=True
        )

        self.registered_point_keys = ['points_proj', 'points_ct']

    def init_encoder(self, cfg):
        self.encoder = EncoderV_mv(cfg)

    def freeze_ft(self):
        # encoder
        print('--- freeze encoder')
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ''' codebook
        print('--- freeze codebook')
        for cb in self.cb:
            cb.freeze() # do not update codebook
            for p in cb.parameters(): # freeze pre/post layers
                p.requires_grad = False
        # '''

    def encode(self, data):
        loss_vq_all = 0.
        n_layer = 0

        if self.is_finetune:
            loss_ft_all = 0.
            feats_3d_lists, feats_2d = self.encoder(
                data, 
                require_2d=True, 
                view_masks=[None, data['view_mask']]
            )
            for i, (feats_3d_dense, feats_3d) in enumerate(zip(*feats_3d_lists)):
                n_layer += 1

                # codebook
                # NOTE: loss_vq is not required as the encoder is fixed
                feats_3d_dense, _ = self.cb[i](feats_3d_dense)
                feats_3d, _ = self.cb[i](feats_3d)

                # denoise
                feats_3d = self.denoise_layers[i](feats_3d)
                loss_ft = F.l1_loss(feats_3d, feats_3d_dense.detach())
                loss_ft_all += loss_ft

                # decoder
                if i > 0:
                    feats_3d = torch.cat([feats_3d, feats_out], dim=1)
                feats_out = self.decoders[i](feats_3d)

            # loss_ft
            loss_vq_all += loss_ft_all * 1.0

        else:
            feats_3d, feats_2d = self.encoder(data, require_2d=True)
            for i, feats in enumerate(feats_3d):
                n_layer += 1

                # codebook
                feats, loss_vq = self.cb[i](feats)
                loss_vq_all += loss_vq

                # decoder
                if i > 0:
                    feats = torch.cat([feats, feats_out], dim=1)
                feats_out = self.decoders[i](feats)
        
        return {
            'feats_2d': feats_2d,
            'feats_3d': feats_out,
            'loss_vq': loss_vq_all / n_layer
        }
    
    def forward_points(self, feats_dict, data):
        p_feats = index_3d(feats_dict['feats_3d'], data['points_ct'])

        for feats_2d in feats_dict['feats_2d']:
            p_feats_2d = query_view_feats(
                view_feats=feats_2d, 
                points_proj=data['points_proj'],
                fusion='max',
                view_mask=data['view_mask']
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
        