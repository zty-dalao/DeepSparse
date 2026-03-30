import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.utils import index_3d
from models.conv_utils import StackedResConv
from models.encoder import EncoderV_ct, EncoderV_mv
from models.point_decoder import PointDecoder
from models.ema_codebook import WrappedEMAVQ3d



class Model_mixed(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder_ct = EncoderV_ct(cfg.encoder)
        self.encoder_mv = EncoderV_mv(cfg.encoder)

        chs = self.encoder_ct.chs
        out_ch = cfg.decoder.out_ch

        self.cb_ct = nn.ModuleList()
        self.cb_mv = nn.ModuleList()
        self.decoders_ct = nn.ModuleList()
        self.decoders_mv = nn.ModuleList()

        n_codebook = cfg.encoder.n_layer + 1
        for i in range(n_codebook):
            self.cb_ct.append(WrappedEMAVQ3d(
                n=cfg.codebook.n_embed, 
                dim=chs[i]
            ))
            self.cb_mv.append(WrappedEMAVQ3d(
                n=cfg.codebook.n_embed, 
                dim=chs[i]
            ))
            self.decoders_ct.append(StackedResConv(
                in_ch=chs[i] if i == 0 else chs[i] + out_ch,
                out_ch=out_ch,
                n_layer=cfg.decoder.n_conv3d,
                n_dim='3d'
            ))
            self.decoders_mv.append(StackedResConv(
                in_ch=chs[i] if i == 0 else chs[i] + out_ch,
                out_ch=out_ch,
                n_layer=cfg.decoder.n_conv3d,
                n_dim='3d'
            ))

        self.point_decoder = PointDecoder(
            channels=[out_ch] + cfg.point_decoder.mlp_chs,
            residual=True,
            use_bn=True
        )

        self.registered_point_keys = ['points_proj', 'points_ct']
        self.enabled_encoders = 'all'

    def encode(self, data):
        loss_kd = 0.
        loss_vq_ct_all = 0.
        loss_vq_mv_all = 0.
        n_layer = 0
        for i, (feats_ct, feats_mv) in enumerate(zip(
                self.encoder_ct(data), self.encoder_mv(data)
            )):
            n_layer += 1
            feats_ct, loss_vq_ct = self.cb_ct[i](feats_ct)
            feats_mv, loss_vq_mv = self.cb_mv[i](feats_mv)

            loss_vq_ct_all += loss_vq_ct
            loss_vq_mv_all += loss_vq_mv

            if i > 0:
                feats_ct = torch.cat([feats_ct, feats_ct_out], dim=1)
                feats_mv = torch.cat([feats_mv, feats_mv_out], dim=1)
            
            feats_ct_out = self.decoders_ct[i](feats_ct)
            feats_mv_out = self.decoders_mv[i](feats_mv)

            loss_kd += F.mse_loss(feats_ct_out.detach(), feats_mv_out)
        
        return {
            'feats_ct': feats_ct_out,
            'feats_mv': feats_mv_out,
            'loss_vq_ct': loss_vq_ct_all / n_layer,
            'loss_vq_mv': loss_vq_mv_all / n_layer,
            'loss_kd': loss_kd / n_layer
        }
    
    def forward_points(self, feats_dict, data):
        if self.enabled_encoders == 'all':
            feats_ct = feats_dict['feats_ct']
            p_feats_ct = index_3d(feats_ct, data['points_ct'])
            p_pred_ct = self.point_decoder(p_feats_ct)

            feats_mv = feats_dict['feats_mv']
            p_feats_mv = index_3d(feats_mv, data['points_ct'])
            p_pred_mv = self.point_decoder(p_feats_mv)

            return p_pred_ct, p_pred_mv
        else:
            if self.enabled_encoders == 'ct':
                feats = feats_dict['feats_ct']
            else: feats = feats_dict['feats_mv']
            
            p_feats = index_3d(feats, data['points_ct'])
            p_pred = self.point_decoder(p_feats)
            
            return p_pred

    def forward(self, data, is_eval=False, eval_npoint=100000):
        feats_dict = self.encode(data) # these features are shared for any sampled 3D points

        if not is_eval:
            p_pred_ct, p_pred_mv = self.forward_points(feats_dict, data)
            return {
                'points_pred_ct': p_pred_ct,
                'points_pred_mv': p_pred_mv,
                'loss_vq_ct': feats_dict['loss_vq_ct'],
                'loss_vq_mv': feats_dict['loss_vq_mv'],
                'loss_kd': feats_dict['loss_kd']
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


class Model_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.init_encoder(cfg.encoder)

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
            channels=[out_ch] + cfg.point_decoder.mlp_chs,
            residual=True,
            use_bn=True
        )

        self.registered_point_keys = ['points_proj', 'points_ct']

    def init_encoder(self, cfg):
        # self.encoder will be initialized here
        pass

    def encode(self, data):
        loss_vq_all = 0.
        n_layer = 0
        for i, feats in enumerate(self.encoder(data)):
            n_layer += 1
            feats, loss_vq = self.cb[i](feats)
            loss_vq_all += loss_vq

            if i > 0:
                feats = torch.cat([feats, feats_out], dim=1)
            
            feats_out = self.decoders[i](feats)
        
        return {
            'feats': feats_out,
            'loss_vq': loss_vq_all / n_layer
        }
    
    def forward_points(self, feats_dict, data):
        feats = feats_dict['feats']
        p_feats = index_3d(feats, data['points_ct'])
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
        

class Model_ct(Model_base):
    def init_encoder(self, cfg):
        self.encoder = EncoderV_ct(cfg)


class Model_mv(Model_base):
    def init_encoder(self, cfg):
        self.encoder = EncoderV_mv(cfg)
