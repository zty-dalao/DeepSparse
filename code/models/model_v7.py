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

        chs = self.encoder.chs                          # chs=[128,64,32,16]，分别对应每一层的输出通道数
        out_ch = cfg.decoder.out_ch
        self.is_finetune = cfg.get('finetune', False)

        self.cb = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if self.is_finetune:
            self.denoise_layers = nn.ModuleList()
        # 为每个尺度添加一个码本
        for i in range(cfg.encoder.n_layer + 1):
            self.cb.append(WrappedEMAVQ3d(              # WrappedEMAVQ3d 是一个 3D 版本的 EMA VQ（Exponential Moving Average Vector Quantization）模块，用于对 3D 特征图进行向量量化。
                n=cfg.codebook.n_embed, 
                dim=chs[i]
            ))
            if self.is_finetune:
                self.denoise_layers.append(StackedResConv(in_ch=chs[i], n_layer=2, n_dim='3d'))
            self.decoders.append(StackedResConv(                # StackedResConv 是一个堆叠的残差卷积块（Residual Convolutional Block），用于对 3D 特征图进行解码。
                in_ch=chs[i] if i == 0 else chs[i] + out_ch,    # 如果是第一层（i=0），输入通道数为 chs[i]；否则，输入通道数为 chs[i] + out_ch（即当前层的输出通道数加上上一层的输出通道数）。第一层的in_ch=chs[0]=128，第二层的in_ch=chs[1]+out_ch=64+128=`91`，第三层的in_ch=chs[2]+out_ch=32+128=160，第四层的in_ch=chs[3]+out_ch=16+128=144。
                out_ch=out_ch,
                n_layer=cfg.decoder.n_conv3d,
                n_dim='3d'
            ))

        in_ch = out_ch + sum(chs)                       # out_ch=128，sum(chs)=128+64+32+16=240，in_ch=128+240=368
        self.point_decoder = PointDecoder(
            channels=[in_ch, in_ch // 2] + cfg.point_decoder.mlp_chs,   # channels=[368, 184, 128, 64, 16, 1]
            residual=True,
            use_bn=True
        )

        self.registered_point_keys = ['points_proj', 'points_ct']   # 这一行的作用是

    def init_encoder(self, cfg):
        self.encoder = EncoderV_mv(cfg) # 负责“从多视角投影图像生成 3D 特征”的模块

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
            feats_3d, feats_2d = self.encoder(data, require_2d=True)    # 得到的数据类型如下，feats_3d_lists的长度为4，第一个元素shape为torch.Size([2, 128, 32, 32, 32])，最后一个为torch.Size([2, 16, 32, 32, 32]).feats_2d_lists也是4长度，元素的shape从torch.Size([2, 24, 128, 32, 32])到torch.Size([2, 24, 16, 256, 256])
            
            # ============================================================
            # [GPU计时] codebook量化 + 3D decoder跨尺度融合
            # 测量4个尺度的codebook VQ + cat拼接 + 3D decoder的总耗时
            # ============================================================
            gpu_timer_cb = torch.cuda.Event(enable_timing=True)
            gpu_timer_cb_end = torch.cuda.Event(enable_timing=True)
            gpu_timer_cb.record()

            for i, feats in enumerate(feats_3d):                        # 循环4此
                n_layer += 1

                # codebook
                feats, loss_vq = self.cb[i](feats)                      # 此处进行codebook操作：替换。返回的是经过码本替换后的体素空间(feats.shape不变)和码本损失（是个数）。输入的feats数据类型如下，第1次为torch.Size([2, 128, 32, 32, 32])，最后一次为torch.Size([2, 16, 32, 32, 32])
                loss_vq_all += loss_vq

                # decoder
                if i > 0:
                    feats = torch.cat([feats, feats_out], dim=1)        # 第一个不用维度方向上拼接，直接进行decoder获得体素信息。第2个的直接在维度上与前面已经融合的好的3d体素在维度上拼接，下一行再进行融合。如此反复，从128channel向16channel融合.
                feats_out = self.decoders[i](feats)                     # 进行decode，获取真正的体素信息。feats_out.shape 总是等于 torch.Size([2, 128, 32, 32, 32])
            
            gpu_timer_cb_end.record()
            torch.cuda.synchronize()
            print(f"[GPU计时] codebook量化 + 3D decoder融合耗时 {gpu_timer_cb.elapsed_time(gpu_timer_cb_end):.3f} ms")

        return {
            'feats_2d': feats_2d,               # feats_2d_lists也是4长度，元素的shape从torch.Size([2, 24, 128, 32, 32])到torch.Size([2, 24, 16, 256, 256])
            'feats_3d': feats_out,              # feats_3d_lists的长度为4，第一个元素shape为torch.Size([2, 128, 32, 32, 32])，最后一个为torch.Size([2, 16, 32, 32, 32])
            'loss_vq': loss_vq_all / n_layer    # 返回32^3下，不同channel的（128，64，32，16）平均码本损失
        }
    
    def forward_points(self, feats_dict, data):

        # ============================================================
        # [GPU计时] index_3d: 从3D特征体(128,32,32,32)中按点坐标采样
        # ============================================================
        gpu_timer_idx3d = torch.cuda.Event(enable_timing=True)
        gpu_timer_idx3d_end = torch.cuda.Event(enable_timing=True)
        gpu_timer_idx3d.record()

        p_feats = index_3d(feats_dict['feats_3d'], data['points_ct'])   # 返回值 p_feats的shape为[2, 128, 10000]。data['points_ct']是一个稀疏采样点
        
        gpu_timer_idx3d_end.record()
        torch.cuda.synchronize()
        print(f"[GPU计时] index_3d 3D体素→稀疏点特征耗时 {gpu_timer_idx3d.elapsed_time(gpu_timer_idx3d_end):.3f} ms")


        # ============================================================
        # [GPU计时] query_view_feats ×4: 2D多视角特征→稀疏点特征
        # 将10000个训练点从4个尺度的2D多视角特征图中采样并拼接为368维
        # ============================================================
        gpu_timer_qvf = torch.cuda.Event(enable_timing=True)
        gpu_timer_qvf_end = torch.cuda.Event(enable_timing=True)
        gpu_timer_qvf.record()

        for feats_2d in feats_dict['feats_2d']:     # len(feats_dict['feats_2d'])=4。元素的shape从torch.Size([2, 24, 128, 32, 32])到torch.Size([2, 24, 16, 256, 256])
            p_feats_2d = query_view_feats(
                view_feats=feats_2d, 
                points_proj=data['points_proj'],
                fusion='max',                       # 采用最大融合
                view_mask=data['view_mask']         # data['view_mask']的作用是屏蔽无效的填充视图，确保只有真实采样的视图参与特征聚合。
            )
            p_feats = torch.cat([p_feats_2d, p_feats], dim=1)

        gpu_timer_qvf_end.record()
        torch.cuda.synchronize()
        print(f"[GPU计时] query_view_feats×4 2D多视角→稀疏点特征耗时 {gpu_timer_qvf.elapsed_time(gpu_timer_qvf_end):.3f} ms")

        # ============================================================
        # [GPU计时] PointDecoder MLP: 368维点特征→1维HU值预测
        # ============================================================
        gpu_timer_mlp = torch.cuda.Event(enable_timing=True)
        gpu_timer_mlp_end = torch.cuda.Event(enable_timing=True)
        gpu_timer_mlp.record()

        p_pred = self.point_decoder(p_feats)        # 进入point_decoder.py 进行点查询，将368channel消去，得到1w个点真实的HU值   p_feats.shape = torch.Size([2, 128, 10000])。p_pred.shape = torch.Size([2, 1, 10000])
        
        gpu_timer_mlp_end.record()
        torch.cuda.synchronize()
        print(f"[GPU计时] PointDecoder MLP 368→1 HU预测耗时 {gpu_timer_mlp.elapsed_time(gpu_timer_mlp_end):.3f} ms")

        return p_pred

    def forward(self, data, is_eval=False, eval_npoint=100000):
        feats_dict = self.encode(data) # these features are shared for any sampled 3D points。返回见该文件MOdel_mv类的decode函数的返回。第115行

        if not is_eval:                                                 # 训练阶段下，不用评估，即is_eval=False
            return {
                'points_pred': self.forward_points(feats_dict, data),   # 返回见该文件MOdel_mv类的decode，得到1w个点真实的HU值   self.forward_points(feats_dict, data)得到的变量的shape = torch.Size([2, 128, 10000])
                'loss_vq': feats_dict['loss_vq']
            }
        else:
            # ============================================================
            # [GPU计时] 完整256³评估总耗时(encode + 分批forward_points)
            # ============================================================
            gpu_timer_eval = torch.cuda.Event(enable_timing=True)
            gpu_timer_eval_end = torch.cuda.Event(enable_timing=True)
            gpu_timer_eval.record()

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

            gpu_timer_eval_end.record()
            torch.cuda.synchronize()
            print(f"[GPU计时] 完整256³评估总耗时(encode+{n_batch}批forward_points) {gpu_timer_eval.elapsed_time(gpu_timer_eval_end):.0f} ms = {gpu_timer_eval.elapsed_time(gpu_timer_eval_end)/1000:.1f} s")

            return {
                'points_pred': torch.cat(pred_list, dim=2)
            }
        