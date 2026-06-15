import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



class WrappedEMAVQ3d(nn.Module):
    '''
    该类是一个封装了 EMA Vector Quantizer 的模块，专门用于 3D 特征图的量化。它包含了量化前的投影、实际的 EMA 量化器，以及量化后的恢复步骤。
    WrappedEMAVQ3d 是一个 3D codebook 量化封装器：
    pre_quant：量化前投影
    codebook：实际 EMA vector quantizer
    post_quant：量化后恢复
    freeze：冻结 codebook 更新
    '''
    def __init__(self, n, dim):
        '''
        n：codebook token 数量（n_embed），即 codebook 中行数。
        dim：输入特征通道数，也是 codebook 向量的维度。
        '''
        super().__init__()

        self.pre_quant = nn.Conv3d(dim, dim, kernel_size=1)     # 定义一个 1x1x1 的 3D 卷积层。
                                                                # 作用是把输入特征通道做一次线性变换，准备进入量化器。
        self.post_quant = nn.Conv3d(dim, dim, kernel_size=1)    # 定义另一个 1x1x1 3D 卷积层。
                                                                #作用是把量化后的特征再映射回原始通道空间。
        self.codebook = EMAVectorQuantizer(                     # 创建 EMAVectorQuantizer 实例。
            n_embed=n,                                          # n_embed=n：codebook 大小。
            embedding_dim=dim                                   # embedding_dim=dim：codebook 向量维度。
        )
        self.update = True                                      # 标志位，控制是否允许 codebook 更新。
                                                                # False 时会在 forward 中把 no_update=True，从而禁止 EMA 更新 

    def freeze(self):
        '''
        冻结方法。
        调用后 self.update=False，使后续前向不再更新 codebook。
        '''
        self.update = False
    
    def forward(self, x, no_update=False):
        '''
        x：输入 3D 特征，形状一般为 [B, C, D, H, W]。
        no_update：是否禁止 codebook 更新，默认 False。
        '''
        if not self.update:                                 # 如果模块被 freeze() 冻结，则强制开启 no_update。
                                                            # 这样即使外部传 no_update=False，也不会更新 codebook。
            no_update = True
        x = self.pre_quant(x)                               # 先通过 pre_quant 卷积层对输入特征进行线性变换，准备进入量化器。
        x, _, loss = self.codebook(x, no_update=no_update)  # 调用 EMAVectorQuantizer 量化输入 x。
                                                            # 返回三个值：
                                                            #   x：量化后特征
                                                            #   _：诊断信息 (perplexity, encodings, encoding_indices)
                                                            #   loss：量化 loss
        x = self.post_quant(x)                              # 通过 post_quant 卷积层把量化后的特征映射回原始通道空间。
        return x, loss


def gather_tensor(tensor):
    '''
    gather_tensor 用于分布式训练时聚合张量：
        收集所有卡上的数据
        计算平均值
        确保 codebook 更新使用全局统计而不是单卡统计
    '''
    gathered_results = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, tensor)
    gathered_results = torch.stack(gathered_results, dim=0)
    gathered_mean = torch.mean(gathered_results, dim=0)
    return gathered_mean


class EmbeddingEMA(nn.Module):
    '''
    定义一个继承自 nn.Module 的类，用于管理 EMA 版本的 codebook embedding。

    EmbeddingEMA 的作用是：维护一个可查询的 codebook，并用 EMA 统计来更新它，而不是梯度更新。
    self.weight：最终的 codebook embedding 矩阵，形状 [num_tokens, codebook_dim]。
    self.cluster_size：每个 codebook 项被选中的频次统计，帮助归一化更新。
    self.embed_avg：每个 codebook 项对应输入向量的 EMA 累计和。
    decay 控制 EMA 平滑程度，eps 用于数值稳定。
    '''
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        '''
        num_tokens：codebook 中向量的数量，也就是词表大小。
        codebook_dim：每个 embedding 向量的维度。
        decay：EMA 衰减系数，值越靠近 1，更新越平滑，历史信息权重越大。
        eps：数值稳定项，防止除 0 或极小值导致不稳定。
        '''
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)                         # 初始化 codebook embedding 的原始值：从标准正态分布随机生成一个矩阵，形状 [num_tokens, codebook_dim]。
                                                                                        # 将 embeddings 包装成参数张量，但 requires_grad=False 表示不通过反向传播直接训练它。
                                                                                        # 这里的 embedding 是“由 EMA 更新”而不是梯度优化。
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)  # cluster_size 是每个 codebook token 的软聚类大小统计，形状 [num_tokens]。
                                                                                        # 初始化为 0，后续通过 EMA 累积每个 token 的选择次数。
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)              # embed_avg 是每个 token 对应的累积输入向量和。
                                                                                        # 初始值复制随机 weight，后续通过 EMA 更新为每个 token 的“输入均值累加”
        self.update = True                                                              # 是否允许更新，冻结后不再更新 EMA 统计。
                                                                                        # 训练过程中可以通过外部设置为 False 使 codebook 停止更新。
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)                                       # forward 方法根据索引 embed_id 返回对应的 embedding 向量

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)    # 这行的作用是直接更新 cluster_size 参数，使用 EMA 方式将 new_cluster_size 融入到当前的 cluster_size 中。 
                                                                                                # 更新方式类似于动量更新

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)          # 这行的作用是直接更新 embed_avg 参数，使用 EMA 方式将 new_embed_avg 融入到当前的 embed_avg 中。

    def weight_update(self, num_tokens):
        '''
        num_tokens：codebook 大小，同 self.weight.shape[0]，即 token 的数量，即行数。
        这里传入是为了计算平滑项时使用。
        '''
        n = self.cluster_size.sum()                                                 # 计算所有 token 的总聚类大小，作为归一化的分母。如第一个用了5，第二个用了7次，加起来就是n=12次   
        smoothed_cluster_size = (                                                   
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n        # 根据当前的 cluster_size 计算平滑后的 cluster_size，加入 eps 防止除零，并进行归一化处理，使得总和仍然为 n。 
                                                                                    # 这里获得的值会比原始值偏小一些，总和还是极为接近n的
        )   
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)      # 用每个 token 的累计向量和除以对应的“平滑簇大小”，得到新的 token 均值。
                                                                                    # unsqueeze(1) 让形状从 [num_tokens] 变成 [num_tokens, 1]，以便广播除法
        self.weight.data.copy_(embed_normalized)                                    # 更新 codebook 实际 embedding weights。
                                                                                    # 将 self.weight 直接替换为归一化后的结果。


class EMAVectorQuantizer(nn.Module):
    '''
    这个类实现了带 EMA 更新的 vector quantization，负责把输入特征映射到 codebook 向量。

    负责把特征 z 映射到最近的 codebook 向量
    负责计算 quantized output
    负责生成 one-hot 编码和 perplexity
    负责 EMA 更新 codebook，不用梯度直接维护 weight
    '''
    def __init__(self, n_embed, embedding_dim, beta=1, decay=0.99, eps=1e-5):
        '''
        n_embed：codebook 的大小，也就是 embedding 的行数。
        embedding_dim：每个 embedding 向量的维度。
        beta：量化 loss 的权重系数，后面用于 loss = beta * mse_loss(...)。
        decay：EMA 更新的衰减系数，越接近 1，EMA 越平滑。
        eps：数值稳定项，防止除零或极小值。
        '''
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z, no_update=False):
        '''
        z：输入特征
        no_update：是否禁止 EMA 更新；True 时只做量化，不更新 codebook。
        '''
        # z: [b, c, *],b 是 batch size，c 是特征维度，* 是空间维度（如 H, W, D 等）。输入的 z 是需要被量化的特征图。
        # z_q: [b, c, *]
        z_shape = z.shape[2:]
        b, c = z.shape[:2]
        z = z.reshape(b, c, -1)
        n = z.shape[-1]         # 把空间维度折成一个维度，得到 [b, c, n]，其中 n 是空间位置总数。
        z = z.transpose(1, 2) # [b, c, n] -> [b, n ,c]，交换1，2维度，即便于后续把每个位置当作一个向量处理。

        assert c == self.codebook_dim, f'inconsistent dimension: {c}, required: {self.codebook_dim}'    # 检查输入通道数是否匹配 codebook 维度。
                                                                                                        # 不匹配则报错，避免后续向量距离计算错误。
        z_flattened = z.reshape(-1, c) # [bn, c].把所有样本和空间位置合并成一个平面向量集合，形状 [b*n, c].
        
        # 算输入向量到每个 codebook vector 的平方距离。
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # [bn, 1] + [k,] + [bn, k] -> [bn, k]
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)                                       # 选出最近的 codebook 索引。
                                                                                        # 每个输入向量对应一个最小距离的 embedding。
        z_q = self.embedding(encoding_indices).view(z.shape)                            # 根据索引查表，得到量化后的向量。
                                                                                        # self.embedding(encoding_indices) 返回 shape [bn, c]，再 reshape 回 [b, n, c]。
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)          # 生成 one-hot 矩阵，shape [bn, num_tokens]。
                                                                                        # 这个矩阵表示每个向量被分配到哪个 codebook entry。

        avg_probs = torch.mean(encodings, dim=0)                                        # 计算每个 token 在当前 batch 中的平均激活概率。
                                                                                        # 等价于该 token 的频率, shape [num_tokens]。
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))    # 计算 codebook 使用的 perplexity。计算 perplexity，衡量 codebook 的使用情况。perplexity 越高，说明越多的 token 被使用，越低说明集中在少数 token 上。
                                                                                        # 这是一个衡量 codebook 向量使用均匀度的指标。

        if self.training and self.embedding.update and (not no_update):                 # 只有在训练模式下，且没有禁用更新时，才进行 EMA 更新。
                                                                                        # self.embedding.update 可以由外部设置成 False，用于冻结 codebook。
            if dist.is_initialized():                                                   # 如果启用了分布式训练，则先把各卡的 z_flattened 聚合平均。
                z_flattened = gather_tensor(z_flattened.contiguous())                   # 这样在多卡训练中，codebook 更新使用全局样本统计。
                '''
                z_flattened_gathered = gather_tensor(z_flattened)
                print('rank: {}, z_flattened: {}, z_flattened_gathered: {}'.format(
                    dist.get_rank(),
                    z_flattened.mean(),
                    z_flattened_gathered.mean()
                ))
                z_flattened = z_flattened_gathered
                '''
            
            # EMA cluster size
            encodings_sum = encodings.sum(0)                        # 计算每个 token 在当前 batch 中被选中的总次数，shape [num_tokens]。
            self.embedding.cluster_size_ema_update(encodings_sum)   # 更新 cluster_size 的 EMA 统计。

            # EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened      # 计算每个 token 选中向量的总和，形状 [num_tokens, c]。
                                                                    # 这是 embedding 更新所需的累加和。
            self.embedding.embed_avg_ema_update(embed_sum)          # 更新 embed_avg 的 EMA 统计。
            
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)           # 用 cluster_size 和 embed_avg 计算新的 codebook weight。
                                                                    # 这一步把 EMA 统计转成实际的 codebook embedding。

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)              # 计算 commitment loss。
                                                                    # z_q.detach() 表示 codebook 不参与这个 loss 的梯度传递，只把梯度传给 encoder 输出 z。


        # preserve gradients
        z_q = z + (z_q - z).detach()                                # 这是 straight-through estimator。
                                                                    # 前向使用 z_q，反向梯度直接传给 z，不传给 quantized output 的离散选择过程。

        # reshape back to match original input shape
        z_q = z_q.reshape(b, n, c).transpose(1, 2)                  # [bn, c] -> [b, c, n].
        z_q = z_q.reshape(b, c, *z_shape)                           # 将量化后的结果恢复成输入的形状 [b, c, *z_shape]。

        '''
        返回三个内容：
            z_q：量化后的特征
            (perplexity, encodings, encoding_indices)：诊断信息
            loss：用于训练的量化 loss
        '''
        return z_q, (perplexity, encodings, encoding_indices), loss
    
    
if __name__ == '__main__':
    decoder = EMAVectorQuantizer(n_embed=256, embedding_dim=128).cuda()
    z = torch.randn(4, 128, 512, 12, 12).cuda()

    y, __, _ = decoder(z)
    per, encoding, inx = __
    
    print(y.size())
    print(per)
    print(encoding.size())
    print(inx.size())
    import pdb; pdb.set_trace()

'''
torch.nn.parameter.Parameter的核心作用是将一个张量“标记”为模型的参数，从而让深度学习框架能够自动识别并管理它。那这里不让进行反向传播进行更新，反而是使用手工编写的函数进行更新，所以为什么不用普通张量进行表达，这样也能更新啊，为什么要用torch.nn.parameter.Parameter呢？

为什么这里用 nn.Parameter 而不是普通 Tensor
nn.Parameter 的核心作用确实是“标记为模型参数”，但它还有一个更基础的角色：把这个张量注册到 nn.Module 中，变成模块可管理的状态。

1. 自动被 nn.Module 识别和管理
nn.Parameter 会被 Module 自动收集到：

model.parameters()
model.named_parameters()
model.state_dict()
即使 requires_grad=False，它仍然是模块的一部分。

2. 自动迁移设备
当你调用 model.cuda() / model.to(device) 时：

nn.Parameter 会自动移动到目标设备
普通 Tensor 不会，除非你自己手动处理
这对 weight, cluster_size, embed_avg 这种状态很重要。

3. 自动保存与加载
nn.Parameter 会被包含进 state_dict()：

torch.save(model.state_dict(), ...)
model.load_state_dict(...)
如果用普通 Tensor，默认不会保存，除非额外注册或手动写。

这里为什么 requires_grad=False 仍然要用 Parameter
因为这几个张量是“模型的状态”，不是“模型的梯度参数”：

weight 是 codebook 矩阵
cluster_size 和 embed_avg 是 EMA 统计值
它们需要：

在训练过程中保存
在不同设备间迁移
在模型加载时恢复
所以用 Parameter(requires_grad=False) 是一种“既可管理又不参与梯度更新”的合理方式。

你说的“用普通张量也能更新”是对的，但不够方便
普通 Tensor 也确实可以：

self.weight = torch.randn(...)
然后手动 self.weight = self.weight.to(device) 等
但这样会失去模块管理能力：

不会自动加入 state_dict
也不会自动随 model.to() 迁移
也不会出现在参数集合中，很多工具和框架假设只有 Parameter 才是模型状态
另外一个选择：register_buffer
如果你想强调“这不是可训练参数”，也可以用：

self.register_buffer('cluster_size', torch.zeros(...))
但这里作者选用 Parameter(requires_grad=False)，可能因为：

代码更直接，self.weight / self.cluster_size 语义清晰
这些值仍然是“模型的可保存状态”
weight 这种本质上和参数类似的矩阵，用 Parameter 更习惯
结论
nn.Parameter(..., requires_grad=False) 的作用是：

把张量注册为模块状态
使其跟 Module 一起迁移、保存、加载
同时不参与反向传播
因此，这里用它比普通 Tensor 更合理。

'''