import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



class WrappedEMAVQ3d(nn.Module):
    def __init__(self, n, dim):
        super().__init__()

        self.pre_quant = nn.Conv3d(dim, dim, kernel_size=1)
        self.post_quant = nn.Conv3d(dim, dim, kernel_size=1)
        self.codebook = EMAVectorQuantizer(
            n_embed=n,
            embedding_dim=dim
        )
        self.update = True

    def freeze(self):
        self.update = False
    
    def forward(self, x, no_update=False):
        if not self.update:
            no_update = True
        x = self.pre_quant(x)
        x, _, loss = self.codebook(x, no_update=no_update)
        x = self.post_quant(x)
        return x, loss


def gather_tensor(tensor):
    gathered_results = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, tensor)
    gathered_results = torch.stack(gathered_results, dim=0)
    gathered_mean = torch.mean(gathered_results, dim=0)
    return gathered_mean


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta=1, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z, no_update=False):
        # z: [b, c, *]
        # z_q: [b, c, *]
        z_shape = z.shape[2:]
        b, c = z.shape[:2]
        z = z.reshape(b, c, -1)
        n = z.shape[-1]
        z = z.transpose(1, 2) # [b, c, n] -> [b, n ,c]

        assert c == self.codebook_dim, f'inconsistent dimension: {c}, required: {self.codebook_dim}'
        z_flattened = z.reshape(-1, c) # [bn, c]
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # [bn, 1] + [k,] + [bn, k] -> [bn, k]
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update and (not no_update):
            if dist.is_initialized():
                z_flattened = gather_tensor(z_flattened.contiguous())
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
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)

            # EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.reshape(b, n, c).transpose(1, 2) # [bn, c] -> [b, c, n]
        z_q = z_q.reshape(b, c, *z_shape)
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
