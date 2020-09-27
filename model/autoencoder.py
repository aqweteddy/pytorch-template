import torch
from torch import nn
from torch.jit import freeze
import torch.nn.functional as F
from model.attn import AdditiveAttention, LuongAttention


class AutoEncoder(nn.Module):
    def __init__(self, embed_size, asp_cnt, v_size, aspects_embed=None) -> None:
        super(AutoEncoder, self).__init__()
        self.embed_size = embed_size
        # self.attention = AdditiveAttention(embed_size, v_size)
        self.attention = LuongAttention(embed_size)
        self.reduction = nn.Sequential(nn.Linear(embed_size, asp_cnt), nn.Softmax(-1))
        if aspects_embed is None:
            self.aspect_embed = nn.Embedding(asp_cnt, embed_size)
            nn.init.kaiming_uniform_(self.aspect_embed.weight)
        else:
            self.aspect_embed = nn.Embedding.from_pretrained(aspects_embed, freeze=False)
    
    def forward(self, x, loss_fl=True):
        """x
        x: tensor: [B, S, E]
        """
        x, _ = self.attention(x.mean(1), x)
        composition = self.reduction(x) # [B, asp_cnt]
        reconstructed = torch.matmul(composition, self.aspect_embed.weight) # [B, embed_size] = [B, asp_cnt] * [asp_cnt, embed_size]
        if not self.training or not loss_fl: # eval, no loss
            return x, reconstructed, 0.
        return x, reconstructed, self.get_loss(F.normalize(reconstructed), x)
        
    def eval(self):
        self.eval_flag = True
        super(AutoEncoder, self).eval()
    
    def get_loss(self, r_s, z_s):
        device = r_s.device
        self.z_n = self.get_neg_samples(z_s)
        pos = torch.bmm(z_s.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
        negs = torch.bmm(self.z_n, r_s.unsqueeze(2)).squeeze()
        J = torch.ones(negs.shape).to(device) - pos.expand(negs.shape) + negs
        loss = torch.mean(torch.clamp(J, min=0.0))
        r =  1 * self.orthogonal_regularization()
        # print(loss, r)
        return r + loss

    def get_neg_samples(self, x):
        batch_size = x.shape[0]
        mask = self._create_neg_mask(batch_size).to(x.device)
        return x.expand(batch_size, batch_size, self.embed_size).gather(1, mask)

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, 10)
        mask = torch.multinomial(multi_weights, neg)
        mask = mask.unsqueeze(2).expand(batch_size, neg, self.embed_size)
        return mask
    
    def orthogonal_regularization(self):
        T_n = F.normalize(self.aspect_embed.weight, dim=1)
        I = torch.eye(T_n.shape[0]).to(T_n.device)
        return torch.norm(T_n.mm(T_n.t()) - I)