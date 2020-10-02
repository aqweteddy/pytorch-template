import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attn import AdditiveAttention, AspectAttention, LuongAttention
from model.autoencoder import AutoEncoder


class DocEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, output_size, v_size, dropout, asp_cnt, is_concat, pretrained=None, aspects_embed=None):
        super(DocEncoder, self).__init__()
        if pretrained is None:
            print('testing')
            self.embedding = nn.Embedding(200, embed_size) # for unit_test
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)

        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.autoencoder = AutoEncoder(embed_size, asp_cnt=asp_cnt, v_size=v_size, aspects_embed=aspects_embed)
        self.is_concat = is_concat
        if is_concat:
            # self.attn2 = AdditiveAttention(embed_size, v_size)
            self.attn2 = LuongAttention(embed_size)
        else:
            self.attn2 = AspectAttention(embed_size, v_size)
            # self.attn2 = LuongAttention(embed_size)
        # self.proj = nn.Linear(embed_size, output_size)

    def forward(self, x, loss_fl=True):
        embed = self.drop(self.embedding(x))
        embed = embed.permute(1, 0, 2)
        outputs_1, _ = self.mha(embed, embed, embed)
        outputs_1 = outputs_1.permute(1, 0, 2)
        
        # loss_fl is False: loss=-1
        _, latent_asp, loss = self.autoencoder(embed.permute(1, 0, 2), loss_fl) # [B, S], [1]
        if not self.is_concat:
            outputs, score = self.attn2(latent_asp, outputs_1)
            return outputs, score, loss
        else:
            # outputs, score = self.attn2(outputs_1)
            outputs, score = self.attn2(outputs_1.mean(1), outputs_1)
            return torch.cat([outputs, latent_asp], dim=-1), score, loss

    def get_ae_aspects(self):
        E = F.normalize(self.embedding.weight, 1)
        T = F.normalize(self.autoencoder.aspect_embed.weight, 1)
        proj = torch.mm(E, T.t()).t()
        return torch.sort(proj, dim=1)
