import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attn import AdditiveAttention, AspectAttention
from model.autoencoder import AutoEncoder


class DocEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, output_size, v_size, dropout, asp_cnt, pretrained=None):
        super(DocEncoder, self).__init__()
        if pretrained is None:
            print('testing')
            self.embedding = nn.Embedding(200, embed_size) # for unit_test
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)

        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.autoencoder = AutoEncoder(embed_size, asp_cnt=asp_cnt, v_size=v_size)
        self.attn2 = AspectAttention(embed_size, v_size)
        # self.proj = nn.Linear(embed_size, output_size)

    def forward(self, x):
        embed = self.drop(self.embedding(x))
        # embed = embed.permute(1, 0, 2)
        # outputs_1, _ = self.mha(embed, embed, embed)
        # outputs_1 = outputs_1.permute(1, 0, 2)
        x, latent_asp, loss = self.autoencoder(embed) # [B, S], [1]
        # print(embed.shape)
        # print(x.shape, latent_asp.shape)
        # print(outputs_1.shape, latent_asp.shape)
        # outputs = self.attn2(latent_asp, outputs_1)
        return x, loss
