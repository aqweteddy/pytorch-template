import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attn import AdditiveAttention

class DocEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, output_size, v_size, dropout, pretrained=None):
        super(DocEncoder, self).__init__()
        if pretrained is None:
            print('testing')
            self.embedding = nn.Embedding(200, embed_size) # for unit_test
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)

        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.attn = AdditiveAttention(embed_size, v_size)
        # self.proj = nn.Linear(embed_size, output_size)    

    def forward(self, x):
        embed = self.drop(self.embedding(x)).permute(1, 0, 2)
        outputs, _ = self.mha(embed, embed, embed)
        outputs = self.attn(outputs.permute(1, 0, 2))
        
        return outputs
