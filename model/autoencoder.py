import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding

class AutoEncoder(nn.Module):
    def __init__(self, asp_cnt, embed_size) -> None:
        super(AutoEncoder, self).__init__()
        self.aspect_embed = nn.Embedding(asp_cnt, embed_size)
        self.reduction = nn.Sequential(nn.Linear(asp_cnt, embed_size), nn.Softmax(-1))

        nn.init.kaiming_normal_(self.aspect_embed)
        
    
    def forward(self, x):
        """x
        x: tensor: [B, E]
        """
        composition = self.reduction(x) # [B, asp_cnt]
        reconstructed = torch.bmm(composition, self.aspect_embed.weight) # [B, embed_size] = [B, asp_cnt] * [asp_cnt, embed_size]
        return reconstructed
    
    def loss(self):
        pass