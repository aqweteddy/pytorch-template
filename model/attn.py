import torch.nn as nn
import torch.nn.functional as F
import torch


class AdditiveAttention(nn.Module):
    def __init__(self, in_size, v_size) -> None:
        super(AdditiveAttention, self).__init__()

        self.proj = nn.Sequential(nn.Linear(in_size, v_size),
                                  nn.Tanh(),
                                  nn.Linear(v_size, 1, bias=False),
                                  )

    def forward(self, x):
        """attention

        Args:
            x (tensor): B, S, H
        """
        score = torch.softmax(self.proj(x).squeeze(-1), dim=-1)  # B, S
        return torch.bmm(score.unsqueeze(1), x).squeeze(1)

class AspectAttention(nn.Module):
    def __init__(self, in_size, v_size) -> None:
        super(AspectAttention, self).__init__()
        self.wq = nn.Linear(in_size, v_size)
        self.wv = nn.Linear(in_size, v_size)
        self.V = nn.Sequential(nn.Tanh(), nn.Linear(v_size, 1, bias=False))

    
    def forward(self, query: torch.Tensor, value: torch.Tensor):
        """foward

        Args:
            q (tensor): [B, in_size]
            v (tensor): [B, S, in_size]
        """
        query = query.unsqueeze(1).repeat(1, value.size(1), 1) # [B, S, in_size]
        w = self.wq(query) + self.wv(value) # [B, S, v_size]
        score = torch.softmax(self.V(w).squeeze(-1), dim=-1) # [B, S]
        return torch.bmm(score.unsqueeze(1), value).squeeze(1)