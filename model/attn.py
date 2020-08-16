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
