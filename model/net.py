import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attn import AdditiveAttention
from model.doc_encoder import DocEncoder


class NRMS(nn.Module):
    def __init__(self, hparams, pretrained=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams['embed_size'],
                                      hparams['doc_nhead'],
                                      hparams['doc_encoder_size'],
                                      hparams['doc_v_size'],
                                      hparams['dropout'],
                                      pretrained)
        
        if hparams['user_encoder'] == 'gru':
            self.user_encoder = nn.GRU(hparams['doc_encoder_size'], hparams['doc_encoder_size'], num_layers=1)
        elif hparams['user_encoder'] =='mha':
            self.user_encoder = nn.MultiheadAttention(
            hparams['doc_encoder_size'], hparams['user_nhead'], dropout=hparams['dropout'])
        self.attn = AdditiveAttention(
            hparams['doc_encoder_size'], hparams['click_v_size'])
        self.drop = nn.Dropout(hparams['dropout'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, clicks, cands, labels=None):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        seq_len = clicks.shape[2]

        clicks = clicks.reshape(-1, seq_len)
        cands = cands.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)
        if self.hparams['user_encoder'] == 'mha':
            click_output, _ = self.user_encoder(click_embed, click_embed, click_embed)
        elif  self.hparams['user_encoder'] == 'gru':
            click_output, _ = self.user_encoder(click_embed)

        click_output = self.drop(click_output.permute(1, 0, 2))

        click_repr = self.attn(click_output)
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(
            0, 2, 1)).squeeze(1)  # [B, 1, hid], [B, 10, hid]
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return torch.sigmoid(logits)
