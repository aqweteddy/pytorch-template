import pytorch_lightning as pl
from torch import optim
import numpy as np
from torch.utils import data
from model import NRMS
from gensim.models import Word2Vec
import torch
from dataset import TrainDataset, TestDataset

from sklearn.metrics import roc_auc_score
from metrics import ndcg_score, mrr_score

class Lightning(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(Lightning, self).__init__()
        self.hparams = hparams
        self.w2v = Word2Vec.load(hparams['pretrained'])
        self.nrms = NRMS(hparams['model'], pretrained=torch.tensor(self.w2v.wv.vectors))

    def forward(self):
        pass

    def prepare_data(self) -> None:
        w2id = {key: self.w2v.wv.vocab[key].index for key in self.w2v.wv.vocab}
        d = self.hparams['data']
        self.train_ds = TrainDataset(d['news_tsv'], d['user_tsv'], w2id, d['neg_sample'], d['click_sample'], d['maxlen'])
        self.val_ds = TestDataset(d['dev_news_tsv'], d['dev_user_tsv'], w2id, d['neg_sample'], d['click_sample'], d['maxlen'])
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.nrms.parameters(), self.hparams['lr'], weight_decay=1e-5)
        return optimizer

    def train_dataloader(self) ->data.DataLoader:
        return data.DataLoader(self.train_ds, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=5)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=100, num_workers=10)
    
    def training_step(self, batch, batch_idx):
        click, cand, label = batch
        _, label = label.max(-1)
        loss, _ = self.nrms(click, cand, label)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs])
        log = {'mean_loss': loss.mean().item()}
        return {'progress_bar': log, 'log': log}
    
    def validation_step(self,batch, batch_idx):
        click, cand, label = batch
        with torch.no_grad():
            logits = self.nrms(click, cand) # B, N
        result = {'ndcg5': [], 'ndcg10': [], 'mrr': [], 'rocauc': []}
        logits = logits.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        for l, pred in zip(label, logits):
            result['ndcg5'].append(ndcg_score(l, pred, k=5))
            result['ndcg10'].append(ndcg_score(l, pred, k=10))
            result['mrr'].append(mrr_score(l, pred))
            try:
                result['rocauc'].append(roc_auc_score(l, pred))
            except Exception:
                result['rocauc'] = 0.
        return result
    
    def validation_epoch_end(self, outputs):
        ndcg5 = np.concatenate([o['ndcg5'] for o in outputs]).reshape(-1).mean()
        ndcg10 = np.concatenate([o['ndcg10'] for o in outputs]).reshape(-1).mean()
        mrr = np.concatenate([o['mrr'] for o in outputs]).reshape(-1).mean()
        rocauc = np.concatenate([o['rocauc'] for o in outputs]).reshape(-1).mean()
        self.logger.log_hyperparams(self.hparams)

        log = {'ndcg5': torch.tensor(ndcg5), 'ndcg10': torch.tensor(ndcg10), 'mrr': torch.tensor(mrr), 'rocauc': torch.tensor(rocauc)}
        print()
        return {'progress_bar': log, 'log': log}
