import pytorch_lightning as pl
from torch.optim import optimizer
from torch import optim
from torch.utils import data

class Lightning(pl.LightningModule):
    def __init__(self, hparams) -> None:
        self.hparams = hparams

    def prepare_data(self) -> None:
        self.ds
    
    def configure_optimizers(self):
        return optimizer

    def train_dataloader(self) ->data.DataLoader:
        return data.DataLoader()
    
    def training_step(self, batch, batch_idx):
        return {'loss': None, 'log': None}
    
    def training_epoch_end(self, outputs):
        return {'progress_bar': None, 'log': None}
    