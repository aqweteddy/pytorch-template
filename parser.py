import click
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from default_config import hparams
from train import Lightning
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option('--topk', default=5, help='checkpoint top k')
@click.option('--gpu', default=0)
@click.option('--epochs', default=hparams['epochs'])
@click.option('--batch_size', default=hparams['batch_size'])
@click.option('--lr', default=hparams['lr'])
@click.option('--name', default=hparams['name'])
def train(topk, gpu, epochs, **kwargs):
    print(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    click.echo(f'Now GPU: {torch.cuda.get_device_name(0)}')

    hparams = set_hparams(**kwargs)
    model = Lightning(hparams)
    trainer = Trainer(gpus=1,
                      max_epochs=epochs,
                      logger=TensorBoardLogger(save_dir=os.getcwd(),
                                               name=os.path.join('lightning_logs', hparams['name'])),
                      checkpoint_callback=ModelCheckpoint(None,
                                                          monitor='val_loss',
                                                          save_top_k=topk),
                      early_stop_callback=EarlyStopping('val_loss', 0.1),
                      )
    trainer.fit(model)


@cli.command()
@click.option('--ckpt', './test.ckpt', help='checkpoints')
@click.option('--batch_size', default=hparams['batch_size'])
@click.option('--eval_file', default='')
def evalute(ckpt, batch_size):
    model = Lightning.load_from_checkpoint(ckpt)
    pass


def set_hparams(**kwargs):
    for key in kwargs.keys():
        if key in hparams.keys():
            hparams[key] = kwargs[key]
        if key in hparams['model'].keys():
            hparams['model'][key] = kwargs[key]
        if key in hparams['data'].keys():
            hparams['data'][key] = kwargs[key]
    return hparams


if __name__ == '__main__':
    cli()
