from dataset import TrainDataset, TestDataset
from torch.utils import data
from tqdm import tqdm
from gensim.models import Word2Vec
from default_config import hparams
from model import NRMS
import torch


def ds():
    print('loading w2v...')
    w2v = Word2Vec.load('./data/w2v/news_200d.bin')
    w2id = {key: w2v.wv.vocab[key].index for key in w2v.wv.vocab}

    # ds = TrainDataset('./data/small_train/news.tsv', './data/small_train/behaviors.tsv', w2id, 4, 50)
    ds = TestDataset('./data/small_dev/news.tsv', './data/small_dev/behaviors.tsv', w2id, 4, 50)
    loader = data.DataLoader(ds, batch_size=32, num_workers=10)
    print(len(ds))
    print(ds[1])
    for d in tqdm(loader):
        pass
    print(ds[1])
# ds()

def model():
    nrms = NRMS(hparams['model'])
    clicked = torch.randint(0, 100, (64, 50, 10))
    cand = torch.randint(0, 100, (64, 4, 10))
    nrms(clicked, cand)
model()

def autoencoer():
    from model.autoencoder import AutoEncoder
    m = AutoEncoder(100, 13)
    x = torch.rand(8, 100)
    print(m(x)[1])

# autoencoer()