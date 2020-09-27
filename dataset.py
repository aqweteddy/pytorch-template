from os import nice
from typing import List
from torch.utils import data
import pandas as pd
import random
import torch
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize

class TrainDataset(data.Dataset):
    def __init__(self, news_tsv, user_tsv, w2id, neg_sample=4, click_sample=50, maxlen=15) -> None:
        self.w2id = w2id
        self.maxlen = maxlen
        self.neg_sample = neg_sample
        self.click_sample = click_sample
        print('Reading data...')
        news_df = pd.read_csv(news_tsv, sep='\t', header=None)
        news_df.columns = ['news_id', 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entity', 'abstract_entity']
        self.news = news_df[['news_id', 'title']]
        self.news.loc[len(self.news)] = ['0', '']
        self.news = self.news.set_index('news_id')
        user_df = pd.read_csv(user_tsv, header=None, sep='\t') 
        user_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        print('Preparing...')
        self.click_ev = self.prepare(user_df)
        print(f'dataset_size:{len(self.click_ev)}')

    def prepare(self, user_df):
        click_ev = []
        for histories, impressions in zip(user_df['history'], user_df['impressions']):
            try:
                histories = histories.strip().split(' ')
            except AttributeError:
                continue
            pos_list, neg_list = [], []
            # gen candidate
            # print(impressions.strip().split(' '))
            for impression in impressions.strip().split(' '):
                news_id, fl = impression.split('-')
                if fl == '1':
                    pos_list.append(news_id)
                else:
                    neg_list.append(news_id)
            if len(histories) > self.click_sample:
                clicked = histories[:self.click_sample]
                # clicked = random.sample(histories, self.click_sample)
            else:
                clicked = ['0'] * (self.click_sample - len(histories)) + histories
            for pos in pos_list:
                try:
                    sample = random.sample(neg_list, self.neg_sample)
                except Exception:
                    sample = random.choices(neg_list, k=self.neg_sample)
                sample.append(pos)
                sample_idx = [0] * self.neg_sample + [1]
                # print(sample, sample_idx)
                sample_idx, sample = shuffle(sample_idx, sample)
                
                # clicked += ['0'] * (self.click_sample - len(clicked))
                # print(clicked)
                click_ev.append([clicked, sample, sample_idx])
        return click_ev
    
    def __len__(self) -> int:
        return  len(self.click_ev)
    
    def get_title_index(self, index: List[str]):
        def get_idx(sent):
            idx = [self.w2id.get(word.lower(), 0) for word in word_tokenize(sent)]
            idx = idx[:self.maxlen]
            idx += [0] * (self.maxlen - len(idx))
            return idx
        return [get_idx(self.news['title'][idx]) for idx in index]
    
    def __getitem__(self, index: int):
        clicked, sample, sample_idx = self.click_ev[index]
        clicked, sample = self.get_title_index(clicked), self.get_title_index(sample)
        return torch.tensor(clicked), torch.tensor(sample), torch.tensor(sample_idx)


class TestDataset(TrainDataset):
    def __init__(self, *args, **kwargs):
        super(TestDataset, self).__init__(*args, **kwargs)

    def prepare(self, user_df):
        click_ev = []
        for histories, impressions in zip(user_df['history'], user_df['impressions']):
            try:
                histories = histories.strip().split(' ')
            except AttributeError:
                continue
            # gen candidate
            impressions = [im.split('-') for im in impressions.strip().split(' ')]
            if len(histories) < self.click_sample:
                histories = ['0'] * (self.click_sample - len(histories)) + histories
            histories = histories[:self.click_sample]
            
            impressions = [(news_id, int(fl)) for news_id, fl in impressions]
            while len(impressions) % 50 != 0:
                impressions.append(['0', 0])
            
            impressions = [impressions[i:i+50] for i in range(0, len(impressions), 50)]
            
            for impression in impressions:
                for imp in impression:
                    if imp[1] != 0:
                        click_ev.append((histories, impression))
                        break
        return click_ev

    def __getitem__(self, index: int):
        clicked, imppresion = self.click_ev[index]

        clicked = self.get_title_index(clicked)
        sample_idx = [imp[1] for imp in imppresion]
        sample = self.get_title_index([imp[0] for imp in imppresion])
        # print(sample)
        return torch.tensor(clicked), torch.tensor(sample), torch.tensor(sample_idx, dtype=torch.float)
