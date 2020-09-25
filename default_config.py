hparams = {
    'epochs': 1,
    'batch_size': 64,
    'lr': 0.001,
    'name': 'exp4_alpha',
    'description': 'alpha: 1.',
    'pretrained': 'data/w2v/news_200d.bin',
    'alpha': 1.,
    'model': {
        'user_encoder': 'gru',
        'doc_nhead': 10,
        'embed_size': 200,
        'doc_encoder_size': 200,
        'doc_asp_cnt': 50,
        'user_asp_cnt': 50,
        'doc_v_size': 200,
        'click_nhead': 10,
        'click_v_size': 200,
        'dropout': 0.2,
        'user_nhead': 20,
        'gru_num_layers': 1,
    },
    'data': {
        'news_tsv': './data/train/news.tsv',
        'user_tsv': './data/train/behaviors.tsv',
        'dev_news_tsv': './data/dev/news.tsv',
        'dev_user_tsv': './data/dev/behaviors.tsv',
        'neg_sample': 4,
        'click_sample': 50,
        'maxlen':15
    }
}
