hparams = {
    'epochs': 4,
    'batch_size': 64,
    'lr': 7e-4,
    'name': 'autoencoder',
    'pretrained': 'data/w2v/news_200d.bin',
    'model': {
        'user_encoder': 'gru',
        'doc_nhead': 10,
        'embed_size': 200,
        'doc_encoder_size': 200,
        'doc_v_size': 150,
        'click_nhead': 10,
        'click_v_size': 150,
        'dropout': 0.2,
        'user_nhead': 20
    },
    'data': {
        'news_tsv': './data/small_train/news.tsv',
        'user_tsv': './data/small_train/behaviors.tsv',
        'dev_news_tsv': './data/small_dev/news.tsv',
        'dev_user_tsv': './data/small_dev/behaviors.tsv',
        'neg_sample': 4,
        'click_sample': 50,
        'maxlen':15
    }
}