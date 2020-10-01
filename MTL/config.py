################################################
# Config
################################################
BATCH_MAXIMUM_LEN = 1000 # How many utterances are we considering at most? (max_len <= BATCH_MAXIMUM_LEN)
NUM_GENRES = 24 #
VOCAB = 5000 # Size of TF / TF-IDF vocabulary
FEATS_SIZES = {
    'word2vec': 300,
    'w2v': 300,
    'doc2vec': 300,
    'd2v':300,
    'p2v':300,
    's2v':300,
    'sent2vec': 300,
    'paragraph2vec':300,
    'afinn': 1,
    'AFINN': 1,
    'vader': 1,
    'ngrams': VOCAB * 2,
    'tfidf': VOCAB,
    'hatebase': 1018,
    'abusive': 1,
    'empath_192': 192,
    'empath_2': 2,
    # 'linguistic': 6 # Not implemented yet
    'sentiment': 100,
    'toronto': 700,
    'toronto_sent2vec': 700,
    'script_w2v': 300,
    'script_word2vec': 300,
    # 'bert': 1024,
    'bert': 1536, #scriptbert (2 layers),
    'sst': 1024,
    'bert_base': 768,
    'bert_large': 1024,
    'moviebert': 1536
}

# Defined in __main__ argparse
# epochs = 30
# batch_size = 16
# FEAT = ['ngrams', 'afinn', 'w2v', 'vader', 'hatebase', 'empath_192', 'abusive', 'empath_2']
# VOCAB_SIZE = sum(map(FEATS_SIZES.__getitem__, FEAT))
