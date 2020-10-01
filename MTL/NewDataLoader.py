import os
import numpy as np
NEW_VER = True
################################################
# Sept 20th, 2019
# This fixes the new load API which requries read_pickle
################################################
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

################################################
################################################
# Load each file accordingly
################################################
def load_labels(label_f, categorize = None, categorize_F = None):
    """
    """
    with np.load(label_f) as f:
        if 'labels' in f:
            batch_labels = f['labels']
            additional_labels = None

        elif 'violence_rating' in f:
            batch_labels = f['violence_rating']
            additional_labels = np.vstack((f['sex_rating'], f['drugs_rating'])).T # ? x 2
        else:
            raise Exception("Neither labels nor violence_rating were found")

    return batch_labels, additional_labels

def load_genre(i, batch_dir):
    batch_genre = []
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_genre = tmp['genre']
    return batch_genre

def load_text(i, batch_dir):
    batch_text = []
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_text = tmp['text']
    return batch_text

def load_lexicon_features(FEAT, i, batch_dir, FEATS_SIZES, BATCH_MAXIMUM_LEN = 1000):
    batch_features = []
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:

        if 'afinn' in FEAT or 'AFINN' in FEAT:
            batch_afinn = tmp['afinn']
            batch_afinn = np.concatenate(batch_afinn, axis = 0)\
                            .reshape(-1, BATCH_MAXIMUM_LEN, FEATS_SIZES['afinn']) #? x 1000 x 1
            batch_features.append(batch_afinn)

        if 'vader' in FEAT:
            batch_vader = tmp['vader']
            batch_vader = np.concatenate(batch_vader, axis = 0)\
                            .reshape(-1, BATCH_MAXIMUM_LEN, FEATS_SIZES['vader'])
            batch_features.append(batch_vader)

        if 'abusive' in FEAT:
            batch_abusive = tmp['abusive_scores']
            batch_abusive = np.concatenate(batch_abusive, axis = 0)\
                              .reshape(-1, BATCH_MAXIMUM_LEN, FEATS_SIZES['abusive'])
            batch_features.append(batch_abusive)

        if 'hatebase' in FEAT:
            batch_hate = tmp['hatebase']
            batch_hate = np.concatenate(batch_hate, axis = 0)
            batch_hate = np.array(batch_hate)\
                           .reshape(-1, BATCH_MAXIMUM_LEN, FEATS_SIZES['hatebase'])
            batch_features.append(batch_hate)

    return batch_features

def load_empath_192(i, batch_dir, BATCH_MAXIMUM_LEN = 1000):
    batch_features = []
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_empath = tmp['empath']
        batch_empath = np.concatenate(batch_empath, axis = 0)\
                         .reshape(-1, BATCH_MAXIMUM_LEN, 194)
        mask = np.ones(batch_empath.shape[2], dtype = bool)
        mask[[173, 192]] = False
        batch_empath = batch_empath[:, :, mask]
        batch_features.append(batch_empath)
    return batch_features

def load_empath_2(i, batch_dir, BATCH_MAXIMUM_LEN = 1000):
    batch_features = []
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_empath = tmp['empath']
        batch_empath = np.concatenate(batch_empath, axis = 0)\
                         .reshape(-1, BATCH_MAXIMUM_LEN, 194)
        batch_empath = batch_empath[:, :, [173, 192]]
        batch_features.append(batch_empath)
    return batch_features

def load_tf_or_idf(i, batch_dir, mode = "ngrams"):
    if mode not in ['ngrams', 'tfidf']:
        raise Exception("Mode has to be either ngrams or tfidf")
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_ngrams = tmp[mode]
    return batch_ngrams

def load_w2v_or_p2v(i, batch_dir, FEATS_SIZES, mode = "word2vec", BATCH_MAXIMUM_LEN = 1000):
    with np.load(os.path.join(batch_dir, "{}.npz".format(i))) as tmp:
        batch_w2v = np.concatenate(tmp[mode], axis = 0)\
                      .reshape(-1, BATCH_MAXIMUM_LEN, FEATS_SIZES[mode])
    return batch_w2v

def load_BERT(i, batch_dir, FEATS_SIZES, mode = "bert", BATCH_MAXIMUM_LEN = 1000, bert_selector = None):
    word_features = load_w2v_or_p2v(i, batch_dir, FEATS_SIZES, mode=mode, BATCH_MAXIMUM_LEN=BATCH_MAXIMUM_LEN)
    size = word_features.shape[2]

    if bert_selector == 'first':
        size = int(size / 2)
        word_features = word_features[:, :, :size]
    elif bert_selector == 'last':
        size = int(size / 2)
        word_features = word_features[:, :, -size:]

    return word_features
