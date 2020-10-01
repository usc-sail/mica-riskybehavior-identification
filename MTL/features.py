import os
import numpy as np

import torch
from transformers import BertTokenizer

from tensorflow.keras.utils import to_categorical
from NewDataLoader import *
from config import *

import warnings

class Features:
    def __init__(self, **kwargs):
        self.max_len = kwargs.get('max_len', 250)
        self.categorical = kwargs.get('categorical', True)
        self.wordrepr = kwargs.get('wordrepr', 'toronto_sent2vec')
        self.sentrepr = kwargs.get('sentrepr', 'sentiment')
        self.bert_selector = kwargs.get('bert_selector', 'None')

        # Transform into H/M/L
        self.categorize_F = np.vectorize(self.categorize)

        # Feature size
        self.WORD_SIZE = FEATS_SIZES[self.wordrepr]
        if self.bert_selector == "first" or self.bert_selector == "last":
            self.WORD_SIZE = int(self.WORD_SIZE / 2)

        self.SENT_SIZE = FEATS_SIZES[self.sentrepr]
        if self.sentrepr == "bert":
            if self.bert_selector == "first" or self.bert_selector == "last":
                self.SENT_SIZE = int(self.SENT_SIZE / 2)

        print("Features:", self.wordrepr, self.sentrepr, self.max_len, self.bert_selector)

    ################################################
    # Transform ordinal ratings into categorical
    ################################################
    def categorize(self, rating):
        if rating >= 4:
            return 0 #HIGH
        elif rating > 2:
            return 1 #MED
        else:
            return 2 #LOW

    ################################################
    # Loads features and trims them to max_len
    ################################################
    def get_feats(self, label_f, batch_dir = None):

        if not batch_dir:
            batch_dir = os.path.dirname(label_f)

        # Labels
        batch_labels, additional_labels = load_labels(label_f)
        batch_labels = np.c_[batch_labels, additional_labels]

        if self.categorical:
            batch_labels = self.categorize_F(batch_labels) #H/M/L
            batch_labels = to_categorical(batch_labels, num_classes = 3) #One-hot encoding

        vio, sex, drugs = batch_labels[:, 0, :], batch_labels[:, 1, :], batch_labels[:, 2, :]
        y = [vio, sex, drugs]

        # Get the index from the filename
        i = os.path.basename(label_f).split("_")[0]
        i = i.replace('.npz', '')

        # Genre
        batch_genre = load_genre(i, batch_dir)

        # Words
        if self.wordrepr in ['sent2vec', 'word2vec', 'script_word2vec', 'toronto_sent2vec']:
            word_features = load_w2v_or_p2v(i, batch_dir, FEATS_SIZES, self.wordrepr)
        elif self.wordrepr in ['bert_large', 'bert_base', 'sst', 'moviebert']:
            word_features = load_BERT(i, batch_dir, FEATS_SIZES, mode  = self.wordrepr, bert_selector = self.bert_selector)
        elif self.wordrepr in ['ngrams', 'tfidf']:
            word_features = load_tf_or_idf(i, batch_dir, self.wordrepr)

        # Sentiment
        if self.sentrepr in ['sentiment']:
            sentiment_features = load_w2v_or_p2v(i, batch_dir, FEATS_SIZES, "sentiment")
        elif self.sentrepr in ['bert_large', 'bert_base', 'sst', 'moviebert']:
            sentiment_features = load_BERT(i, batch_dir, FEATS_SIZES, mode = self.sentrepr, bert_selector = self.bert_selector)
        # elif sentrepr in ['sent_post', 'posteriors']:
            # sentiment_features = ???

        word_features = word_features[:, -self.max_len:, :] #Trim
        sentiment_features = sentiment_features[:, -self.max_len:, :]

        return ([word_features, sentiment_features, batch_genre], y)

    def get_feats_any_only(self, label_f, index = 0, batch_dir = None):
        ([word_features, sentiment_features, batch_genre], y) = self.get_feats(label_f, batch_dir = batch_dir)
        return ([word_features, sentiment_features, batch_genre], y[index])

    def get_feats_vio_only(self, label_f, batch_dir = None):
        return self.get_feats_any_only(label_f, index = 0, batch_dir = batch_dir)

    def get_feats_sex_only(self, label_f, batch_dir = None):
        return self.get_feats_any_only(label_f, index = 1, batch_dir = batch_dir)

    def get_feats_drugs_only(self, label_f, batch_dir = None):
        return self.get_feats_any_only(label_f, index = 2, batch_dir = batch_dir)

    def get_concat_feats(self, label_f, batch_dir = None):
        (word_features, sentiment_features, batch_genre), batch_labels = self.get_feats(label_f, batch_dir)
        feats = np.concatenate([word_features, sentiment_features], axis = 2)
        return [feats, batch_genre], batch_labels[0]


class BertFeatures(Features):
    """This class goes from text to padded transformer features"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs.get('bert_name', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(self.name)

        self.max_len = kwargs.get('max_len', self.tokenizer.max_len)
        self.categorical = kwargs.get('categorical', True)

        if self.max_len > self.tokenizer.max_len:
            warnings.warn("max_len > tokenizer({}).max_len.".format(self.name))

        print("BertFeatures:", self.name, self.max_len)

    def get_feats(self, label_f, batch_dir = None):
        if not batch_dir:
            batch_dir = os.path.dirname(label_f)

        # Labels
        batch_labels, additional_labels = load_labels(label_f)
        batch_labels = np.c_[batch_labels, additional_labels]

        if self.categorical:
            batch_labels = self.categorize_F(batch_labels) #H/M/L
            batch_labels = to_categorical(batch_labels, num_classes = 3) #One-hot encoding

        vio, sex, drugs = batch_labels[:, 0], batch_labels[:, 1], batch_labels[:, 2]
        y = [vio, sex, drugs]

        # Get the index from the filename
        i = os.path.basename(label_f).split("_")[0]
        i = i.replace('.npz', '')

        # Genre
        batch_genre = load_genre(i, batch_dir)

        #
        features = []
        for row in load_text(i, batch_dir):

            # Tokenize and trim
            text = self.tokenizer.tokenize(row)[-self.max_len:]

            # Encode text
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens = True)])

            features.append(input_ids)

        # Convert to tensor
        features = torch.cat(features, dim = 0)

        return ([features, batch_genre], y)
