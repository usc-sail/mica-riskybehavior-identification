from numpy.random import seed
seed(5393)

import tensorflow as tf
if tf.__version__ == "2.0.0":
    set_seed = tf.random.set_seed
else:
    from tensorflow import set_random_seed as set_seed

set_seed(12011)

from random import seed
seed(12345)

# From https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94
## required for efficient GPU use
# from keras.backend import tensorflow_backend
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# session = tf.Session(config=config)
# tensorflow_backend.set_session(session)

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, losses
from tensorflow.keras.utils import to_categorical
from Attention import BahdanauAttention as Attention
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from tqdm import tqdm
from time import time
from generator import Generator
from glob import iglob as glob

from features import Features
from config import *

tf.keras.backend.set_floatx('float32')

############################################################
# Violence model
############################################################
def create_model(max_len = 250, WORD_SIZE = 700, SENT_SIZE = 100, ATTN_SIZE = 500):

    ############################################################
    # Word model
    ############################################################
    word_inp = Input(shape = (max_len, WORD_SIZE))
    x = Dropout(0.5)(word_inp)
    gru, forward_h, backward_h = Bidirectional(GRU(16, return_sequences = True, return_state = True, reset_after = False))(x)
    state_h = Concatenate()([forward_h, backward_h])

    ############################################################
    # Sentiment model
    ############################################################
    sent_inp = Input(shape = (max_len, SENT_SIZE))
    s = Dropout(0.5)(sent_inp)
    s_gru, s_forward_h, s_backward_h = Bidirectional(GRU(16, return_sequences = True, return_state = True, reset_after = False))(s)
    s_state_h = Concatenate()([s_forward_h, s_backward_h])

    ############################################################
    # Genre
    ############################################################
    genre_inp = Input(shape = (NUM_GENRES,))

    ############################################################
    # Classifier (violence)
    ############################################################
    vio_sem, w_sem = Attention(ATTN_SIZE)(state_h, gru)
    vio_sent, w_sent = Attention(ATTN_SIZE)(s_state_h, s_gru)
    vio = Concatenate()([vio_sem, vio_sent, genre_inp])

    vio = Dense(32, activation = "relu")(vio)
    vio = Dense(3, activation="softmax", name = 'violence')(vio)

    ############################################################
    # Classifier (sex)
    ############################################################
    sex_sem, w_sem = Attention(ATTN_SIZE)(state_h, gru)
    sex_sent, w_sent = Attention(ATTN_SIZE)(s_state_h, s_gru)
    sex = Concatenate()([sex_sem, sex_sent, genre_inp])
    #
    # #
    sex = Dense(32, activation = "relu")(sex)
    sex = Dense(3, activation = "softmax", name = "sex")(sex)
    # #
    # ############################################################
    # # Regresor (drugs)
    # ############################################################
    drugs_sem, w_sem = Attention(ATTN_SIZE)(state_h, gru)
    drugs_sent, w_sent = Attention(ATTN_SIZE)(s_state_h, s_gru)
    drugs = Concatenate()([drugs_sem, drugs_sent, genre_inp])
    #
    # #
    drugs = Dense(32, activation = "relu")(drugs)
    drugs = Dense(3, activation = "softmax", name = "drugs")(drugs)

    ############################################################
    # Model definition
    ############################################################
    model = Model(inputs=[word_inp, sent_inp, genre_inp], outputs=[vio, sex, drugs])
    model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  optimizer='adam',
                  metrics=['accuracy'])


    # *-Only model:
    # model = Model(inputs=[word_inp, sent_inp, genre_inp], outputs=[vio])
    # model.compile(loss = ['categorical_crossentropy'],
                  # optimizer='adam',
                  # metrics=['accuracy'])

    return model



############################################################
# TRAIN / EVAL
############################################################

# Loads data in test_dir
def get_testdata(test_dir, features):
    files = list(glob(os.path.join(test_dir, "*_labels.npz")))
    NEW_VER = False

    if len(files) == 0:
        files = list(glob(os.path.join(test_dir, "*.npz")))
        NEW_VER = True

    words, sentiment, genres = [], [], []
    vio, sex, drugs = [], [], []

    for label_f in tqdm(files):
        (word_features, sentiment_features, batch_genre), (batch_vio, batch_sex, batch_drugs) = features.get_feats(label_f, batch_dir = test_dir)

        words.append(word_features)
        sentiment.append(sentiment_features)
        genres.append(batch_genre)

        vio.append(batch_vio)
        sex.append(batch_sex)
        drugs.append(batch_drugs)

    # Concatenate lists of lists into tensor
    vio = np.concatenate(vio, axis = 0)
    sex = np.concatenate(sex, axis = 0)
    drugs = np.concatenate(drugs, axis = 0)

    y_test = [vio, sex, drugs]

    words = np.concatenate(words, axis = 0)
    sentiment = np.concatenate(sentiment, axis = 0)
    genres = np.concatenate(genres, axis = 0).astype(float)

    return y_test, words, sentiment, genres

def train_and_evaluate_model(model, data_train_gen, data_eval_folder, data_test_folder, features, data_eval_gen = None, epochs = 30, model_name = "best_model.hdf5"):

    # Load eval data
    y_eval, word_eval, sentiment_eval, genre_eval = get_testdata(data_eval_folder, features)

    model.fit_generator(epochs = epochs,
                        generator = data_train_gen,
                        steps_per_epoch = len(data_train_gen),
                        validation_data = ([word_eval, sentiment_eval, genre_eval], y_eval),
                        max_queue_size = 30,
                        workers = 1,
                        use_multiprocessing = True,
                        callbacks = [
                                        EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto"),
                                        ModelCheckpoint(model_name, monitor='val_loss', verbose = 1, save_weights_only = True, save_best_only = True)
                                    ])

    # Evaluate
    model = None #just to be absolutely sure
    model = create_model(max_len = features.max_len, WORD_SIZE=features.WORD_SIZE, SENT_SIZE=features.SENT_SIZE)
    model.load_weights(model_name)

    # Model predictions
    y_test, words, sentiments, genres = get_testdata(data_test_folder, features)
    y_pred = model.predict([words, sentiments, genres])

    return [y_test, y_pred]

############################################################
#
############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN with document probas')
    parser.add_argument("folddir", type=str, help = "CV fold directory")
    parser.add_argument("outf", type=str, help = "output npz file")
    parser.add_argument("--modelname", type=str, help = "best model hdf5 name", default="models/best_model.hdf5")
    parser.add_argument("--max_len", type=int, help="num of utterances to use (before end)", default=500)
    parser.add_argument("--epochs", type = int, help = "num epochs", default = 30)
    parser.add_argument("--batch_size", type = int, help = "batch size", default = 16)
    parser.add_argument("--bert_selector", type = str, help = "", default=None)
    parser.add_argument("--wordrepr", type = str, default="sent2vec")
    parser.add_argument("--sentrepr", type = str, default="sentiment")

    args = parser.parse_args()

    features = Features(wordrepr = args.wordrepr,
                        sentrepr = args.sentrepr,
                        bert_selector = args.bert_selector,
                        max_len = args.max_len,
                        categorical = True)
    model = create_model(max_len = features.max_len, WORD_SIZE=features.WORD_SIZE, SENT_SIZE=features.SENT_SIZE)
    preds = train_and_evaluate_model(model,
                                     data_train_gen = Generator(f"{args.folddir}/train", features.get_feats_sex_only),
                                     data_eval_folder = f"{args.folddir}/eval",
                                     data_test_folder = f"{args.folddir}/test",
                                     features = features,
                                     model_name = args.modelname,
                                     epochs = args.epochs)

    print(",".join([args.wordrepr, args.sentrepr, str(args.max_len), str(args.bert_selector) if args.bert_selector else ""]))
    print("Done")
    with open(args.outf, "wb") as outpt:
        pickle.dump(preds, outpt)
