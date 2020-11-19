# Joint Estimation and Analysis of Risk Behavior Ratings in Movie Scripts

![MultiTask Model for risk behavior prediction](https://sail.usc.edu/~victorrm//images/MultitaskDiagram.png)

## Introduction
It’s not only about violence! Risk behaviors frequently co-occur with one another, both in real-life (Brener & Collins, 1998) and in entertainment media (Bleakley et al., 2014, 2017; Thompson & Yokota, 2004).
To capture the this co-occurrence, we develop a multi-task learning (MTL) approach to predict a movie script’s violent, sexual and substance-abusive content.
By using a multi-task approach, our model could improve violent content classification, as well as provide insights on its relation to other dimensions of risk behaviors depicted in film media.


### Pre-trained MovieBERT
Semantic representations of character utterances were obtained using movieBERT, our own fine-tuned BERT-base model.
This model consists of 12 transformer layers that learn a 768-dimensional representation for a utterance. We train this model over a dataset of 6,000 movie scripts with a 85%-15% train-test split.
Following the original implementation of BERT (Devlin et al., 2019), we optimize the model for two tasks: next-sentence prediction and masked language models.
In the former, the model has to predict the sentence that follows a given sentence, and in the latter, a random word in a sentence is masked with a token, and the model has to recover the original word.
We initialize the weights of our model with those from the pre-trained BERT-base model, and continue training for 10,000 steps.
Model parameters: learning rate of 2×10^−5, batch size of 32, and sequences length of 128.
MovieBERT achieves 96.5% accuracy on the next sentence prediction task, and a 65.9% accuracy on the masked language model--an absolute improvement from the BERT-base model of 24.5% and 12.43%, respectively.

#### Obtaining MovieBERT
We are still in the process of making movieBERT readily available. In the meantime, feel free to contact us via email (victorrm at usc), Github Issue (open an issue here), or any other type of electronic signal.

#### Using MovieBERT
The following shows how to load movieBERT into memory for feature extraction or representation learning, using [huggingface transformers](https://huggingface.co/transformers/) library:

```
import torch
from transformers import BertModel, BertTokenizer

movieBERT = BertModel.from_pretrained(<movieBERT location>)
movieBERT.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "Luke, I am your father!"
tokens = tokenizer.encode(sentence, add_special_tokens=True)
tokens = torch.tensor([tokens])

outputs = movieBERT(tokens)
repr = outputs[0] #(1, #tokens, 768)
```

### Sentiment Model
We obtain sentiment representations from an implementation of (Tai et al. 2015): 50-dimensional hidden representation, dropout (p=0.1), trained with Adam optimizer on a batch size of 25 and a L2penalty of 10^−4.
Pre-trained weights can be found in `features/sentiment/sst.h5`. The following code loads the sentiment model into memory for feature extraction. `GLOVE_FILE` corresponds to [GLOVE]() embeddings (typically `glove.6B.300d.w2vformat.txt`).

```
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import *

inp = Input(shape=(56,))

embeddings_weights = KeyedVectors.load_word2vec_format(<GLOVE FILE>, binary=False)
Embeddings =  embeddings_weights.get_keras_embedding(train_embeddings=True)

x = Embeddings(inp)

x = Dropout(dropout)(x)
x = Bidirectional(LSTM(lstm_dim))(x)
x = Dropout(dropout)(x)

model = Model(inputs = inp, outputs = x)
model.compile('adam', 'categorical_crossentropy',
          metrics=['accuracy'])


model.load_weigths("./features/sentiment/sst.h5", by_name = True)
model.summary()
```

## Model implementation
Our MTL model is implemented using Keras, trained with Adam optimizer, batch size of 16, and high dropout probability (0.5).
For the RNN layer, we used bi-directional GRUs.
For the sentiment models, we trained a Bi-LSTM parameters were: 50-dimensional hidden representation, dropout (0.1), trained with Adam optimizer on a batch size of 25 and a L2-penalty of 10^-4.
Model implementation can be found in `MTL/model.py`.

## Data
Due to license restrictions, we cannot share Common Sense Media (CSM) ratings directly.
Readers are invited to reach to CSM directly to ask how to obtain the data.

Instead, we provided manually alignment of a subset of 989 movie scripts from our dataset to the content ratings found in (Martinez et al., 2019).
This alignment can be found in `data/MASTER movie scripts database - Sheet1.tsv`.
This file contains movie titles, IMDb identifiers, and CSM identifiers (corresponding to either their full URL or their unique ID used by the CSM API).

## Method
To run the experiments, we first assume that there is a file containing all the examples, their feature sequence, and their risk-behavior scores in `data/995movies/dataframe_all_labels_sent2vec__BERTS.pkl`. This pickle file is a pandas dataframe with at least the following columns: `moviebert_mean_pool`, `sentiment_embeddings`, `genre`, `violence_rating`, `sex_rating`, and `drugs_rating`.

In order to run the 10-fold CV experiments in a reduced amount of memory, we split the dataset into 10 folds, each one with its own `train/test/dev` split.
Inside each of the train/test/dev folders, we save a numpy array with all the features of a batch. To help the reader replicate this structure we have provided a auxiliary script in `utils/createBatches.py`.

In its current form, `MTL/model.py` takes a CV fold and runs the experiments for that split. To run all folds, an auxiliary bash script is provided in `MTL/runCV.sh`. This script takes as parameters the name of the python file to run, the CUDA device where to run the experiments, a directory where the batches are stored, an experiment name, and additional parameters sent to `model.py`. As an example, the following command is used to train the best model in our paper:

```
./runCV.sh model.py 3 /home/victor/batches/10-fold/533reversefolds_allclasses_multigenres 10CV_late_500_multitask_movieBERT --wordrepr moviebert --sentrepr sentiment --max_len 500
```


## Citation
If you use this code for research, please cite [our paper](http://sail.usc.edu/publications/files/VioSexDrugs_EMNLP20.pdf) as follows:
```
@inproceedings{Martinez2020JointEstimationandAnalysis,
 author = {Martinez, Victor and Somandepalli, Krishna and Tehranian-Uhls, Yalda and Narayanan, Shriknath},
 booktitle = {In proceedings of EMNLP},
 link = {http://sail.usc.edu/publications/files/VioSexDrugs_EMNLP20.pdf},
 location = {Dominican Republic},
 month = {November},
 title = {Joint Estimation and Analysis of Risk Behavior Ratings in Movie Scripts},
 year = {2020}
}
```
