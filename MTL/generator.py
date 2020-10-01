import os
import threading
from random import Random
from glob import iglob as glob
from tensorflow.keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, batch_dir, feat_func, shuffle = True, shuffler = Random(42), globbed = False):
        self.batch_dir = batch_dir
        self.shuffler = shuffler
        self.shuffle = shuffle
        self.feat_func = feat_func
        self.files = list(glob(os.path.join(self.batch_dir, "*.npz")))
        self.shuffler.shuffle(self.files)
        self.length = len(self.files)
        self.globbed = globbed
        self.on_epoch_end()
        # print('generator initiated')

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffler.shuffle(self.files)

    def __getitem__(self, index):
        """Generates one batch of data"""
        # print(f'generator: {index}')
        label_f = self.files[index % self.length]
        if not self.globbed:
            return self.feat_func(label_f, batch_dir = self.batch_dir)
        else:
            return self.feat_func(label_f)

    def __len__(self):
        return self.length
