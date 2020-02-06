# coding=utf-8
from tensorflow import keras
# from nlputils.data_sequence import DataSequence
import numpy as np
import config


class TitleSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, maxlen, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.maxlen = maxlen
        self.shuffle = shuffle
        self._length = int(np.ceil(len(self.x) / float(self.batch_size)))

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: min(self.__len__(), (idx + 1)) * self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]

        batch_x = keras.preprocessing.sequence.pad_sequences(batch_x, maxlen=self.maxlen)
        batch_y = keras.utils.to_categorical(batch_y, config.NUM_CLASSES)

        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)