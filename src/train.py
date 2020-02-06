# coding=utf-8
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import time
import os
import numpy as np
from models import *
import argparse
from title_sequence import TitleSequence
import config

# Specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(**kwargs):
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.mkdir(config.CHECKPOINTS_DIR)

    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    session = tf.Session(config=c)
    tf.keras.backend.set_session(session)

    if kwargs['mode'] == 'char':
        dataset = np.load(config.TRAIN_CHAR_PATH, allow_pickle=True)
        maxlen = config.MAX_CHAR_TEXT_LENGTH
    else:
        dataset = np.load(config.TRAIN_WORD_PATH, allow_pickle=True)
        maxlen = config.MAX_WORD_TEXT_LENGTH

    titles = dataset['title_ids']
    labels = dataset['label_ids']

    kf = StratifiedKFold(5, shuffle=True)

    acc = 0
    for i, (tr_ind, te_ind) in enumerate(kf.split(np.zeros([len(labels)]), labels)):
        model_save_path = config.CHECKPOINTS_DIR + '/' + 'model_{}_{}_{}.h5'.format(kwargs['model'], kwargs['mode'], i)
        print('FOLD: {}'.format(str(i)))
        print('Training: {}, validation: {}'.format(len(tr_ind), len(te_ind)))

        train_titles = titles[tr_ind]
        test_titles = titles[te_ind]

        train_labels = labels[tr_ind]
        test_labels = labels[te_ind]

        train_data = TitleSequence(train_titles, train_labels, kwargs['batch_size'], maxlen)
        test_data = TitleSequence(test_titles, test_labels, kwargs['batch_size'], maxlen)

        # Set all the callbacks.
        Fname = 'Titles_'
        Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        tensorboard = keras.callbacks.TensorBoard(log_dir=config.LOG_DIR + '/' + Time,
                                                  histogram_freq=0, write_graph=False, write_images=False,
                                                  embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        ear = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='min', baseline=None,
                            restore_best_weights=True)
        checkpoint = keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)

        model = model_func[kwargs['model']](kwargs['dropout'], kwargs['lr'], kwargs['mode'])
        print(model.summary())

        history = model.fit_generator(generator=train_data,
                                      epochs=300,
                                      validation_data=test_data,
                                      callbacks=[ear, checkpoint, tensorboard, reduce_lr],
                                      shuffle=True)
        tf.keras.backend.clear_session()
        acc += history.history["val_categorical_accuracy"][-1]

    print(acc/5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dropout', '--dropout', default=0.5, type=float,
                        help='The dropout rate for the last dense layers.'
                             'Default 0.5.')
    parser.add_argument('-lr', '--lr', default=1e-3, type=float, help='Learning rate. Default 1e-3.')
    parser.add_argument('-batch_size', '--batch_size', default=128, type=int, help='Batch size. Default 128.')
    parser.add_argument('-model', '--model', default='textcnn', type=str, choices=['textcnn', 'bilstm', 'bigru'],
                        help='The classification model. Default textcnn.')
    parser.add_argument('-mode', '--mode', default='char', type=str, help='Type of the embedding input.')

    model_func = {'textcnn': setTextCNN, 'bilstm': setBiLSTM, 'bigru': setBiGRU}
    kwargs = vars(parser.parse_args())

    print(kwargs)
    train(**kwargs)






