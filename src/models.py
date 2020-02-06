# coding=utf-8
from tensorflow import keras
import numpy as np
import config


def setTextCNN(dropout_rate, lr, mode='char'):
    if mode == 'char':
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_CHAR_TEXT_LENGTH
    else:
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_WORD_TEXT_LENGTH

    main_input = keras.layers.Input([text_length])

    filter_sizes = [2, 3, 4, 5, 6]
    conv_outputs = []
    num_hidden = 0

    if mode == 'char' or mode == 'word':
        x = keras.layers.Embedding(input_dim=vocab_size, embeddings_initializer=keras.initializers.Constant(W),
                                   output_dim=vec_dim, input_length=text_length)(main_input)
        # Output shape [batch_size, text_length, embedding_dim, 1]

        for filter_size in filter_sizes:
            conv = keras.layers.Conv1D(150, filter_size, activation='relu')(x)
            conv = keras.layers.GlobalMaxPooling1D()(conv)
            conv_outputs.append(conv)
        num_hidden = 256

    elif mode == 'multi':
        # Static channel
        x1 = keras.layers.Embedding(input_dim=vocab_size, embeddings_initializer=keras.initializers.Constant(W),
                       output_dim=vec_dim, input_length=text_length, trainable=False)(main_input)
        # Non-Static channel
        x2 = keras.layers.Embedding(input_dim=vocab_size, output_dim=vec_dim, input_length=text_length)(main_input)
        x = keras.layers.Concatenate()(x1, x2)
        # TODO: test
        for filter_size in filter_sizes:
            conv = keras.layers.Conv2D(150, (filter_size, vec_dim), activation='relu')(x1)
            conv = keras.layers.GlobalMaxPooling2D()(conv)
            conv_outputs.append(conv)
        num_hidden = 512

    else:
        raise ValueError('Mode `{}` not supported.'.format(mode))

    x = keras.layers.Concatenate()(conv_outputs)

    x = keras.layers.Dense(num_hidden,
              kernel_regularizer=keras.regularizers.l2(0.01),
              bias_regularizer=keras.regularizers.l2(0.01),
              activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    predictions = keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=main_input, outputs=predictions)
    optimizer = keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def setBiLSTM(dropout_rate, lr, mode='char'):
    assert mode in ['char', 'word']

    if mode == 'char':
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_CHAR_TEXT_LENGTH
    else:
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_WORD_TEXT_LENGTH

    main_input = keras.layers.Input([text_length])

    x = keras.layers.Embedding(input_dim=vocab_size, embeddings_initializer=keras.initializers.Constant(W),
                               output_dim=vec_dim, input_length=text_length, mask_zero=True)(main_input)
    # Output shape [batch_size, text_length, embedding_dim, 1]

    # Output shape [batch_size, text_length, embedding_dim, 1]
    x = keras.layers.Bidirectional(keras.layers.LSTM(64), merge_mode='concat')(x)
    x = keras.layers.Dense(256,
              kernel_regularizer=keras.regularizers.l2(0.01),
              bias_regularizer=keras.regularizers.l2(0.01),
              activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    predictions = keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=main_input, outputs=predictions)
    optimizer = keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def setBiGRU(dropout_rate, lr, mode='char'):
    assert mode in ['char', 'word']

    if mode == 'char':
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_CHAR_TEXT_LENGTH
    else:
        W = np.load(config.FIT_CHAR_PATH)['embeddings']
        vocab_size, vec_dim = W.shape
        text_length = config.MAX_WORD_TEXT_LENGTH

    main_input = keras.layers.Input([text_length])

    x = keras.layers.Embedding(input_dim=vocab_size, embeddings_initializer=keras.initializers.Constant(W),
                               output_dim=vec_dim, input_length=text_length, mask_zero=True)(main_input)
    # Output shape [batch_size, text_length, embedding_dim, 1]

    # Output shape [batch_size, text_length, embedding_dim, 1]
    x = keras.layers.Bidirectional(keras.layers.GRU(64), merge_mode='concat')(x)
    x = keras.layers.Dense(256,
              kernel_regularizer=keras.regularizers.l2(0.01),
              bias_regularizer=keras.regularizers.l2(0.01),
              activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    predictions = keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=main_input, outputs=predictions)
    optimizer = keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


# def setAttentionBiGRU(dropout_rate, lr, mode='char'):
#     if mode == 'char':
#         text_length = config.MAX_CHAR_TEXT_LENGTH
#         vec_dim = config.VEC_DIM
#         vocab_size = config.CHAR_VOCAB_SIZE + 2
#         main_input = Input([text_length])
#         x = Embedding(input_dim=vocab_size, mask_zero=True,
#                       output_dim=vec_dim, input_length=text_length)(main_input)
#     elif mode == 'word':
#         text_length = config.MAX_WORD_TEXT_LENGTH
#         vec_dim = config.PRETRAINED_VEC_DIM
#         vocab_size = config.WORD_VOCAB_SIZE + 2
#         with open(config.FIT_WORD_PATH, 'rb') as f:
#             W = pickle.load(f)
#         main_input = Input([text_length])
#         x = Embedding(input_dim=vocab_size, embeddings_initializer=Constant(W), mask_zero=True,
#                       output_dim=vec_dim, input_length=text_length)(main_input)
#     else:
#         raise ValueError('Mode `{}` not supported.'.format(mode))

#     # Output shape [batch_size, text_length, embedding_dim, 1]
#     rnn_outputs, hidden_forward, hidden_backward = \
#         Bidirectional(GRU(64, return_state=True, return_sequences=True),merge_mode='concat')(x)

#     hidden_state = Concatenate()([hidden_forward, hidden_backward])

#     x = Attention(16, 'additive')([hidden_state, rnn_outputs])
#     x = Dense(128,
#               kernel_regularizer=l2(0.01),
#               bias_regularizer=l2(0.01),
#               activation='relu')(x)
#     x = Dropout(dropout_rate)(x)

#     predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)

#     model = keras.Model(inputs=main_input, outputs=predictions)
#     optimizer = keras.optimizers.Adam(lr)
#     model.compile(optimizer=optimizer,
#                   loss='categorical_crossentropy',
#                   metrics=['categorical_accuracy'])
#     return model





