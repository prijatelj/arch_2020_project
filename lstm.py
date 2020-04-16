"""Keras predictor for"""
import argparse
import logging
import os

from keras.layers import LSTM, GRU, Activation, Input, Dense, CuDNNLSTM, \
    CuDNNGRU
from keras.models import Model
from keras.callbacks.tensorboard_v1 import TensorBoard
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf

import data_loader
import exp_io

def window_data(x, window_size, batch_size=1):
    """Delievers the data in a window format, appending zeros to beginning of
    window buffer and ending at last element of data.
    """
    assert window_size >= 1

    if len(x.shape) == 1:
        windowed_x = np.append(np.zeros(window_size), x)
    else:
        windowed_x = np.vstack((
            np.zeros((window_size, x.shape[1])),
            x,
        ))

    # format of windowed data?
    for i in range(window_size + 1, len(windowed_x) + 1):
        yield windowed_x[i - window_size: i]


class BranchRNN(object):
    """Recurrent Neural Network branch predictor, either with a LSTM by default
    of a GRU.

    Attributes
    ----------
    batch_size
    history_size : int
        The maximum size of the history queue. Default is 0, meaning no
        history. 1 means that the history is the prior sample and is used in
        training along with the current sample.
    feature_history : np.ndarray
        The history queue where first entry is first feature of an event to
        have occurred.
    label_history : np.ndarray
        The history queue where first entry is first label of an event to
        have occurred.
    """

    def __init__(
        self,
        input_shape,
        units=1,
        hidden_layers=1,
        output_shape=1,
        window_size=1,
        tfboard_log=None,
        update_freq=10000,
        cudnn=True,
        gru=False,
        batch_size=1,
        history_size=0,
        epochs=1,
    ):
        self.input_vector = Input(shape=[window_size, input_shape])
        self.window_size = window_size
        #if batch_size != 1 and history_size <= 0:
        #    raise NotImplementedError(' '.join([
        #        'Batch size greater than 1 is not implemented without a',
        #        'history of 1 or greater.'
        #    ]))
        self.batch_size = max(1, batch_size)
        self.epochs = max(1, epochs)
        self.history_size = max(0, history_size)
        self.feature_history = None
        self.label_history = None

        # Make the RNN model
        x = self.input_vector

        if gru:
            for i in range(hidden_layers):
                x = CuDNNGRU(units)(x) if cudnn else GRU(units)(x)
        else: # LSTM is default
            for i in range(hidden_layers):
                x = CuDNNLSTM(units)(x) if cudnn else LSTM(units)(x)

        self.output_vector = Dense(1, activation='sigmoid')(x)

        self.model = Model(
            inputs=self.input_vector,
            outputs=self.output_vector,
        )
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        if isinstance(tfboard_log, str):
            self.callbacks = [TensorBoard(
                tfboard_log,
                update_freq=update_freq,
            )]
        else:
            self.callbacks = None

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def update_history(self, features, labels):
        if self.feature_history is None or self.feature_history is None:
            # Initialize the history
            self.feature_history = features
            self.label_history = labels
        else:
            # adds the new data samples to end of queue and removes any old
            # entries that does not fit within the history_size
            self.feature_history = np.concatenate(
                (self.feature_history, features),
                axis=0,
            )[:self.history_size + 1]
            self.label_history = np.concatenate(
                (self.label_history, labels),
                axis=0,
            )[:self.history_size + 1]

    def online(self, x, y):
        y = y.reshape(-1, 1)

        logging.info('y reshape shape = %s', str(y.shape))

        preds = []
        if self.history_size > 0 or self.batch_size == 1:
            # Online where fits every single sample
            for i, win_x in enumerate(window_data(x, self.window_size)):
                win_x = win_x[np.newaxis, ...]
                preds.append(self.predict(win_x, batch_size=self.batch_size))

                if self.history_size > 0:
                    self.update_history(win_x, y[i])
                    self.fit(
                        x=self.feature_history,
                        y=self.label_history,
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                    )
                else:
                    self.fit(
                        x=win_x,
                        y=y[i],
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                    )
        else: # Accelerated batch prediction simulation, no history atm
            # Fits per batch sized intervals

            # TODO perhaps implement usage of history here
            # TODO perhaps break dependency on batch_size for a fit epoch's
            # number of samples, where epoch is # samples before fitting

            batch_features = []
            for i, win_x in enumerate(window_data(x, self.window_size)):
                win_x = win_x[np.newaxis, ...]
                # NOTE do following for more accurate simulation
                #preds.append(self.predict(win_x, batch_size=1))

                batch_features.append(win_x)

                if i % self.batch_size == 0 or i >= len(x) - 1:
                    batch_features = np.concatenate(batch_features, axis=0)

                    # NOTE do following for probably faster simulation
                    preds += self.predict(
                        batch_features,
                        batch_size=self.batch_size,
                    ).tolist()

                    # fit every batch size
                    self.fit(
                        x=batch_features,
                        y=y[max(0, i + 1 - len(batch_features)):i + 1],
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=self.epochs,
                        callbacks=self.callbacks,
                    )

                    batch_features = []

        return preds




def parse_args():
    parser = argparse.ArgumentParser(
        description='Test the LSTM branch predictor.',
    )

    parser.add_argument(
        'in_file',
        help='The data file to test on.',
    )

    parser.add_argument(
        '-o',
        '--output_dir',
        default=None,
        help='The CSV output file path.',
    )

    parser.add_argument(
        '-l',
        '--hidden_layers',
        default=1, type=int,
        help='The number of hidden layers in the ANN.',
    )

    parser.add_argument(
        '-u',
        '--units',
        default=1,
        type=int,
        help='The number of units per hidden layers in the ANN.',
    )

    parser.add_argument(
        '-w',
        '--window_size',
        default=1,
        type=int,
        help='The size of the window for making a sequence of the input data.',
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        default=1,
        type=int,
        help='The batch size of the recurrent neural network.',
    )

    parser.add_argument(
        '-e',
        '--epochs',
        default=1,
        type=int,
        help='The number of epochs per fitting session.',
    )

    parser.add_argument(
        '--history_size',
        default=0,
        type=int,
        help=' '.join([
            'The size of the queue of event feature and label pairs used in',
            'training per epoch. Defaults to 0, meaning no history queue.',
            'Each epoch is just the current sample if batch size 1.'
        ]),
    )

    parser.add_argument(
        '-k',
        '--lsb_bits',
        default=8,
        type=int,
        choices=range(1, 13),
        help='Number of LSB bits of PC to consider.',
    )

    parser.add_argument(
        '-t',
        '--tfboard_log',
        default=None,
        help='The TensorBoard log path.',
    )

    parser.add_argument(
        '--tfboard_freq',
        default=10000,
        help='The update freq of TensorBoard.',
    )

    parser.add_argument(
        '--cudnn',
        action='store_true',
        help='Uses CuDNN version of LSTM or GRU',
    )

    parser.add_argument(
        '--gru',
        action='store_true',
        help='Uses GRU instead of the LSTM',
    )

    exp_io.add_hardware_args(parser)
    exp_io.add_logging_args(parser, 'INFO')

    args = parser.parse_args()

    exp_io.set_hardware(args)
    exp_io.set_logging(args)

    return args


if __name__ == '__main__':
    args = parse_args()

    features, labels = data_loader.get_data(
        data_path=args.in_file,
        k=args.lsb_bits,
    )

    rnn = BranchRNN(
        features.shape[1],
        units=args.units,
        hidden_layers=args.hidden_layers,
        window_size=args.window_size,
        tfboard_log=args.tfboard_log,
        update_freq=args.tfboard_freq,
        cudnn=args.cudnn,
        gru=args.gru,
        batch_size=args.batch_size,
        history_size=args.history_size,
        epochs=args.epochs,
    )

    preds = np.round(np.concatenate(rnn.online(features, labels)))
    logging.info('%s', str(accuracy_score(labels, preds)))

    if isinstance(args.output_dir, str):
        used_rnn = 'gru' if args.gru else 'lstm'
        out_file = f'pred_{used_rnn}_{args.window_size}w-{args.units}u-{args.batch_size}b-{args.epochs}e-{args.history_size}h.csv'
        np.savetxt(
            exp_io.create_filepath(os.path.join(args.output_dir, out_file)),
            preds,
        )
