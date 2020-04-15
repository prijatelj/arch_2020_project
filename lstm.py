"""Keras predictor for"""
import argparse

from keras import backend
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
    ):
        self.input_vector = Input(shape=[window_size, input_shape])
        self.window_size = window_size
        #if batch_size != 1 and history_size <= 0:
        #    raise NotImplementedError(' '.join([
        #        'Batch size greater than 1 is not implemented without a',
        #        'history of 1 or greater.'
        #    ]))
        self.batch_size = max(1, batch_size)
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
        # TODO save history and repeat in order for multiple epochs?
        y = y.reshape(-1, 1)

        print(f'y reshape shape = {y.shape}')

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
                        epochs=1,
                        callbacks=self.callbacks,
                    )
                else:
                    self.fit(
                        x=win_x,
                        y=y[i],
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=1,
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

                if i % self.batch_size == 0:
                    batch_features = np.concatenate(batch_features, axis=0),

                    # NOTE do following for probably faster simulation
                    preds += self.predict(
                        batch_features,
                        batch_size=self.batch_size,
                    ).tolist()

                    # fit every batch size
                    self.fit(
                        x=batch_features,
                        y=y[max(0, i + 1 - self.batch_size):i + 1],
                        batch_size=self.batch_size,
                        shuffle=False,
                        epochs=1,
                        callbacks=self.callbacks,
                    )

                    batch_features = []

        return preds


def get_tf_config(cpu_cores=1, cpus=1, gpus=0, allow_soft_placement=True):
    return tf.ConfigProto(
        intra_op_parallelism_threads=cpu_cores,
        inter_op_parallelism_threads=cpu_cores,
        allow_soft_placement=allow_soft_placement,
        device_count={
            'CPU': cpus,
            'GPU': gpus,
        } if gpus >= 0 else {'CPU': cpus},
    )


def add_hardware_args(parser):
    """Adds the arguments detailing the hardware to be used."""
    # TODO consider packaging as a dict/NestedNamespace
    # TODO consider a boolean or something to indicate when to pass a
    # tensorflow session or to use it as default

    parser.add_argument(
        '--cpu',
        default=1,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )
    parser.add_argument(
        '--which_gpu',
        default=None,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )


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
        '--out_file',
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
        default=2,
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
        choices=range(1,13),
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

    add_hardware_args(parser)

    args = parser.parse_args()

    # Set the Hardware
    backend.set_session(tf.Session(config=get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )))

    return args


if __name__ == '__main__':
    args = parse_args()

    features, labels = data_loader.get_data(
        data_path=args.in_file,
        k=args.lsb_bits,
    )

    lstm = BranchRNN(
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
    )

    preds = np.concatenate(lstm.online(features, labels))
    print(accuracy_score(labels, np.round(preds)))

    if isinstance(args.out_file, str):
        np.savetxt(exp_io.create_filepath(args.out_file), preds)
