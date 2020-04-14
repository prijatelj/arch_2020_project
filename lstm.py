"""Keras predictor for"""
import argparse

from keras import backend
from keras.layers import LSTM, Activation, Input, Dense, CuDNNLSTM
from keras.models import Model
from keras.callbacks.tensorboard_v1 import TensorBoard
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf

import data_loader

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


class BranchLSTM(object):

    def __init__(
        self,
        input_shape,
        units=1,
        hidden_layers=1,
        output_shape=1,
        window_size=1,
        tfboard_log='./log/lstm',
        update_freq=10000,
        cudnn=True
    ):
        self.input_vector = Input(shape=[window_size, input_shape])
        self.window_size = window_size

        x = self.input_vector
        for i in range(hidden_layers):
            x = CuDNNLSTM(units)(x) if cudnn else LSTM(units)(x)

        self.output_vector = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=self.input_vector, outputs=self.output_vector)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        self.callbacks = [TensorBoard(
            tfboard_log,
            update_freq=update_freq,
        )]

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def online(self, x, y):
        # TODO save history and repeat in order for multiple epochs?
        y = y.reshape(-1,1)

        print(f'y reshape shape = {y.shape}')

        preds = []
        for i, win_x in enumerate(window_data(x, self.window_size)):
            win_x = win_x[np.newaxis, ...]
            preds.append(self.predict(win_x, batch_size=1))
            self.fit(
                x=win_x,
                y=y[i],
                batch_size=1,
                shuffle=False,
                epochs=1,
                callbacks=self.callbacks,
            )

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
        default='./log/lstm/',
        help='The TensorBoard log path.',
    )

    parser.add_argument(
        '--tfboard_freq',
        default=10000,
        help='The update freq of TensorBoard.',
    )

    mle.add_argument(
        '--cudnn',
        action='store_true',
        help='Uses CuDNN version of LSTM or GRU',
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

    lstm = BranchLSTM(
        features.shape[1],
        units=args.units,
        hidden_layers=args.hidden_layers,
        window_size=args.window_size,
        cudnn=args.cudnn,
    )

    preds = np.concatenate(lstm.online(features, labels))
    print(accuracy_score(labels, np.round(preds)))

    if isinstance(args.out_file, str):
        np.savetxt(args.out_file, preds)
