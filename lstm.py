"""Keras predictor for"""
import argparse

from keras.layers import LSTM, Activation, Input, Dense
from keras.models import Model
from keras.callbacks.tensorboard_v1 import TensorBoard
import numpy as np

import data_loader

def window_data(x, window_size, batch_size=1):
    """Delievers the data in a window format, appending zeros to beginning of
    window buffer and ending at last element of data.
    """
    assert window_size >= 1

    if len(x.shape) == 1:
        windowed_x = np.append(np.zeros(window_size-1), x)
    else:
        windowed_x = np.vstack((
            np.zeros((window_size - 1, x.shape[1], x.shape[1])),
            x,
        ))

    # format of windowed data?
    for i in range(window_size + 1, len(windowed_x) + 1):
        yield windowed_x[i - window_size: i].flatten,


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
    ):
        self.input_vector = Input(shape=input_shape * window_size)
        self.window_size = window_size

        x = self.input_vector
        for i in range(hidden_layers):
            x = LSTM(units)(x)

        self.output_vector = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=self.input_vector, outputs=self.output_vector)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            batch_size=1,
        )

        self.callbacks = [TensorBoard(
            tfboard_log,
            update_freq=10000,
        )]

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def online(self, x, y):
        # TODO save history and repeat in order for multiple epochs?

        preds = []
        for i in len(x):
            win_x = window_data(x, self.window_size)
            preds.append(self.predict(x, batch_size=1))
            self.fit(
                x,
                y,
                batch_size=1,
                shuffle=False,
                epochs=1,
                callbacks=self.callbacks,
            )

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
        '--out_file',
        default=None,
        help='The CSV output file path.',
    )

    parser.add_argument(
        '-h',
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

    return parser.parse_args()


if __name__ == '__main__':
    combine(**vars(parse_args()))


    features, labels = data_loader(data_path=args.in_file)

    lstm = BranchLSTM(
        x.shape[1],
        units=args.units,
        hidden_layers=args.hidden_layers,
        window_size=args.window_size,
    )

    preds = lstm.online(features, labels)

    np.savetxt(args.out_file, preds)
