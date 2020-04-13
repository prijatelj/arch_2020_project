"""Keras predictor for"""
import argparse

from keras.layers import LSTM, Activation, Input, Dense
from keras.models import Model
from keras.callbacks.tensorboard_v1 import TensorBoard
import numpy as np
from sklearn.metrics import accuracy_score

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
            np.zeros((max(1, window_size - 1), x.shape[1])),
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
    ):
        self.input_vector = Input(shape=[window_size, input_shape])
        self.window_size = window_size

        x = self.input_vector
        for i in range(hidden_layers):
            x = LSTM(units)(x)

        self.output_vector = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=self.input_vector, outputs=self.output_vector)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        self.callbacks = [TensorBoard(
            tfboard_log,
            update_freq=10000,
        )]

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def online(self, x, y):
        # TODO save history and repeat in order for multiple epochs?
        y = y.reshape(-1,1)

        preds = []
        for i,win_x in enumerate(window_data(x, self.window_size)):
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

    return parser.parse_args()


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
    )

    preds = np.concatenate(lstm.online(features, labels))
    print(accuracy_score(labels, preds))

    if isinstance(args.out_file, str):
        np.savetxt(args.out_file, preds)
