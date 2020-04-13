"""Keras predictor for"""
from keras.layers import LSTM, Activation, Input, Dense
from keras.models import Model
import numpy as np


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
        units=2,
        hidden_layers=1,
        output_shape=1,
        input_buffer_size=1,
    ):
        self.input_vector = Input(shape=input_shape * input_buffer_size)
        self.input_buffer_size = input_buffer_size

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

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def online(self, x, y):
        # TODO make windows

        for i in len(x):
            win_x = window_data(x, self.input_buffer_size)
            preds = self.predict(x, batch_size=1)
            self.fit(x, y, batch_size=1)

        return preds
