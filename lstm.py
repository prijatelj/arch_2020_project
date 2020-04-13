"""Keras predictor for"""
from keras.layers import LSTM, Activation, Input
from keras.models import Model

class BranchLSTM(object):

    def __init__(
        self,
        input_shape,
        units=2,
        hidden_layers=1,
        output_shape=1,
        input_buffer_size=1,
    ):
        self.input_vector = Input(shape=input_shape)

        x = self.input_vector
        for i in range(hidden_layers):
            x = LSTM(units)(x)

        self.output_vector = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=self.input_vector, outputs=self.output_vector)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def online(self, ):
        # TODO make windows?

        #self.predict()
        #self.fit()

        return preds
