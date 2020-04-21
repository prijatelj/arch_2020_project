import argparse
from tqdm import tqdm
import torch
import numpy as np
from os import path, makedirs
from data_loader import get_data


class Model():
    def __init__(self, input_size):
        # initiate with input_size + bias
        self.weights = np.zeros(shape=input_size + 1)


PREDICTION = {-1: 'NT', 1: 'T'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Run Single/Multi-layer Perceptron')
    parser.add_argument('-k', type=int, help="Number of LSB bits of PC to consider",
                        choices=range(1, 13), default=8)
    parser.add_argument('-m', type=int, help="global branch history bits", choices=range(0, 12), default=6)
    parser.add_argument('--file', '-f', required=True, help="Input file name.")
    parser.add_argument('--output', '-o', help='Output file for the predictions.', type=str)

    args = parser.parse_args()

    if not path.exists(path.split(args.output)[0]):
        makedirs(path.split(args.output)[0])

    torch.manual_seed(0)
    np.random.seed(0)

    data = get_data(args.file, args.k)
    step = 0
    total_accuracy = 0.
    step = 0

    loop = tqdm(range(0, len(data[0])))

    history_str = '0' * args.m
    perceptrons = {}
    predictions = []

    for idx in loop:
        addresses = ''.join(list(data[0][idx]))

        history_input = np.array(list(history_str)).astype('int')
        history_input[history_input == 0] = -1

        if addresses not in perceptrons:
            perceptrons[addresses] = Model(args.m)

        outputs = perceptrons[addresses].weights[1:] * history_input
        outputs = sum(outputs) + perceptrons[addresses].weights[0]

        if outputs >= 0:
            y_pred = 1
        else:
            y_pred = -1

        y_true = int(data[1][idx])
        y_true_1 = y_true
        if y_true_1 == 0:
            y_true_1 = -1

        accuracy = round(np.sum(y_true_1 == y_pred), 4)
        total_accuracy += accuracy

        if y_true_1 == y_pred:
            perceptrons[addresses].weights[1:] += history_input
        else:
            perceptrons[addresses].weights[1:] -= history_input

        predictions.append([PREDICTION[y_pred]])

        history_str = history_str[1:] + str(y_true)
        step += 1

    np.savetxt(args.output, predictions, fmt='%s')
    print(f'Misprediction rate: {round((1 - (total_accuracy / step)) * 100, 2)}')
