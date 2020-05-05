import argparse
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os import path, makedirs
from data_loader import get_data


class Perceptron(Module):
    def __init__(self, input_size=12, multi_layer=True):
        super(Perceptron, self).__init__()
        self.multi_layer = multi_layer
        if self.multi_layer:
            self.hidden = Linear(input_size, input_size)
        self.output = Linear(input_size, 2)

    def forward(self, x):
        if self.multi_layer:
            x = self.hidden(x)

        x = self.output(x)

        return x


class Model():
    def __init__(self, input_size, multi_layer, device):
        self.model = Perceptron(input_size, multi_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)
        self.model.to(device)
        self.model.train()
        self.loss = None


PREDICTION = {0: 'NT', 1: 'T'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Run Single/Multi-layer Perceptron')
    parser.add_argument('-k', type=int, help="Number of LSB bits of PC to consider",
                        choices=range(1, 13), default=8)
    parser.add_argument('-m', type=int, help="global branch history bits", choices=range(0, 12), default=6)
    parser.add_argument('--file', '-f', required=True, help="Input file name.")
    parser.add_argument('--multi_layer', '-ml', action='store_true', help="Train a multi layer perceptron.")
    parser.add_argument('--batch_size', '-bs', type=int, help="Batch size.", default=1)
    parser.add_argument('--prefix', '-p', help='Prefix to save the model.', type=str)
    parser.add_argument('--output', '-o', help='Output file for the predictions.', type=str)

    args = parser.parse_args()

    if not path.exists(path.split(args.output)[0]):
        makedirs(path.split(args.output)[0])

    torch.manual_seed(0)
    np.random.seed(0)

    log_path = path.join('./log/', args.prefix)

    if not path.exists(log_path):
        makedirs(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_history = args.m

    writer = SummaryWriter(log_path)
    data = get_data(args.file, args.k)
    running_loss = 0.
    step = 0
    running_acc = 0.
    total_accuracy = 0.

    step = 0
    log = 1000

    batch_size = args.batch_size

    loop = tqdm(range(0, len(data[0]), batch_size))

    history_str = '0' * global_history
    perceptrons = {}
    predictions = []

    for idx in loop:
        addresses = ''.join(list(data[0][idx]))
        labels = torch.tensor(data[1][idx:idx + batch_size], dtype=torch.long, device=device)

        if addresses not in perceptrons:
            perceptrons[addresses] = Model(global_history, args.multi_layer, device)
            model_parameters = filter(lambda p: p.requires_grad, perceptrons[addresses].model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])

        history_input = np.zeros(shape=(1, global_history))
        history_input[0, :] = np.array(list(history_str)).astype('int')
        history_input[history_input == 0] = -1

        history = torch.tensor(history_input, dtype=torch.float, device=device)

        perceptrons[addresses].optimizer.zero_grad()
        outputs = perceptrons[addresses].model(history)
        perceptrons[addresses].loss = cross_entropy(outputs, labels)

        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = labels.cpu().numpy()

        predictions.append([PREDICTION[y_pred[0]]])

        accuracy = round(np.sum(y_true == y_pred) / len(y_pred), 4)

        perceptrons[addresses].loss.backward()

        running_loss += perceptrons[addresses].loss.item()
        running_acc += accuracy
        total_accuracy += accuracy

        perceptrons[addresses].optimizer.step()

        if step % log == 0:
            loss_board = running_loss / log
            acc_board = running_acc / log
            writer.add_scalar('running_loss', loss_board, step)
            writer.add_scalar('running_acc', acc_board, step)
            running_loss = 0.
            running_acc = 0.

        history_str = history_str[1:] + str(y_true[0])

        step += 1
        loop.set_postfix(loss=perceptrons[addresses].loss.item(), accuracy=accuracy)

    np.savetxt(args.output, predictions, fmt='%s')
    print(f'Misprediction rate: {round((1 - (total_accuracy / step)) * 100, 2)}')
