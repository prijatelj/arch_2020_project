import argparse
from torch.nn import Linear, Module, LSTM
from torch.nn.functional import cross_entropy
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os import path, makedirs
from data_loader import get_data


class myLSTM(Module):
    def __init__(self, input_size=12, multi_layer=True):
        super(myLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = input_size
        self.batch_size = 1
        self.num_layers = 1

        self.multi_layer = multi_layer
        self.lstm = LSTM(input_size, input_size)
        self.output = Linear(input_size, 2)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        x, self.hidden = self.lstm(x)
        x = self.output(x.view(1, -1))

        return x


class Model():
    def __init__(self, input_size, multi_layer, device):
        self.model = myLSTM(input_size, multi_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)
        self.model.to(device)
        self.model.train()
        self.loss = None


PREDICTION = {0: 'NT', 1: 'T'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM')
    parser.add_argument('-k', type=int, help="Number of LSB bits of PC to consider",
                        choices=range(1, 13), default=8)
    parser.add_argument('-m', type=int, help="global branch history bits", choices=range(0, 12), default=6)
    parser.add_argument('--file', '-f', required=True, help="Input file name.")
    parser.add_argument('--multi_layer', '-ml', action='store_true', help="Train a multi layer myLSTM.")
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
    myLSTMs = {}
    predictions = []

    for idx in loop:
        addresses = ''.join(list(data[0][idx]))
        labels = torch.tensor(data[1][idx:idx + batch_size], dtype=torch.long, device=device)

        if addresses not in myLSTMs:
            myLSTMs[addresses] = Model(global_history, args.multi_layer, device)
            myLSTMs[addresses].model.hidden = myLSTMs[addresses].model.init_hidden()

        history_input = np.zeros(shape=(1, 1, global_history))
        history_input[0, :] = np.array(list(history_str)).astype('int')
        history_input[history_input == 0] = -1

        history = torch.tensor(history_input, dtype=torch.float, device=device)

        myLSTMs[addresses].optimizer.zero_grad()
        outputs = myLSTMs[addresses].model(history)
        myLSTMs[addresses].loss = cross_entropy(outputs, labels)

        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = labels.cpu().numpy()

        predictions.append([PREDICTION[y_pred[0]]])

        accuracy = round(np.sum(y_true == y_pred) / len(y_pred), 4)

        myLSTMs[addresses].loss.backward()

        running_loss += myLSTMs[addresses].loss.item()
        running_acc += accuracy
        total_accuracy += accuracy

        myLSTMs[addresses].optimizer.step()

        if step % log == 0:
            loss_board = running_loss / log
            acc_board = running_acc / log
            writer.add_scalar('running_loss', loss_board, step)
            writer.add_scalar('running_acc', acc_board, step)
            running_loss = 0.
            running_acc = 0.

        history_str = history_str[1:] + str(y_true[0])

        step += 1
        loop.set_postfix(loss=myLSTMs[addresses].loss.item(), accuracy=accuracy)

    np.savetxt(args.output, predictions, fmt='%s')
    print(f'Misprediction rate: {round((1 - (total_accuracy / step)) * 100, 2)}')