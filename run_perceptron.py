import argparse
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy, relu
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
        self.hidden = Linear(input_size, input_size)
        self.output = Linear(input_size, 2)

    def forward(self, x):
        if self.multi_layer:
            x = self.hidden(x)
            x = relu(x)

        x = self.output(x)

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Run Single/Multi-layer Perceptron')
    parser.add_argument('-k', type=int, help="Number of LSB bits of PC to consider",
                        choices=range(1, 13), default=8)
    parser.add_argument('--file', '-f', required=True, help="Input file name.")
    parser.add_argument('--multi_layer', '-ml', action='store_true', help="Train a multi layer perceptron.")
    parser.add_argument('--batch_size', '-bs', type=int, help="Batch size.", default=32)
    parser.add_argument('--prefix', '-p', help='Prefix to save the model.', type=str)

    args = parser.parse_args()

    log_path = path.join('./log/', args.prefix)

    if not path.exists(log_path):
        makedirs(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Perceptron(args.k, args.multi_layer)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(log_path)
    dummy_input = torch.zeros(1, args.k)
    writer.add_graph(model, dummy_input)

    model = model.to(device)

    data = get_data(args.file, args.k)
    model.train()
    running_loss = 0.
    step = 0
    running_acc = 0.
    total_accuracy = 0.

    step = 0
    log = 1000

    batch_size = args.batch_size

    loop = tqdm(range(0, len(data[0]), batch_size))

    for idx in loop:
        imgs = torch.tensor(data[0][idx:idx + batch_size], dtype=torch.float, device=device)
        labels = torch.tensor(data[1][idx:idx + batch_size], dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = cross_entropy(outputs, labels)

        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = labels.cpu().numpy()

        accuracy = round(np.sum(y_true == y_pred) / len(y_pred), 4)

        loss.backward()

        running_loss += loss.item()
        running_acc += accuracy
        total_accuracy += accuracy

        optimizer.step()

        if step % log == 0:
            loss_board = running_loss / log
            acc_board = running_acc / log
            writer.add_scalar('running_loss', loss_board, step)
            writer.add_scalar('running_acc', acc_board, step)
            running_loss = 0.
            running_acc = 0.

        step += 1
        loop.set_postfix(loss=loss.item(), accuracy=accuracy)

    print(f'Accuracy: {total_accuracy / step}')
