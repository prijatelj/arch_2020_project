import numpy as np


def get_data(data_path, k=0):
    samples = []
    targets = []

    with open(data_path, 'r') as reader:
        for line in reader.readlines():
            rec = line.strip().split(" ")
            addr_binary = bin(int(rec[0], 16))[2 + k:]

            target = ''
            if rec[1] == "T":
                target = 1
            else:
                target = 0

            sample = np.array(list(addr_binary))

            samples.append(sample)
            targets.append(target)

    return np.asarray(samples).astype('int'), np.asarray(targets).astype('int')
