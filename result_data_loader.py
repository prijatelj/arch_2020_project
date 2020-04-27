# Loads results file where each line is in this format:
# <prediction. either T/NT>

import numpy as np

def get_data(data_path):
    targets = []

    with open(data_path, 'r') as reader:
        for line in reader.readlines():
            rec = line.strip().split(" ")

            target = ''
            if rec[0] == "T":
                target = 1
            else:
                target = 0

            targets.append(target)

    return np.asarray(targets).astype('int')