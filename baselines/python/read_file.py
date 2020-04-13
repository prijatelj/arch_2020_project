import numpy as np

if __name__ == '__main__':
    file_10k = '../../trace_files/gcc-10K.txt'
    file_8m = '../../trace_files/gcc-8M.txt'
    data = np.empty(shape=(0,2))

    with open(file_8m, 'r') as reader:
        for line in reader.readlines():
            rec = line.strip().split(" ")
            addr_binary = bin( int(rec[0], 16) )[2:]
            prediction = ''
            if rec[1] == "T":
              prediction = 1
            else:
              prediction = 0

            row = np.array([[addr_binary, prediction]])
            data = np.vstack( (data, row) )

    # print(len(data))