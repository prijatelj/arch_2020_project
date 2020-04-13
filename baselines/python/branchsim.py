import sys
import argparse

class BranchPredict():
    total = 0
    hits = 0
    bht_list = {}

    def branch_predictor(self, pc_addr, curStatus, n, ght):
        self.total += 1
        if n == 1:
            # 1-bit predictor
            if ght not in self.bht_list:
                self.bht_list[ght] = {}
                if curStatus == 0:
                    self.hits += 1
                    self.bht_list[ght][pc_addr] = 0
                else:
                    self.bht_list[ght][pc_addr] = 1
            else:
                if pc_addr in self.bht_list[ght]:
                    # if pc_addr in history get the previous branch status and update it with the current branch status
                    prevStatus = self.bht_list[ght][pc_addr]
                    if prevStatus == curStatus:
                        self.hits += 1
                    else:
                        self.bht_list[ght][pc_addr] = curStatus
                else:
                    # if pc_addr is not in history save the current branch status
                    if curStatus == 0:
                        self.hits += 1
                        self.bht_list[ght][pc_addr] = 0
                    else:
                        self.bht_list[ght][pc_addr] = 1
        if n == 2:
            # 2-bit predictor
            if ght not in self.bht_list:
                self.bht_list[ght] = {}
                if curStatus == 0:
                    self.hits += 1
                    self.bht_list[ght][pc_addr] = 0
                elif curStatus == 1:
                    self.bht_list[ght][pc_addr] = 1
            else:
                # if pc_addr in history get the previous branch status and update it with the current branch status
                if pc_addr in self.bht_list[ght]:
                    prevStatus = self.bht_list[ght][pc_addr]
                    newStatus = None
                    if curStatus == 0:
                        if prevStatus == 0:
                            self.hits += 1
                            newStatus = 0
                        elif prevStatus == 1:
                            self.hits += 1
                            newStatus = 0
                        elif prevStatus == 2:
                            newStatus = 0
                        elif prevStatus == 3:
                            newStatus = 2
                    else:
                        if prevStatus == 0:
                            newStatus = 1
                        elif prevStatus == 1:
                            newStatus = 3
                        elif prevStatus == 2:
                            self.hits += 1
                            newStatus = 3
                        elif prevStatus == 3:
                            self.hits += 1
                            newStatus = 3
                    self.bht_list[ght][pc_addr] = newStatus
                else:
                    if curStatus == 0:
                        self.hits += 1
                        self.bht_list[ght][pc_addr] = 0
                    elif curStatus == 1:
                        self.bht_list[ght][pc_addr] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Branch Prediction Simulator', allow_abbrev=False)
    parser.add_argument('-m', type=int, help="global branch history bits", choices=range(0, 12), default=6)
    parser.add_argument('-n', type=int, help="predictor type (1-bit/2-bit)", choices=range(1, 3), default=1)
    parser.add_argument('-k', type=int, help="number of LSB bits of PC to consider",
                        choices=range(1, 13), default=8)
    parser.add_argument('-f', required=True, help="input file name")
    args = parser.parse_args()
    filename = args.f

    # initializing the global history table
    m = int(args.m)
    n = int(args.n)
    k = int(args.k)

    if m > 0:
        ght = '0' * m
    else:
        ght = 'dummy'

    predict = BranchPredict()
    try:
        with open(filename, 'r') as reader:
            for line in reader.readlines():
                rec = line.strip().split(" ")
                pc_addr = bin(int(rec[0], 16))[-k:]
                decision = rec[1]
                branch_taken = None
                if m >= 0:
                    if decision == "T":
                        branch_taken = 1
                        predict.branch_predictor(pc_addr, 1, n, ght)
                    else:
                        branch_taken = 0
                        predict.branch_predictor(pc_addr, 0, n, ght)
                if m > 0:
                    # shifting the global history table by 1 bit
                    ght = ght[1:] + str(branch_taken)
            print('dataset: ', filename.split('/')[-1])
            print('(m,n): ', m, n)
            print('LSB bits of PC used: ', k)
            print('Total: ', predict.total)
            print('Hits: ', predict.hits)
            print('Misprediction rate: ', round(((predict.total - predict.hits) / predict.total) * 100, 4), '%')
            entries_used = 0
            for key in predict.bht_list:
                entries_used += len(predict.bht_list[key])
            print('Entries Used: ', entries_used)
    except:
        print('Error reading ', filename)
