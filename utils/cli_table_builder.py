import numpy as np


class TableBuilder:

    def __init__(self):
        self.max_line_len = 0

        self.inter = ' │ '
        self.vl = '│'
        self.hl = '─'

        self.blocks = [None]
        self.block_sizes = []

    def new_block(self):
        self.blocks.append(None)

    def get_line_len(self, block):
        return len(self.inter) * (len(self.block_sizes[block])-1) + np.sum(self.block_sizes[block])

    def add_line(self, *args):
        if self.blocks[-1] is None:
            self.blocks[-1] = [list(args)]
            self.block_sizes.append(np.array([len(s) for s in args]))
        else:
            if len(self.block_sizes[-1]) != len(args):
                raise Exception(f'collumn number ({len(args)}) does not match current block ({len(self.blocks[-1])})')
            self.blocks[-1].append(list(args))

            cell_lengths = np.array([len(s) for s in args])
            self.block_sizes[-1] = np.maximum(cell_lengths, self.block_sizes[-1])

    def print_hline(self, p):
        if p == -1:
            print('┌' + self.hl * (self.max_line_len) + '┐')
        if p == 0:
            print('├' + self.hl * (self.max_line_len) + '┤')
        if p == 1:
            print('└' + self.hl * (self.max_line_len) + '┘')

    @staticmethod
    def adjust_element(txt, target_len):
        return txt + ' '*(target_len-len(txt))

    def build_line(self, elements, block_sizes):
        line = self.inter.join([self.adjust_element(el, bs) for el, bs in zip(elements, block_sizes)])
        return self.vl \
            + line + ' '*(self.max_line_len-len(line))\
            + self.vl

    def print(self):
        self.max_line_len = max([self.get_line_len(i) for i in range(len(self.blocks))])

        self.print_hline(-1)
        for i, (lines, target_size) in enumerate(zip(self.blocks, self.block_sizes)):
            if i != 0:
                self.print_hline(0)

            for line in lines:
                print(self.build_line(line, target_size))

        self.print_hline(1)
