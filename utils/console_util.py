import sys

import numpy as np


def print_progress_bar(prefix, i, cnt, suffix, fill_char='█', bar_length=50):

    perc = i/cnt
    filled = round(perc*bar_length)
    not_filled = bar_length-filled
    bar = fill_char*filled + '-'*not_filled
    print(f'\r{prefix}:\t|{bar}| ({i}/{cnt} | {round(perc*100)}%) - {suffix}', end='', flush=True)


class ScreenBuilder:

    def __init__(self):
        self.current_line = 0
        self.current_col = 0

        self.marks={}

    def print_line(self, txt):
        print(txt)
        self.current_line += 1

    def reset_position(self):
        sys.stdout.write('\x1B[0;0H')
        self.current_line = 0

    def go_up(self, y):
        sys.stdout.write(f'\x1B[0;{self.current_line-y}H')
        self.current_line -= y

    def mark_line(self, name):
        self.marks[name] = self.current_line

    def goto_mark(self, name):
        sys.stdout.write(f'\x1B[0;{self.marks[name]}H')
        self.current_line = self.marks[name]

    def has_mark(self, name):
        return name in self.marks


class TableBuilderEx:

    # if name is not None, table will always print to the same location
    def __init__(self, sb: ScreenBuilder, name: str = None):
        self.max_line_len = 0

        self.name = name

        self.inter = ' │ '
        self.vl = '│'
        self.hl = '─'

        self.blocks = [None]
        self.block_sizes = []

        self.sb = sb

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
            self.sb.print_line('┌' + self.hl * (self.max_line_len) + '┐')
        if p == 0:
            self.sb.print_line('├' + self.hl * (self.max_line_len) + '┤')
        if p == 1:
            self.sb.print_line('└' + self.hl * (self.max_line_len) + '┘')

    @staticmethod
    def adjust_element(txt, target_len):
        return txt + ' '*(target_len-len(txt))

    def build_line(self, elements, block_sizes):
        line = self.inter.join([self.adjust_element(el, bs) for el, bs in zip(elements, block_sizes)])
        return self.vl \
            + line + ' '*(self.max_line_len-len(line))\
            + self.vl

    def print(self):

        if self.name is not None:
            if self.sb.has_mark(self.name):
                self.sb.goto_mark(self.name)
            else:
                self.sb.mark_line(self.name)

        self.max_line_len = max([self.get_line_len(i) for i in range(len(self.blocks))])

        self.print_hline(-1)
        for i, (lines, target_size) in enumerate(zip(self.blocks, self.block_sizes)):
            if i != 0:
                self.print_hline(0)

            for line in lines:
                self.sb.print_line(self.build_line(line, target_size))

        self.print_hline(1)
