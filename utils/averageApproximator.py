import numpy as np


class AverageApproximator:
    def __init__(self, initial=1, cnt=10):
        self.vals = [initial]
        self.max_cnt = cnt
        self.average = initial

    def add(self, val):
        self.vals.append(val)
        if len(self.vals) > self.max_cnt:
            self.vals = self.vals[1:self.max_cnt+1]
        self.average = np.mean(self.vals)
        return self.average

