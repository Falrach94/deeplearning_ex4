import copy

import numpy as np


class ConfigUpdateField:
    def __init__(self, keys, range):
        self.keys = keys
        self.range = range

    def update(self, config, i):
        target = config
        for j, k in enumerate(self.keys):
            if j != len(self.keys)-1:
                target = target[k]
            else:
                target[k] = self.range[i]
        return config

    def count(self):
        return len(self.range)


class ConfigUpdater:
    class Iterator:
        def __init__(self, updater):
            self.updater = updater
            self.selection = [0]*len(updater.update_fields)
            self.max_val = [f.count() for f in updater.update_fields]

            self.i = 0

        def __next__(self):
            if self.i == len(self.updater):
                raise StopIteration

            config = self.updater[self.selection]

            for ix_field, (sel, max) in enumerate(zip(self.selection, self.max_val)):
                self.selection[ix_field] += 1
                if self.selection[ix_field] == max:
                    self.selection[ix_field] = 0
                else:
                    break

            self.i += 1

            return config

    def __init__(self, base_config, update_fields):
        self.base_config = base_config
        self.update_fields = update_fields

    def __iter__(self):
        return ConfigUpdater.Iterator(self)

    def __getitem__(self, selection):
        config = copy.deepcopy(self.base_config)

        for i, uf in enumerate(self.update_fields):
            config = uf.update(config, selection[i])

        return config

    def __len__(self):
        return np.prod([f.count() for f in self.update_fields])

