import os
import uuid

import torch


class Checkpoint:
    def __init__(self, name):
        self.data = None
        self.name = name

    def is_loaded(self):
        return self.data is not None

    def is_valid(self):
        return os.path.exists(f'assets/checkpoints/{self.name}.ckp')

    def erase(self):
        path = f'assets/checkpoints/{self.name}.ckp'
        if os.path.exists(path):
            os.remove(path)
        else:
            print("tried to erase non existing checkpoint file")

    def update(self, data):
        self.data = data
        torch.save(self.data, f'assets/checkpoints/{self.name}.ckp')

    def reset(self, data):
        self.data = data

    def load(self):
        if self.is_loaded():
            return True

        try:
            self.data = torch.load(f'assets/checkpoints/{self.name}.ckp', 'cuda')
            return True
        except FileNotFoundError:
            print(f"Checkpoint {self.id} couldn't be loaded!")
            return False
        return False

    '''
    def to_json(self):
        return '{"id":"' + str(self.name) + '"}'

    @staticmethod
    def from_json(dic):
        return Checkpoint(uid=uuid.UUID(dic['id']))
    '''
