

class ListChangedArgs:
    ADDED = 0
    REMOVED = 1
    UPDATED = 2
    RESET = 3

    def __init__(self, type, index=-1, data=None):
        self.type = type
        self.index = index
        self.data = data

