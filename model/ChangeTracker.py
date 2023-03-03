
CONFIDENCE_THRESHOLD = 0.2

class Tracker:

    def __init__(self, training_dl, model):
        self.dl = training_dl
        self.model = model


    def analyse(self):
        self.model.eval()
        for x,y in self.dl:
