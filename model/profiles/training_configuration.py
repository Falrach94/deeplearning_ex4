
# training configuration abstraction
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter


class TrainingConfiguration:
    def __init__(self, model_descriptor=None, loss_descriptor=None, optimizer_descriptor=None, data_descriptor=None):
        self.model = model_descriptor
        self.loss = loss_descriptor
        self.optimizer = optimizer_descriptor
        self.data = data_descriptor


    def get_name(self):
        model = '?' if self.model is None else self.model.text()
        opt = '?' if self.optimizer is None else self.optimizer.text()
        loss = '?' if self.loss is None else self.loss.text()
        data = '?' if self.data is None else self.data.text()
        return f"{model}-{loss}-{opt}-{data}"

    def is_complete(self):
        return not (self.model is None or self.optimizer is None or self.loss is None or self.data is None)

    def __eq__(self, other):
        return other.model == self.model \
               and other.loss == self.loss \
               and other.optimizer == self.optimizer \
               and other.data == self.data

    def clone(self):
        return TrainingConfiguration(self.model.clone(), self.loss.clone(), self.optimizer.clone(), self.data.clone())

    def to_json(self):
        return '{"model":'+self.model.to_json()\
            + ', "loss":'+self.loss.to_json()\
            + ', "optimizer":'+self.optimizer.to_json()\
            + ', "reader":'+self.data.to_json()\
            + '}'

    @staticmethod
    def from_json(csv_dic):
        return TrainingConfiguration(Descriptor.from_json(csv_dic['model']),
                                    Descriptor.from_json(csv_dic['loss']),
                                    Descriptor.from_json(csv_dic['optimizer']),
                                    Descriptor.from_json(csv_dic['reader']))

