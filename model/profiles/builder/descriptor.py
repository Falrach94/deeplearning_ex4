
from model.profiles.builder.hyper_parameter import HyperParameter


class Descriptor:

    def __init__(self, name, hyper_parameters):
        self.name = name
        self.hyperparams = hyper_parameters

    def text(self):
        par = ''

        if self.hyperparams is not None:
            par = f'[{", ".join([str(p.get_value()) for p in self.hyperparams])}]'

        return self.name + par

    def get(self, hp_name):
        if self.hyperparams is None:
            return None
        for hp in self.hyperparams:
            if hp.name.upper() == hp_name.upper():
                return hp
        return None

    def __eq__(self, other):
        if other.name != self.name:
            return False
        if self.hyperparams is None and other.hyperparams is None:
            return True
        if other.hyperparams is None or self.hyperparams is None:
            return False
        if len(self.hyperparams) != len(other.hyperparams):
            return False
        for i in range(len(self.hyperparams)):
            if other.hyperparams[i].get_value() != self.hyperparams[i].get_value() :
                return False
        return True

    def clone(self):
        chp = [hp.clone() for hp in self.hyperparams] if self.hyperparams is not None else None
        return Descriptor(self.name, chp)

    def to_json(self):
        if self.hyperparams is None:
            return '{"name":"'+self.name+'"}'
        hp_ar = [hp.to_json() for hp in self.hyperparams]
        hp_csv = '['+', '.join(hp_ar)+']'
        return '{"name":"'+self.name+'", "params":' + hp_csv + '}'

    @staticmethod
    def from_json(csv_dic):
        name = csv_dic['name']
        csv_params = csv_dic.get('params')
        params = None if csv_params is None\
            else [HyperParameter.from_json(hp) for hp in csv_dic['params']]
        return Descriptor(name, params)

