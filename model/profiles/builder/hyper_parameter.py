class HyperParameter:

    def __init__(self, name, type, default, min_val=None, max_val=None):
        self.name = name
        self.min = min_val
        self.max = max_val
        self._value = None
        self.type = type
        self.set_value(default)

    def get_value(self):
        return self._value

    def set_value(self, value):
        if self.type == 'int':
            self._value = int(value)
        elif self.type == 'float':
            self._value = float(value)
        elif self.type == 'bool':
            self._value = bool(value)
        else:
            self._value = value

    def clone(self):
        return HyperParameter(self.name, self.type, self._value, self.min, self.max)

    def to_json(self):
        min = '' if self.min is None else (', "min":' + str(self.min))
        max = '' if self.max is None else (', "max":' + str(self.max))
        return '{"name":"' + self.name + '"' \
             + ', "type":"' + self.type + '"' \
             + min \
             + max \
             + ', "value":' + str(float(self._value)) \
             + '}'

    @staticmethod
    def from_json(csv_dic):
        min = csv_dic.get('min')
        max = csv_dic.get('max')
        return HyperParameter(csv_dic['name'],
                              csv_dic['type'],
                              float(csv_dic['value']),
                              None if min is None else float(min),
                              None if max is None else float(max))
