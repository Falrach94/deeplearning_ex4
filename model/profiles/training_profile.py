from model.profiles.training_configuration import TrainingConfiguration
from model.profiles.training_session import Session
import numpy as np


class TrainingProfile:
    def __init__(self, configuration, name=None, sessions=[]):
        self.name = name if name is not None else configuration.get_name()
        self.configuration = configuration
        self.sessions = sessions

    def to_json(self):
        return '{"name":"'+self.name\
                + '", "config":' + self.configuration.to_json()\
                + ', "sessions":['+', '.join([s.to_json() for s in self.sessions]) + ']'\
                + '}'

    @staticmethod
    def from_json(csv_dic):
        sessions = csv_dic.get('sessions')
        return TrainingProfile(TrainingConfiguration.from_json(csv_dic['config']),
                               csv_dic['name'],
                               [] if sessions is None else [Session.from_json(s) for s in sessions])

    def get_average_epoch_time(self):
        times = [s.epoch_time for s in self.sessions]
        times = np.array([t for sl in times for t in sl])
        if len(times) == 0:
            return 0
        av = times.mean()
        return av

