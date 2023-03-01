from model.reader.big_data_reader import BigDataReader
from model.reader.biggest_data_reader import BiggestDataReader
from model.reader.data_reader import DataReader
from model.reader.elpv_reader import InactiveReader
from model.reader.small_reader import SmallDataReader

from model.reader.tree_data_reader import DataReaderTree
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter


class DataReaders:

    reader = ['default', 'small', 'tree', 'inactive', 'big', 'biggest']

    @staticmethod
    def get_descriptor(name):

        if name == 'tree':
            return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                     HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                     HyperParameter('Transform', 'bool', 0, 0),
                                     HyperParameter('Set', 'int', 0, 0)])
        if name == 'inactive':
            return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                     HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                     HyperParameter('Transform', 'bool', 0, 0)])
        if name == 'big':
            return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                     HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                     HyperParameter('Transform', 'bool', 1, 0)])

        if name == 'biggest':
            return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                     HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                     HyperParameter('Transform', 'bool', 1, 0)])
        if name == 'small':
            return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                     HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                     HyperParameter('Transform', 'bool', 1, 0)])

        return Descriptor(name, [HyperParameter('BatchSize', 'int', 64,  1, None),
                                 HyperParameter('ValSplit', 'float', 0.2, 0.01, 0.99),
                                 HyperParameter('Transform', 'bool', 0, 0),
                                 HyperParameter('Oversample', 'bool', 0, 0),
                                 HyperParameter('Forest', 'bool', 0, 0)])

    @staticmethod
    def instantiate(descriptor):
        if descriptor.name == 'default':
            return DataReader()
        if descriptor.name == 'tree':
            return DataReaderTree()
        if descriptor.name == 'inactive':
            return InactiveReader()
        if descriptor.name == 'big':
            return BigDataReader()
        if descriptor.name == 'biggest':
            return BiggestDataReader()
        if descriptor.name == 'small':
            return SmallDataReader()

        raise BaseException(f'Loss name "{descriptor.name}" not recognized!')
