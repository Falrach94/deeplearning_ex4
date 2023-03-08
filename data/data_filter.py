
class DataFilter:
    def __init__(self):
        pass

    @staticmethod
    def filter_out_non_labeled_data(data):
        return data[(data.inactive != 0) | (data.crack != 0)]

