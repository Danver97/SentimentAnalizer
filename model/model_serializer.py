import pickle
from model.model_class import Model

class ModelSerializer:
    def __init__(self):
        pass

    def dumpModel(self, model, filename):
        with open(filename, 'wb') as file_dump:
            pickle.dump(model, file_dump, protocol=pickle.HIGHEST_PROTOCOL)

    def loadModel(self, filename):
        with open(filename, 'rb') as file_load:
            model = pickle.load(file_load)
        return model