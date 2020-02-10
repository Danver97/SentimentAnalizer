from model.model_class import Model
from model.model_serializer import ModelSerializer
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

model_en = Model('en')
model_en.trainModel(None, lemmatized_file="./model/data_lemmatized_en.csv", gridsearch=False)

model_it = Model('it')
estimator_it = model_it.trainModel(None, lemmatized_file="./model/data_lemmatized_it.csv", gridsearch=False)

ms = ModelSerializer()
ms.dumpModel(model_en, './model/model_en.pkl')
ms.dumpModel(model_it, './model/model_it.pkl')
print('Models dumped to file')
