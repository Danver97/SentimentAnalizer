import sys
from model.model_serializer import ModelSerializer
# from model.model_serializer import Model

if sys.argv[1] == 'it':
    file_model = './model/model_it.pkl'
elif sys.argv[1] == 'en':
    file_model = './model/model_en.pkl'

model = ModelSerializer().loadModel(file_model)

dataToSendBack = model.predict(sys.argv[2])[0]

print(dataToSendBack)
sys.stdout.flush()
