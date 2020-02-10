import sys
from model.model_serializer import ModelSerializer
from model.model_serializer import Model

file_model_it = './model/model_it.pkl'
file_model_en = './model/model_en.pkl'

model_it = ModelSerializer().loadModel(file_model_it)
model_en = ModelSerializer().loadModel(file_model_en)

print('Models loaded')
sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if line is None or line == '':
        continue
    splits = line.split(' ')
    reqid = splits[0]
    lang = splits[1]
    review = ' '.join(splits[2:])
    if lang == 'it':
        dataToSendBack = model_it.predict(review)[0]
    elif lang == 'en':
        dataToSendBack = model_en.predict(review)[0]
    # Writes to the stdout
    print(reqid + ' ' + dataToSendBack)
    # Flush the stdout buffer and makes written data
    # 'listenable' from the node process
    sys.stdout.flush()
