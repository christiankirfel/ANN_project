import tarfile, pickle

tar = tarfile.open('ANN_out.tar')
tar.extractall(path='tempout/')
tar.close()

model_history_array = pickle.load(open('tempout/model_history_array.pickle','rb'))
#print(model_history_array)
#import pprint
#pprint.pprint(model_history_array)
loss = []
model_1_loss = []
model_loss = []
val_loss = []
val_model_1_loss = []
val_model_loss = []
for el in model_history_array:
    loss.append(el['loss'])
    model_1_loss.append(el['model_1_loss'])
    model_loss.append(el['model_loss'])
    val_loss.append(el['val_loss'])
    val_model_1_loss.append(el['val_model_1_loss'])
    val_model_loss.append(el['val_model_loss'])

import matplotlib.pyplot as plt

plt.plot(loss)
plt.plot(val_loss)
plt.savefig('test_loss.png')
plt.clf()

plt.plot(model_1_loss)
plt.plot(val_model_1_loss)
plt.savefig('test_model_1_loss.png')
plt.clf()

plt.plot(model_loss)
plt.plot(val_model_loss)
plt.savefig('test_model_loss.png')
plt.clf()