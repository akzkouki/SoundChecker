import scipy.io.wavfile as wav
import librosa
from sklearn.svm import SVC
import os
import numpy
import pickle
from sklearn.grid_search import GridSearchCV

def getMfcc(filename):
    print(filename)
    y, sr = librosa.load(filename,duration=5.0)
    return librosa.feature.mfcc(y=y, sr=sr)

types= ['Kick','Crash','Hihat','Snare']
# types= ['Kick','Hihat','Snare']
sound_train = []
type_train = []
for type in types:
    print('Reading data of %s...' % type)
    files = os.listdir(os.getcwd() + '\\' + type)
    count = 1
    for file in files:
        mfcc = getMfcc(os.getcwd() + '\\' + type + "\\" + type + str(count) + ".wav")
        count = count + 1
        sound_train.append(mfcc.T)
        label = numpy.full((mfcc.shape[1], ),types.index(type), dtype=numpy.int)
        type_train.append(label)

sound_train = numpy.concatenate(sound_train)
type_train = numpy.concatenate(type_train)
print('Learning...')

svc = SVC(C=1, gamma=1e-4)
svc.fit(sound_train,type_train)
print('Learning OK!')

savename = 'model.sav'
pickle.dump(svc, open(savename, 'wb'))

# for type in types:
#     mfcc = getMfcc('/home/kouki/SoundChake/Sample/' + type + '.wav')
#     loadmodel = pickle.load(open(savename, 'rb'))
#     prediction = loadmodel.predict(mfcc.T)
#     counts = numpy.bincount(prediction)
#     result = types[numpy.argmax(counts)]
#     original_title = 'No brand girls(%s Mix)' % type
#     print("Data:" + original_title + "    Result:" + result)

for type in types:
    mfcc = getMfcc(os.getcwd() + '\\' + type + '.wav')
    prediction = svc.predict(mfcc.T)
    counts = numpy.bincount(prediction)
    result = types[numpy.argmax(counts)]
    original_title = '(%s Mix)' % type
    print("Data:" + original_title + "    Result:" + result)

