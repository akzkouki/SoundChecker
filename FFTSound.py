#coding:utf-8
import wave
import numpy
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
import os

def wave_load(filename):
    # open wave file
    wf = wave.open(filename,'r')
    channels = wf.getnchannels()

    # load wave data
    chunk_size = wf.getnframes()
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)   
    data = numpy.frombuffer(data,'int16')
    data = data / amp                  
    data = data[::channels]
   
    return data


def fft_load(filename,size,start,end,name):
    st = 10000  
    hammingWindow = numpy.hamming(size) 
    fs = 44100
    d = 1.0 / fs
    freqList = numpy.fft.fftfreq(size, d)
    
    n = random.randint(start,end)
    wave = wave_load(filename)
    windowedData = hammingWindow * wave[st:st+size] 
    data = numpy.fft.fft(windowedData)
    data = data / max(abs(data))

    draw_graph(data,freqList,fs,name)

    return data

def draw_graph(data,freqList,fs,sound):
    plt.plot(freqList,abs(data))
    plt.axis([0,fs/2,0,1])
    plt.title(sound)
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("amplitude spectrum")
    plt.show()

# print(fft_load("Kick",1024,0,1000))


types= ['Kick','Crash','Hihat','Snare']
# types= ['Kick','Hihat','Snare']
sound_train = []
type_train = []
for type in types:
    print('Reading data of %s...' % type)
    files = os.listdir(os.getcwd() + '\\' + type)
    count = 1
    for file in files:
        fft_data = fft_load(os.getcwd() + '\\' + type + "\\" + type + str(count) + ".wav",1024,0,10000,type)
        count = count + 1
        # sound_train.append(fft_data.T)
        # label = numpy.full((fft_data.shape[1], ),types.index(type), dtype=numpy.int)
        # type_train.append(label)

sound_train = numpy.concatenate(sound_train)
type_train = numpy.concatenate(type_train)
print('Learning...')

svc = SVC(C=1, gamma=1e-4)
svc.fit(sound_train,type_train)
print('Learning OK!')

# savename = 'model.sav'
# pickle.dump(svc, open(savename, 'wb'))

# for type in types:
#     mfcc = getMfcc('/home/kouki/SoundChake/Sample/' + type + '.wav')
#     loadmodel = pickle.load(open(savename, 'rb'))
#     prediction = loadmodel.predict(mfcc.T)
#     counts = numpy.bincount(prediction)
#     result = types[numpy.argmax(counts)]
#     original_title = 'No brand girls(%s Mix)' % type
#     print("Data:" + original_title + "    Result:" + result)

# for type in types:
#     mfcc = getMfcc(os.getcwd() + '\\' + type + '.wav')
#     prediction = svc.predict(mfcc.T)
#     counts = numpy.bincount(prediction)
#     result = types[numpy.argmax(counts)]
#     original_title = '(%s Mix)' % type
#     print("Data:" + original_title + "    Result:" + result)

