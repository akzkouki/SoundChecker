import scipy.io.wavfile as wav
import librosa
from sklearn.svm import SVC
import os
import numpy
import pickle
import re
import shutil

from kivy.resources import resource_add_path
import sys
if hasattr(sys, "_MEIPASS"):
    resource_add_path(sys._MEIPASS)

from kivy.app import App
from kivy.graphics import Rectangle
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
import shutil
types= ['Kick','Crash','Hihat','Snare']
savename = 'model.sav'

def getMfcc(filename):
    y, sr = librosa.load(filename)
    return librosa.feature.mfcc(y=y, sr=sr)

def check(path):
    mfcc = getMfcc(path)
    # loadmodel = pickle.load(open(os.getcwd() + "/" + savename, 'rb'))
    loadmodel = pickle.load(open(os.getcwd() + "/" + savename, 'rb'))
    prediction = loadmodel.predict(mfcc.T)
    counts = numpy.bincount(prediction)
    result = types[numpy.argmax(counts)]

    basename = os.path.basename(path)
    dirname = os.path.dirname(str(path))
    dirname = dirname.lstrip("b")
    dirname = dirname.lstrip('\'')
    print(basename)

    os.makedirs("" + str(dirname) + "/" + str(result),exist_ok=True)
    path = str(path).lstrip("b")
    path = str(path).strip("\'")
    shutil.copy(str(path), "" + str(dirname) + "/" + str(result) )

    return result
    
class SoundChecker(BoxLayout):
    log = StringProperty()
    def __init__(self):
        self.log = "Drag and drop\n"
        super(SoundChecker, self).__init__()
        self._file = Window.bind(on_dropfile=self._on_file_drop)


    def _on_file_drop(self, window, file_path):
        
        print(file_path)
        ext = os.path.splitext(str(file_path))[1][1:]
        if ext == 'wav\'' :
            ext = 'wav'
        print(ext)

        if ext == 'wav' :
            self.log += str(file_path) + " = " + str(check(file_path) + "\n")
            print(self.log)
        else:
            print("This file is not wav")
        return



class SoundCheckerApp(App):
    def build(self):
        return SoundChecker()

SoundCheckerApp().run()

    