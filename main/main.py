import librosa
import numpy as np

from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils

import os
import re
import shutil

from kivy.app import App
from kivy.graphics import Rectangle
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
import shutil

from kivy.resources import resource_add_path

import sys

if hasattr(sys, "_MEIPASS"):
    resource_add_path(sys._MEIPASS)


types= ['Kick','Crash','Hihat','Snare']

def getMfcc(filename):
    n_mfcc = 20
    genre_x = np.zeros((0, n_mfcc))
    print(filename)
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis = 1)
    genre_x = np.vstack((genre_x, mean))
    return genre_x

def check(path):
    mfcc = getMfcc(path)

    model = model_from_json(open('model.json').read())
    model.load_weights('weights.h5')
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
                
    predict_classes = model.predict_classes(mfcc, batch_size=32)
    print(predict_classes)
    result = types[predict_classes.tolist()[0]]

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
