# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:30:41 2022

@author: abdal
"""

from flask import Flask, request, jsonify, render_template,redirect
from tensorflow.python.keras.models import load_model
from scipy.signal import spectrogram
import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import sys
import numpy


app = Flask(__name__)

model=load_model('model.h5')

numpy.set_printoptions(threshold=sys.maxsize)
max_pad_len = 432
rows = 40
columns = 432
channels = 1

def mfcc_feature(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = np.abs(max_pad_len - mfccs.shape[1])
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs.reshape(1,rows , columns , channels)

@app.route('/sound_class', methods=['POST','GET'])
def foot_prediction():
    audio_file = request.files["file"]
    dic = {0:'fan this car has a fan malfunction (abnormal)', 1:'fan normal', 2:'pump this car has a pump malfunction(abnormal)', 3:'pump normal', 4:'slider this car has a slider malfunction (abnormal)' , 5:'slider normal', 6:'valve this car has a valve malfunction (abnormal)'}
    preprocessed = mfcc_feature(audio_file)
    predicted = model.predict(preprocessed)
    result = np.argmax(predicted)
    return jsonify(dic[result])


  

if __name__ == "__main__":
    app.run()
    
    
    



    
