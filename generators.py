
from __future__ import absolute_import, division, print_function, unicode_literals, generators

import os
import pathlib
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from os import path
import ffmpeg

from pydub import AudioSegment
from sklearn.preprocessing import normalize, MinMaxScaler

import sys
import pickle


"""
Defines a generator to create audio sequences. Returns a tuble of (samples, targets)
data_loc is location of source data, in terms of a folder of mp3s
lookback is the length of the sequences to return
bs is batch size, how many to returns
"""

def generator(data_loc, lookback = 25, bs = 200):
    data_location = cata_loc
    data_array = np.zeros((bs,lookback))
    label_array = np.zeros((bs, 1))
    counter = 0
    while True:
        for subdir, dirs, files in os.walk(data_location):
              for file in files:
                  #print os.path.join(subdir, file)
                  filepath = subdir + os.sep + file
                  # Decodes audio
                  if filepath.endswith(".mp3"):
                        mp3_audio = AudioSegment.from_file(filepath, format="mp3")
                        # rudimentary downsample factor of 3
                        audio_array = mp3_audio.get_array_of_samples()[::4]
                        audio_array = np.array(audio_array)
                        audio_array = audio_array.astype('float32')
                        l = len(audio_array)
                        audio_array = audio_array.reshape((l,1))
                        scaler = MinMaxScaler(feature_range=(-1,1))
                        scaler.fit(audio_array)
                        audio_array = scaler.transform(audio_array)
                        audio_array = audio_array.reshape((1,l))
                        audio_array = audio_array[0]
                        audio_array = np.nan_to_num(audio_array, nan=0.0)
                        if not np.isnan(audio_array).any():
                            if not np.isinf(audio_array).any() :
    #                         https://kite.com/python/answers/how-to-check-for-nan-elements-in-a-numpy-array-in-python
                                for i in range (0,len(audio_array) - lookback - 1,100):
                                    data = audio_array[i:i+lookback]
            #                         data = data.reshape((1,lookback,1))
                                    label = audio_array[i+lookback+1:i+lookback+2]
                                    label.reshape((1,1))
                                    data_array[counter] = data
                                    label_array[counter] = label
                                    counter +=1
                                    if(counter == bs):
                                        counter = 0
                                        out_data = data_array.reshape(bs,lookback,1)
                                        out_labels = label_array.reshape(bs,1)
                                        yield (out_data,out_labels)
                            else:
                                print("inf found!")
                        else:
                            print("nan found!")

                        
