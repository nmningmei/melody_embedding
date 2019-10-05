# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:36:09 2019

@author: ning
"""

import os
from glob import glob

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

import librosa
from librosa import display as rosadis

sns.set_style('white')
sns.set_context('poster')

working_dir = '../data'
working_data = glob(os.path.join(working_dir,'*','*.wav'))

figure_dir = '../figure/raw_wave'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

for f in working_data:
    sub_folder = f.split('\\')[-2]
    f_name = f.split('\\')[-1]
    if not os.path.exists(os.path.join(figure_dir,sub_folder)):
        os.mkdir(os.path.join(figure_dir,sub_folder))
    if not os.path.exists(os.path.join(figure_dir,sub_folder,f_name.replace('wav','png'))):
        data,sampling_rate = librosa.load(f)
        print(data.shape,sampling_rate,f)
        
        times = np.arange(data.shape[0]) / sampling_rate
    
        fig,ax = plt.subplots(figsize = (10,6))
        ax.plot(times,data,)
        ax.set(xlabel = 'time',
               title = f'{f_name}')
        
        fig.savefig(os.path.join(
                figure_dir,
                sub_folder,
                f_name.replace('wav','png')),
                dpi = 400,
                bbox_inches = 'tight')
        plt.close('all')

figure_dir = '../figure/spectrogram'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

hop_length = 512
n_fft = 2048
n_mels = 128

for f in working_data:
    sub_folder = f.split('\\')[-2]
    f_name = f.split('\\')[-1]
    if not os.path.exists(os.path.join(figure_dir,sub_folder)):
        os.mkdir(os.path.join(figure_dir,sub_folder))
    if not os.path.exists(os.path.join(figure_dir,sub_folder,f_name.replace('wav','png'))):
        data,sampling_rate = librosa.load(f)
        D = np.abs(librosa.stft(data,n_fft = n_fft,
                                hop_length = hop_length,))
        DB = librosa.amplitude_to_db(D, ref = np.max)
        
        S = librosa.feature.melspectrogram(data, sr = sampling_rate,
                                           n_fft = n_fft,
                                           hop_length = hop_length,
                                           n_mels = n_mels,
                                           )
        S_DB = librosa.power_to_db(S, ref = np.max)
        
        fig = plt.figure(figsize = (20,20),)
        plt.subplot(211)
        rosadis.specshow(DB,sr = sampling_rate,
                              hop_length = hop_length,
                              x_axis = 'time',
                              y_axis = 'log',
                              )
        plt.colorbar()
        plt.title('Spectrogram')
        plt.subplot(212)
        ax = rosadis.specshow(S_DB,sr = sampling_rate,
                              hop_length = hop_length,
                              x_axis = 'time',
                              y_axis = 'mel',
                              )
        plt.colorbar()
        plt.title('Mel Spetrogram')
        fig.savefig(os.path.join(
                figure_dir,
                sub_folder,
                f_name.replace('wav','png')),
                dpi = 400,
                bbox_inches = 'tight')
        plt.close('all')
        
    
    



