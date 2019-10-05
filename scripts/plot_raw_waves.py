# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:36:09 2019

@author: ning
"""

import os
from glob import glob

import numpy as np

from matplotlib import pyplot as plt

import librosa
from librosa import display as rosadis

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




