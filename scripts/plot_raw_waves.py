# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:36:09 2019

@author: ning
"""

import os
from glob import glob

from scipy.io import wavfile
from matplotlib import pyplot as plt

working_dir = '../data'
working_data = glob(os.path.join(working_dir,'*','*.wav'))

figure_dir = '../figure/raw_wave'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

for f in working_data:
    sub_folder = f.split('\\')[-2]
    if not os.path.exists(os.path.join(figure_dir,sub_folder)):
        os.mkdir(os.path.join(figure_dir,sub_folder))
        
    fs,data = wavfile.read(f)
    print(fs,data.shape,f)
    