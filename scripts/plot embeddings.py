#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:54:50 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
from copy import deepcopy

import torch
from torchvision            import models
from torchvision.transforms import functional as TF
from torch                  import optim
from torch.utils.data       import Dataset,DataLoader
from torch.autograd         import Variable
from torch                  import nn

import numpy  as np
import pandas as pd
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array)

from models import CNN_path,RNN_path

# define the dataflow for training -- better than writing the for-loop myself
class custimizedDataset(Dataset):
    def __init__(self,data_root,device = 'cpu'):
        self.samples    = []
        self.device     = device
        for image in glob(os.path.join(data_root[0],'*.png')):
            # find matched wave form file
            wave = image.replace('.png','.npy').replace(image.split("/")[-2],data_root[1].split("/")[-1])
            self.samples.append([image,wave])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        # load images
        img                     = img_to_array(load_img(self.samples[idx][0], target_size = (224,224,3)))
        img_tensor              = TF.to_tensor(img)
        normalize_image_tensor  = TF.normalize(img_tensor,
                                               mean     = [0.485, 0.456, 0.406],
                                               std      = [0.229, 0.224, 0.225],
                                               inplace  = True,
                                               ).to(device)
        # load sound waves
        wave_form               = np.load(self.samples[idx][1]).astype('float32')
        wave_tensor             = torch.from_numpy(wave_form).to(device)
        return (normalize_image_tensor,wave_tensor)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import seaborn as sns
    from scipy.spatial import distance
    
    sns.set_style('white')
    sns.set_context('poster')
    
    # DEFINE MAIN VARIABLES & DIRECTORIES
    spectrograms_dir    = '../data/spectrograms'
    same_length_dir     = '../data/same_length'
    
    experiment = 'train_both_with_reparameterize_trick' # train_one_by_one, train_both_with_reparameterize_trick
    
    # model weights
    weight_dir = '../weights/{}'.format(experiment)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    # results
    saving_dir = '../results/{}'.format(experiment)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    
    # figures
    figure_dir = '../figures/{}'.format(experiment)
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    
    # LOAD SPECTROGRAMS AND WAVEFORMS FILES
    spectrograms        = np.sort(glob(os.path.join(spectrograms_dir, '*.png')))
    audios              = np.sort(glob(os.path.join(same_length_dir, '*.npy')))
    
    # initalize the GPU with specific state: for reproducibility
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate   = 1e-4
    n_epochs        = 1000
    batch_size      = 10
    
    spec_datagen    = DataLoader(custimizedDataset([spectrograms_dir,same_length_dir]),batch_size = batch_size,shuffle = False,)
    
    computer_vision_net     = CNN_path(device = device)
    sound_net               = RNN_path(device = device,batch_size = batch_size)
    
    CVN = []
    RSN = []
    for ii,batch in tqdm(enumerate(spec_datagen)):
        with torch.no_grad():
            input_spectrogram   = Variable(batch[0]).to(device)
            input_soundwave     = Variable(batch[1]).to(device)
            
            input_soundwave     = (input_soundwave - input_soundwave.mean(1).view(-1,1)) / (input_soundwave.std(1).view(-1,1))
            
            _,output_CVN  = computer_vision_net(input_spectrogram)
            _,output_SN   = sound_net(input_soundwave)
            
            CVN.append(output_CVN)
            RSN.append(output_SN)
    CVN = torch.cat(CVN)
    RSN = torch.cat(RSN)
    
    CVN = CVN.detach().cpu().numpy()
    RSN = RSN.detach().cpu().numpy()
    
    data = np.concatenate([CVN,RSN])
    labels = np.concatenate([[item.split('/')[-1].split('.')[0] + '_CNN' for item in spectrograms],
                             [item.split('/')[-1].split('.')[0] + '_RNN' for item in audios      ]
                             ])
    
    df = pd.DataFrame(data.T,columns = labels)
    asdf
    df_plot = distance.squareform(distance.pdist(df.values.T - df.values.T.mean(1).reshape(-1,1),metric = 'cosine'))
    df_plot = pd.DataFrame(df_plot,columns = labels, index = labels)
    
    fig,ax = plt.subplots(figsize = (50,45))
    ax = sns.heatmap(df_plot,
                     xticklabels=True,
                     yticklabels=True,
                     ax = ax,)
    fig.savefig(os.path.join(figure_dir,'RDM_embeddings (before).png'),
#                dpi = 500,
                bbox_inches = 'tight')
    plt.close('all')
    
    
    computer_vision_net.load_state_dict(torch.load(os.path.join(weight_dir,'CNN_path.pth')))
    sound_net.load_state_dict(          torch.load(os.path.join(weight_dir,'RNN_path.pth')))
    
    CVN = []
    RSN = []
    for ii,batch in tqdm(enumerate(spec_datagen)):
        with torch.no_grad():
            input_spectrogram   = Variable(batch[0]).to(device)
            input_soundwave     = Variable(batch[1]).to(device)
            
            _,output_CVN  = computer_vision_net(input_spectrogram)
            _,output_SN   = sound_net(input_soundwave)
            
            CVN.append(output_CVN)
            RSN.append(output_SN)
    CVN = torch.cat(CVN)
    RSN = torch.cat(RSN)
    
    CVN = CVN.detach().cpu().numpy()
    RSN = RSN.detach().cpu().numpy()
    
    data = np.concatenate([CVN,RSN])
    labels = np.concatenate([[item.split('/')[-1].split('.')[0] + '_CNN' for item in spectrograms],
                             [item.split('/')[-1].split('.')[0] + '_RNN' for item in audios      ]
                             ])
    
    df = pd.DataFrame(data.T,columns = labels)
    
    df_plot = distance.squareform(distance.pdist(df.values.T - df.values.T.mean(1).reshape(-1,1),metric = 'cosine'))
    df_plot = pd.DataFrame(df_plot,columns = labels, index = labels)
    
    fig,ax = plt.subplots(figsize = (50,45))
    ax = sns.heatmap(df_plot,
                     xticklabels=True,
                     yticklabels=True,
                     ax = ax,)
    fig.savefig(os.path.join(figure_dir,'RDM_embeddings (after).png'),
#                dpi = 500,
                bbox_inches = 'tight')
    plt.close('all')
