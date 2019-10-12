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

# define the models

## CNN
class CNN_path(nn.Module):
    def __init__(self,
                 device,
                 batch_size = 1):
        super(CNN_path,self).__init__()
        
        self.device         = device
        self.batch_size     = batch_size
        self.base_model     = models.mobilenet_v2(pretrained = False,).features.to(self.device)
        self.pooling        = nn.AdaptiveAvgPool2d((1,1,)).to(self.device)
        self.activation     = nn.Sigmoid().to(self.device)
   
    def forward(self,x):
        base_output = self.base_model(x)
        base_output = self.activation(self.pooling(base_output))
        return torch.squeeze(base_output)

## CRNN
class RNN_path(nn.Module):
    def __init__(self,
                 device,
                 batch_size = 1,):
        super(RNN_path,self).__init__()
        
        self.device             = device
        self.batch_size         = batch_size
        self.conv1              = nn.Conv1d(in_channels     = 1,
                                            out_channels    = 32,
                                            kernel_size     = 1210,
                                            stride          = 12,
                                            padding_mode    = 'valid',
                                            ).to(self.device)
        self.activation         = nn.SELU(inplace           = True
                                          ).to(self.device)
        self.conv2              = nn.Conv1d(in_channels     = 32,
                                            out_channels    = 16,
                                            kernel_size     = 1210,
                                            stride          = 12,
                                            padding_mode    = 'valid',
                                            ).to(self.device)
        self.rnn                = nn.GRU(input_size         = 16,
                                         hidden_size        = 1,
                                         num_layers         = 1,
                                         batch_first        = True,
                                         ).to(self.device)
        self.output_activation  = nn.Sigmoid(
                                            ).to(self.device)
    
    def forward(self,x):
        x       = torch.reshape(x,(self.batch_size,1,-1))
        conv1   = self.activation(self.conv1(x))
        conv2   = self.activation(self.conv2(conv1))
        rnn,_   = self.rnn(conv2.permute(0,2,1)) # don't care about the hidden states
        output  = self.output_activation(rnn)
        return torch.squeeze(output)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import seaborn as sns
    from scipy.spatial import distance
    
    sns.set_style('white')
    sns.set_context('poster')
    
    # DEFINE MAIN VARIABLES & DIRECTORIES
    spectrograms_dir    = '../data/spectrograms'
    same_length_dir     = '../data/same_length'
    
    experiment = 'train_one_by_one'
    
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
    batch_size      = 5
    
    spec_datagen    = DataLoader(custimizedDataset([spectrograms_dir,same_length_dir]),batch_size = batch_size,shuffle = False,)
    
    computer_vision_net     = CNN_path(device = device)
    sound_net               = RNN_path(device = device,batch_size = batch_size)
    
    CVN = []
    RSN = []
    for ii,batch in tqdm(enumerate(spec_datagen)):
        with torch.no_grad():
            input_spectrogram   = Variable(batch[0]).to(device)
            input_soundwave     = Variable(batch[1]).to(device)
            
            output_CVN  = computer_vision_net(input_spectrogram)
            output_SN   = sound_net(input_soundwave)
            
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
            
            output_CVN  = computer_vision_net(input_spectrogram)
            output_SN   = sound_net(input_soundwave)
            
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
