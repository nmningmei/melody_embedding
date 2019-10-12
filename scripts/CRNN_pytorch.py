#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:27:55 2019

@author: nmei

This script a full workflow under the pytorch framework

1. train the computer vision net with sound net is frozen
2. train the sound net with computer vision net is frozen
3. measure the difference between the 2 outputs by p2 norm

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

def train_computer_vision_net_loop(computer_vision_net,
               sound_net,
               loss_func,
               optimizers,
               dataloader,
               idx_epoch    = 1,
               l1           = 0.,
               l2           = 0.01,
               device       = 'cpu',
               epsilon      = 1e-12):
    
    computer_vision_net_loss    = 0.
    
    for ii,(batch) in enumerate(dataloader):
        if ii + 1 <= len(dataloader):
            input_spectrogram   = Variable(batch[0]).to(device)
            input_soundwave     = Variable(batch[1]).to(device)
            
            # update loop for CVN:
            optimizers[0].zero_grad() # Important!!
            output_CVN      = computer_vision_net(input_spectrogram)
            with torch.no_grad(): # freeze the other network
                output_SN   = sound_net(input_soundwave)
            loss_CVN        = loss_func(output_CVN,output_SN) # loss
            # add regularization to the weights
            selected_params = torch.cat([x.view(-1) for x in computer_vision_net.parameters()])
            loss_CVN        += l1 * torch.norm(selected_params,1) + l2 * torch.norm(selected_params,2) + epsilon
            loss_CVN.backward() # autograd
            optimizers[0].step() # modify the weights
            computer_vision_net_loss += loss_CVN
            
            print(f'epoch {idx_epoch}-{ii + 1:3d}/{100*(ii+1)/ len(dataloader):07.3f}%,CV_loss = {computer_vision_net_loss/(ii + 1):.5f}',)
    return computer_vision_net_loss/(ii + 1)

def train_sound_net_loop(computer_vision_net,
               sound_net,
               loss_func,
               optimizers,
               dataloader,
               idx_epoch    = 1,
               l1           = 0.,
               l2           = 0.01,
               device       = 'cpu',
               epsilon      = 1e-12):
    
    sound_net_loss              = 0.
    
    for ii,(batch) in enumerate(dataloader):
        if ii + 1 <= len(dataloader):
            input_spectrogram   = Variable(batch[0]).to(device)
            input_soundwave     = Variable(batch[1]).to(device)
            
            # update loop for SN:
            optimizers[1].zero_grad() # Important!!
            output_SN       = sound_net(input_soundwave)
            with torch.no_grad():
                output_CVN  = computer_vision_net(input_spectrogram)
            loss_SN         = loss_func(output_SN,output_CVN)
            selected_params = torch.cat([x.view(-1) for x in sound_net.parameters()]) # L2 
            loss_SN         += l1 * torch.norm(selected_params,1) + l2 * torch.norm(selected_params,2) + epsilon
            loss_SN.backward()
            optimizers[1].step()
            sound_net_loss  += loss_SN
            
            print(f'epoch {idx_epoch}-{ii + 1:3d}/{100*(ii+1)/ len(dataloader):07.3f}%,SN_loss = {sound_net_loss/(ii + 1):.5f}',)
    return sound_net_loss/(ii + 1)#computer_vision_net_loss/(ii + 1),CV_loss = {computer_vision_net_loss/(ii + 1):.5f},

def validation_loop(computer_vision_net,sound_net,dataloader,device,idx_epoch = 1):
    with torch.no_grad():
        temp = []
        for ii,(batch) in enumerate(dataloader):
            if ii + 1 <= len(dataloader):
                input_spectrogram   = Variable(batch[0]).to(device)
                input_soundwave     = Variable(batch[1]).to(device)
                
                output_CVN  = computer_vision_net(input_spectrogram)
                output_SN   = sound_net(input_soundwave)
                # compute the difference btw the 2 embeddings
                distance    = nn.PairwiseDistance()
                output      = distance(output_CVN,output_SN)
                temp.append(output)
    temp = torch.reshape(torch.cat(temp),(-1,1))
    return temp

if __name__ == '__main__':
    # DEFINE MAIN VARIABLES & DIRECTORIES
    spectrograms_dir    = '../data/spectrograms'
    same_length_dir     = '../data/same_length'
    
    experiment = 'train_one_by_one_BCEweightloss'
    
    # model weights
    weight_dir = '../weights/{}'.format(experiment)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    # results
    saving_dir = '../results/{}'.format(experiment)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    
    # LOAD SPECTROGRAMS AND WAVEFORMS FILES
    spectrograms        = np.sort(glob(os.path.join(spectrograms_dir, '*.png')))
    audios              = np.sort(glob(os.path.join(same_length_dir,  '*.npy')))
    
    # initalize the GPU with specific state: for reproducibility
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate   = 1e-4
    n_epochs        = 1000
    batch_size      = 5
    patience        = 10
    
    spec_datagen    = DataLoader(custimizedDataset([spectrograms_dir,same_length_dir]),batch_size = batch_size,shuffle = True,)
    
    computer_vision_net     = CNN_path(device = device)
    sound_net               = RNN_path(device = device,batch_size = batch_size)
    
    #Loss function
    loss_func   = nn.BCEWithLogitsLoss()
    
    #Optimizers
    optimizer1  = optim.Adam(computer_vision_net.parameters(),  lr = learning_rate)#,weight_decay = 1e-7)
    optimizer2  = optim.Adam(sound_net.parameters(),            lr = learning_rate)#,weight_decay = 1e-7)
    
    if os.path.exists(os.path.join(saving_dir,'scores.csv')):
#        computer_vision_net.load_state_dict(torch.load(os.path.join(weight_dir,'CNN_path.pth')))
        sound_net.load_state_dict(          torch.load(os.path.join(weight_dir,'RNN_path.pth')))
        results     = pd.read_csv(os.path.join(saving_dir,'scores.csv'))
        results     = {col_name:list(results[col_name].values) for col_name in results.columns}
        best_score  = torch.tensor(results['distance'].min(),dtype = torch.float64)
    else:
        print('initialize')
        results = dict(
                train_loss      = [],
                train_model     = [],
                distance        = [],
                epochs          = [],
                learning_rate   = [],
                )
        best_score = torch.from_numpy(np.array(np.inf))
        
        early_stop = 0
        for idx_epoch in range(n_epochs):
            torch.cuda.empty_cache()
            if os.path.exists(os.path.join(weight_dir,'CNN_path.pth')):
                print('load best weights')
                computer_vision_net.load_state_dict(torch.load(os.path.join(weight_dir,'CNN_path.pth')))
#                sound_net.load_state_dict(          torch.load(os.path.join(weight_dir,'RNN_path.pth')))
            print('training ...')
            train_losse_CV = train_computer_vision_net_loop(
                                      computer_vision_net,
                                      sound_net,
                                      loss_func     = loss_func,
                                      optimizers    = [optimizer1,optimizer2],
                                      dataloader    = spec_datagen,
                                      idx_epoch     = idx_epoch,
                                      device        = device,)
            print('validating ...')
            distances = validation_loop(computer_vision_net,
                                        sound_net,
                                        dataloader  = spec_datagen,
                                        device      = device,
                                        idx_epoch   = idx_epoch,)
            print(f'epoch {idx_epoch:3d},distance between the 2 outputs = {distances.mean():.5f}')
            
            print('determine early stop and save weights')
            if distances.mean().cpu().clone().detach().type(torch.float64) < best_score:
                best_score = distances.mean().cpu().clone().detach().type(torch.float64)
                print('saving weights of the best models\n')
                torch.save(computer_vision_net.state_dict(),os.path.join(weight_dir,'CNN_path.pth'))
#                torch.save(sound_net.state_dict(),          os.path.join(weight_dir,'RNN_path.pth'))
                early_stop = 0
            else:
                print(f'nah, I have seen better + {early_stop + 1}\n')
                early_stop += 1
            
            results['train_loss'].append(train_losse_CV.detach().cpu().numpy())
            results['train_model'].append("CNN")
            results['distance'     ].append(distances.mean().detach().cpu().numpy())
            results['epochs'       ].append(idx_epoch + 1)
            results['learning_rate'].append(learning_rate)
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(saving_dir,'scores.csv'),index = False)
            if early_stop >= patience:
                stp = idx_epoch + 1
                break
        early_stop = 0
        for idx_epoch in range(n_epochs):
            torch.cuda.empty_cache()
            if os.path.exists(os.path.join(weight_dir,'RNN_path.pth')):
                print('load best weights')
#                computer_vision_net.load_state_dict(torch.load(os.path.join(weight_dir,'CNN_path.pth')))
                sound_net.load_state_dict(          torch.load(os.path.join(weight_dir,'RNN_path.pth')))
            train_losses_SN = train_sound_net_loop(
                                      computer_vision_net,
                                      sound_net,
                                      loss_func     = loss_func,
                                      optimizers    = [optimizer1,optimizer2],
                                      dataloader    = spec_datagen,
                                      idx_epoch     = idx_epoch + stp,
                                      device        = device,)
            print('validating ...')
            distances = validation_loop(computer_vision_net,
                                        sound_net,
                                        dataloader  = spec_datagen,
                                        device      = device,
                                        idx_epoch   = idx_epoch,)
            print(f'epoch {idx_epoch + stp:3d},distance between the 2 outputs = {distances.mean():.5f}')
            
            print('determine early stop and save weights')
            if distances.mean().cpu().clone().detach().type(torch.float64) < best_score:
                best_score = distances.mean().cpu().clone().detach().type(torch.float64)
                print('saving weights of the best models\n')
#                torch.save(computer_vision_net.state_dict(),os.path.join(weight_dir,'CNN_path.pth'))
                torch.save(sound_net.state_dict(),          os.path.join(weight_dir,'RNN_path.pth'))
                early_stop = 0
            else:
                print(f'nah, I have seen better + {early_stop + 1}\n')
                early_stop += 1
            results['train_loss'].append(train_losses_SN.detach().cpu().numpy())
            results['train_model'].append("CRNN")
            results['distance'     ].append(distances.mean().detach().cpu().numpy())
            results['epochs'       ].append(idx_epoch + 1 + stp)
            results['learning_rate'].append(learning_rate)
            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(saving_dir,'scores.csv'),index = False)
            if early_stop >= patience:
                break