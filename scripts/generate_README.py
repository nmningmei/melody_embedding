#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:40:48 2019

@author: nmei
"""

import os
import pandas as pd

import torch
from torchvision            import models
from torchvision.transforms import functional as TF
from torch                  import optim
from torch.utils.data       import Dataset,DataLoader
from torch.autograd         import Variable
from torch                  import nn

## CNN
class CNN_path(nn.Module):
    def __init__(self,
                 device,
                 batch_size = 1):
        super(CNN_path,self).__init__()
        
        self.device         = device
        self.batch_size     = batch_size
        self.base_model     = models.mobilenet_v2(pretrained = True,).features.to(self.device)
        self.pooling        = nn.AdaptiveMaxPool2d((1,1,)).to(self.device)
        self.activation     = nn.SELU(
                                      ).to(self.device)
        self.mu_node        = nn.Linear(in_features     = 1280,
                                        out_features    = 300,
                                        ).to(self.device)
        self.logvar_node    = nn.Linear(in_features     = 1280,
                                        out_features    = 300,
                                        ).to(self.device)
        self.node_activation= nn.Sigmoid(
                                        ).to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        base_output = self.base_model(x)
        embedding = torch.squeeze(self.activation(self.pooling(base_output)))
        
        mu = self.activation(self.mu_node(embedding))
        logvar = self.activation(self.logvar_node(embedding))
        
        z = self.node_activation(self.reparameterize(mu,logvar))
        
        return z,embedding

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
        self.pooling            = nn.AdaptiveMaxPool1d(1).to(self.device)
        self.rnn                = nn.GRU(input_size         = 16,
                                         hidden_size        = 1,
                                         num_layers         = 1,
                                         batch_first        = True,
                                         ).to(self.device)
        self.output_activation  = nn.SELU(
                                            ).to(self.device)
        self.mu_node            = nn.Linear(in_features     = 1280,
                                            out_features    = 300,
                                            ).to(self.device)
        self.logvar_node        = nn.Linear(in_features     = 1280,
                                            out_features    = 300,
                                            ).to(self.device)
        self.node_activation    = nn.Sigmoid(
                                            ).to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        x       = torch.reshape(x,(self.batch_size,1,-1))
        conv1   = self.activation(self.conv1(x))
        conv2   = self.activation(self.conv2(conv1)).permute(0,2,1)
#        rnn,_   = self.rnn(conv2.permute(0,2,1)) # don't care about the hidden states
        embedding  = torch.squeeze(self.pooling(self.output_activation(conv2)))
        
        mu = self.output_activation(self.mu_node(embedding))
        logvar = self.output_activation(self.logvar_node(embedding))
        
        z = self.node_activation(self.reparameterize(mu,logvar))
        
        return z,embedding


if __name__ == '__main__':
    
    experiment = 'train_one_by_one'
    weight_dir = '../weights/{}'.format(experiment)
    working_dir = '../results/{}'.format(experiment)
    results = pd.read_csv(os.path.join(working_dir,'scores.csv'))
    
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate   = 1e-4
    n_epochs        = 1000
    batch_size      = 5
    
    beginning = """
# melody_embedding

## Self-supervised Learning
[insert text]

## Goals:
- [ ] Train the 2 pathways to extract informative embeddings
- [ ] Examine the embeddings via decoding

## Convolutional Neural Network -- re-train mobileNetV2:
```
{CNN_mobilenet}
```
## Convolutional Recurrent Neural Network:
```
{CRNN}
```
"""
    
    
    computer_vision_net     = CNN_path(device = device)
    sound_net               = RNN_path(device = device,batch_size = batch_size)
    
    beginning = beginning.format(**{'best_score': results['distance'].min(),
                                    "CNN_mobilenet":computer_vision_net.forward,
                                    "CRNN":sound_net.forward})
    
    computer_vision_net.load_state_dict(torch.load(os.path.join(weight_dir,'CNN_path.pth')))
    sound_net.load_state_dict(          torch.load(os.path.join(weight_dir,'RNN_path.pth')))
    
    if os.path.exists('../README.md'):
        os.remove('../README.md')
    
    with open('../README.md','w') as f:
        f.close()
    
    with open('../README.md', 'w') as f:
        f.write(beginning)
        f.close()
