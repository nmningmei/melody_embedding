#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:43:19 2019

@author: nmei
"""

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
        self.pooling        = nn.AdaptiveAvgPool2d((1,1,)).to(self.device)
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
        self.embed_activation = nn.Sigmoid(
                                        ).to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        base_output = self.pooling(self.base_model(x))
        embedding = torch.squeeze(self.embed_activation(base_output))
        
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
                                            out_channels    = 100,
                                            kernel_size     = 5,
                                            stride          = 5,
                                            padding_mode    = 'valid',
                                            ).to(self.device)
        self.activation         = nn.SELU(inplace           = True
                                          ).to(self.device)
        self.avgpooling         = nn.MaxPool1d(kernel_size = 5,
                                               stride = 5).to(self.device)
        self.adaptivepooling    = nn.AdaptiveAvgPool1d(1).to(self.device)
        self.rnn                = nn.LSTM(input_size        = 100,
                                          hidden_size       = 1,
                                          num_layers        = 1,
                                          batch_first       = True,
                                          bidirectional     = True,
                                          ).to(self.device)
        self.embedding_layer    = nn.Linear(in_features     = 8000,
                                            out_features    = 1280,).to(self.device)
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
        self.embed_activation = nn.Sigmoid(
                                        ).to(self.device)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        x       = torch.reshape(x,(self.batch_size,1,-1))
        conv1   = self.conv1(x)
        conv1   = self.avgpooling(conv1)
        conv1   = self.activation(conv1).permute(0,2,1)
        embedding,_   = self.rnn(conv1) # don't care about the hidden states
        embedding = self.adaptivepooling(embedding)
        embedding = self.output_activation(embedding)
        embedding = torch.squeeze(embedding)
        embedding = self.embed_activation(self.embedding_layer(embedding))
        
        mu = self.output_activation(self.mu_node(embedding))
        logvar = self.output_activation(self.logvar_node(embedding))
        
        z = self.node_activation(self.reparameterize(mu,logvar))
        
        return z,embedding