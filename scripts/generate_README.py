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

from models import CNN_path,RNN_path

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
