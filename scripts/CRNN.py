#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 10:09:21 2019

@author: nmei,pedro
"""

import os
import gc
import numpy      as np
import tensorflow as tf
if not tf.executing_eagerly():
    tf.executing_eagerly()
from glob                                 import glob
from tqdm                                 import tqdm

from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array)
from tensorflow.keras                     import (layers,
                                                  models,
                                                  activations,
                                                  applications,
                                                  optimizers,
                                                  losses)
from scipy.spatial                        import distance
from sklearn.utils                        import shuffle

# DEFINE MAIN VARIABLES & DIRECTORIES
spectrograms_dir    = '../data/spectrograms/'
same_length_dir     = '../data/same_length/'

# LOAD SPECTROGRAMS AND WAVEFORMS FILES
spectrograms        = np.sort(glob(os.path.join(spectrograms_dir, '*.png')))
audios              = np.sort(glob(os.path.join(same_length_dir, '*.npy')))

# CNN
tf.keras.backend.clear_session()
base_model          = applications.mobilenet_v2.MobileNetV2(weights         = None,
                                                            input_shape     = (224,224,3),
                                                            pooling         = "avg",
                                                            include_top     = False)
#    for layer in base_model.layers:
#        layer.trainable = False

inputs_spec         = base_model.input
outputs_spec        = base_model.output
outputs_spec        = layers.Activation('sigmoid',name = 'output_sig')(outputs_spec)
spec_model          = models.Model(inputs_spec,outputs_spec,name = 'spectro')

# CRNN
inputs              = layers.Input(shape        = (np.load(audios[0]).astype("float32").shape[0],),
                                   batch_size   = 1,
                                   name         = "audio_input")
reshape = layers.Reshape((-1,1),name = "reshape")(inputs)
kernel_size = 1210
strides = 12
conv1 = layers.Conv1D(filters = 32,
                      kernel_size = kernel_size,
                      strides = strides,
                      padding = "valid",
                      data_format = "channels_last",
                      activation = activations.selu,
                      kernel_initializer = "lecun_normal",
                      name = "conv1_1")(reshape)
conv1 = layers.Conv1D(filters = 16,
                      kernel_size = kernel_size,
                      strides = strides,
                      padding = "valid",
                      data_format = "channels_last",
                      activation = activations.selu,
                      kernel_initializer = "lecun_normal",
                      name = "conv1_2")(conv1)
rnn = layers.GRU(units = 1,
                 activation = 'sigmoid',
                 return_sequences = True,
                 name = "rnn")(conv1)
rnn_reshape = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(rnn)

wave_model = models.Model(inputs,rnn_reshape,name = 'wave')

loss_func = losses.CosineSimilarity()

spec_optimizer = optimizers.Adam(lr = 1e-4,)
wave_optimizer = optimizers.Adam(lr = 1e-4,)

@tf.function
def train_step(img,wave_form):
    with tf.GradientTape() as spec_tape, tf.GradientTape() as wave_tape:
        spec_embedding = spec_model(img[np.newaxis,])
        wave_embedding = wave_model(wave_form[np.newaxis,])
        
        spec_loss = loss_func(wave_embedding,spec_embedding)
        wave_loss = loss_func(spec_embedding,wave_embedding)
        
        spec_gradient = spec_tape.gradient(spec_loss,spec_model.trainable_variables)
        wave_gradient = wave_tape.gradient(wave_loss,wave_model.trainable_variables)
        
        spec_optimizer.apply_gradients(zip(spec_gradient,spec_model.trainable_variables))
        wave_optimizer.apply_gradients(zip(wave_gradient,wave_model.trainable_variables))

n_epochs = 100

for epoch in range(n_epochs):
    spectrograms, audios = shuffle(spectrograms,audios)
    total_loss = 0
    for step,(a,b) in enumerate(zip(spectrograms,audios)):
    
        print(a,b)
    
        img = img_to_array(load_img(a, target_size = (224,224,3))) / 255.
    
        wave_form = np.load(b).astype("float32")
    
        train_step(img,wave_form)
        
        em1 = spec_model(img[np.newaxis])
        em2 = wave_model(wave_form[np.newaxis,])
        print(em1.min(),em1.max(),em2.min(),em2.max())
        total_loss += loss_func(em1,em2).numpy()
        print(f"loss = {total_loss / (step + 1):.4f}")
    
    embeddings1 = np.array([spec_model((img_to_array(load_img(img, target_size = (224,224,3))) / 255.)[np.newaxis,]).numpy() for img in tqdm(spectrograms,desc = 'spectrograms')])
    
    gc.collect()
    embeddings2 = np.array([wave_model(np.load(wave_form).astype("float32")[np.newaxis,]).numpy() for wave_form in tqdm(audios,desc = 'audios')])
    
    embeddings1 = np.squeeze(embeddings1,axis = 1)
    embeddings2 = np.squeeze(embeddings2,axis = 1)
    
    dissimilarity = []
    for a,b in zip(embeddings1,embeddings2):
        dis = distance.cdist(a.reshape(1,-1),
                             b.reshape(1,-1))
#        print(dis)
        dissimilarity.append(dis.flatten()[0])
    dissimilarity = np.array(dissimilarity)
    print(f"epoch {epoch + 1} {dissimilarity.mean():.4f}+/-{dissimilarity.std():.4f}")
    
    
    
    
