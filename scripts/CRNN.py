from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array)
from tensorflow.keras import (layers,
                              models,
                              initializers,
                              activations,
                              applications)
import tensorflow as tf
import os
import glob
import numpy as np
from scipy.signal import resample


# DEFINE MAIN VARIABLES & DIRECTORIES

spectrograms_dir = 'data/spectrograms/'

same_length_dir = 'data/same_length/'


# LOAD SPECTROGRAMS AND WAVEFORMS FILES

spectrograms = np.sort(glob(os.path.join(spectrograms_dir, '*.png')))

audios = np.sort(glob(os.path.join(same_length_dir, '*.npy')))

for a,b in zip(spectrograms,audios):

    print(a,b)

    img = img_to_array(load_img(a, target_size = (224,224,3))) / 255.

    wave_form = np.load(b).astype("float32")

    # CNN
#    inputs = layers.Input(shape = (224,224,3),
#                               batch_size = 1,
#                               name = "image_input")
#    conv1 = layers.Conv2D(filters = 32,
#                          kernel_size = (5,5),
#                          strides = 1,
#                          padding = "valid",
#                          activation = activations.selu,
#                          kernel_initializer = "lecun_normal",
#                          data_format = "channels_last",
#                          name = "conv1_1")(inputs)
#    conv1 = layers.Conv2D(filters = 32,
#                          kernel_size = (5,5),
#                          strides = 1,
#                          padding = "valid",
#                          activation = activations.selu,
#                          kernel_initializer = "lecun_normal",
#                          data_format = "channels_last",
#                          name = "conv1_2")(conv1)
#    conv1 = layers.AveragePooling2D(pool_size = (3,3,),
#                                    strides = 1,
#                                    padding = "valid",
#                                    data_format = "channels_last",
#                                    name = "pool1")(conv1)
#    print(conv1.shape)
    
    base_model = applications.mobilenet_v2.MobileNetV2(weights = "imagenet",
                                                       input_shape = (224,224,3),
                                                       pooling = "avg",
                                                       include_top = False)
    for layer in base_model.layers:

        layer.trainable = False

    inputs = base_model.input
    
    a_embedding = base_model.predict(img[np.newaxis,])
    
    # CRNN

    inputs = layers.Input(shape = (wave_form.shape[0],),
                         batch_size = 1,
                         name = "audio_input")

    reshape = layers.Reshape((-1,1),name = "reshape")(inputs)

    kernel_size = 1210

    strides = 12

    conv1 = layers.Conv1D(filters = 16,
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
#    print(conv1.shape)
   
    rnn = layers.GRU(units = 1,
                     return_sequences = True,
                     name = "rnn")(conv1)

    rnn_reshape = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(rnn)
    
    print(rnn_reshape.shape)
    
    
    
    
    
    
    
    