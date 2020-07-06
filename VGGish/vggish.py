"""VGGish model for Keras. A VGG-like model for audio classification
# Reference
- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)
"""

from __future__ import print_function
from __future__ import absolute_import

import sys
import os

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K

from . import vggish_params as params


# weight path
package_directory = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(package_directory,'weights/vggish_audioset_weights_without_fc2.h5')
WEIGHTS_PATH_TOP = os.path.join(package_directory,'weights/vggish_audioset_weights.h5')

def VGGish(load_weights=True, weights='audioset',
           input_tensor=None, input_shape=None,
           out_dim=None, include_top=True, pooling='flatten', n_non_trainable_conv = None ):
    '''
    An implementation of the VGGish architecture.
    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension
    :param include_top:whether to include the 3 fully-connected layers at the top of the network.
    :param pooling: pooling type over the non-top network, 'avg' or 'max' or 'flatten'
    :param n_non_trainable_conv: number of convolution layers starting from input that will not be trained, max value is 6
    :return: A Keras model instance.
    '''

    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')  
    if n_non_trainable_conv is not None:
        if weights is None:
            raise ValueError('With random initialization, all layers must be trainable')
        elif n_non_trainable_conv > 6:
            raise ValueError('Model has only 6 convolution layers')
        elif n_non_trainable_conv < 0:
            raise ValueError('Number of non trainable layers can not be negative')
    
    if out_dim is None:
        out_dim = params.EMBEDDING_SIZE

    # input shape
    if input_shape is None:
        input_shape = (params.NUM_FRAMES, params.NUM_BANDS, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor
    
    
    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1', 
               trainable=False if n_non_trainable_conv>=1 else True)(aud_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2', 
               trainable=False if n_non_trainable_conv>=2 else True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1', 
               trainable=False if n_non_trainable_conv>=3 else True)(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2', 
               trainable=False if n_non_trainable_conv>=4 else True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1', 
               trainable=False if n_non_trainable_conv>=5 else True)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2', 
               trainable=False if n_non_trainable_conv>=6 else True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)



    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = Dense(out_dim, activation='relu', name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        elif pooling == 'flatten':
            x = Flatten(name='flatten_')(x)


    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='VGGish')


    # load weights
    if load_weights:
        if weights == 'audioset':
            if include_top:
                model.load_weights(WEIGHTS_PATH_TOP)
            else:
                model.load_weights(WEIGHTS_PATH)
        else:
            print("failed to load weights")

    return model