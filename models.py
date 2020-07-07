from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, Conv3D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape

from keras.layers import Deconvolution3D
from keras.optimizers import SGD, Adadelta
from keras import regularizers
from keras.layers import LSTM, RepeatVector, concatenate
from keras import backend as K

"""
Defining Keras models as functions which return model object, aswell as model name and mode type strs.
All models take take img_width and img_height ints, which correpsond to dimensions of images passed to models.

"""
def DSTCAE_C3D(img_width, img_height, win_length):

    input_shape = (win_length, img_width, img_height, 1)
    input_window = Input(shape = input_shape)
    temp_pool = 2

    x = Conv3D(16, (5, 3, 3), activation='relu', padding='same')(input_window)
    x = MaxPooling3D((1, 2, 2), padding='same')(x)

    x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #4

    x = Dropout(0.25)(x)

    x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #2

    # at this point the representation is (2, 8, 8) i.e. 128-dimensional

    x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((temp_pool, 2, 2))(x) #4

    x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((temp_pool, 2, 2))(x) #8

    x = Conv3D(16, (5, 3, 3), activation='relu', padding = 'same')(x)
    x = UpSampling3D((1, 2, 2))(x)

    decoded = Conv3D(1, (5, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'DSTCAE_C3D'
    model = autoencoder

    return model, model_name, model_type


def DSTCAE_UpSampling(img_width, img_height, win_length):
    """
    int win_length: Length of window of frames
    """

    input_shape = (win_length, img_width, img_height, 1)
    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5

    x = Conv3D(16, (temp_depth, 3, 3), activation='relu', padding='same')(input_window)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    # x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)

    x = Dropout(0.25)(x)

    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)

    # at this point the representation is (2, 16, 16) i.e. 128-dimensional

    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((temp_pool, 2, 2))(x)

    x = Conv3D(16, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((temp_pool, 2, 2))(x)

    decoded = Conv3D(1, (temp_depth, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'DSTCAE_UpSamp'
    model = autoencoder

    return model, model_name, model_type


def DSTCAE_Deconv(img_width, img_height, win_length):
    """
    int win_length: Length of window of frames

    Replace Upsampling with Deconv
    """

    input_shape = (win_length, img_width, img_height, 1)
    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    
    x = Conv3D(16, (temp_depth, 3,3), activation='relu', padding='same')(input_window)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    #x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)

    x = Deconvolution3D(8, (temp_depth, 3, 3), strides = (2,2,2), activation='relu', padding='same')(encoded)
    x = Deconvolution3D(16, (temp_depth, 3, 3), strides = (2,2,2), activation='relu', padding='same')(x)
    
    decoded = Conv3D(1, (temp_depth, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'DSTCAE_Deconv'
    model = autoencoder

    return model, model_name, model_type


import numpy as np
if __name__ == "__main__":
    model = dummy_3d(64,64,2)
    print(model.summary())
    # dummy = np.ones((1,8,64,64,1))*255
    # #dummy = dummy- np.mean(dummy)

