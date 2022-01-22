from __future__ import division
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import * 
import tensorflow.keras as keras 
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import categorical_crossentropy
from skimage.morphology import label
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
import numpy as np

def _common_spectral_pool(images, filter_size):
    '''shape: (batch_size, num_channels, height, width)
    shape: (batch_size, height, width,num_channels)'''
    assert len(images.get_shape().as_list()) == 4
    #print(images.get_shape().as_list())
    assert filter_size >= 3
    if filter_size % 2 == 1:
        n = int((filter_size-1)/2)
        top_left = images[:, :n+1, :n+1, :]
        #print(top_left)
        top_right = images[:, :n+1, -n:, :]
        #print(top_right)
        bottom_left = images[:, -n:, :n+1, :]
        #print(bottom_left)
        bottom_right = images[:, -n:, -n:, :]
        #print(bottom_right)
        top_combined = tf.concat([top_left, top_right], axis=-2)
        #print(top_combined)
        bottom_combined = tf.concat([bottom_left, bottom_right], axis=-2)
        #print(bottom_combined)
        all_together = tf.concat([top_combined, bottom_combined], axis=1)
        #print(all_together)
    else:
        n = filter_size // 2
        top_left = images[:, :n, :n, :]
        #print(top_left)
        top_middle = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, :n, n, :] + images[:, :n, -n, :]),
            -2
        )
        #print(top_middle)
        top_right = images[:, :n, -(n-1):, :]
        #print(top_right)
        middle_left = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, n, :n, :] + images[:, -n, :n, :]),
            -3
        )
        #print(middle_left)
        middle_middle = tf.expand_dims(
            tf.expand_dims(
                tf.cast(0.5, tf.complex64) *
                (images[:, n, n, :] + images[:, n, -n, :] +
                 images[:, -n, n, :] + images[:, -n, -n, :]),
                -2
            ),
            -2
        )
        #print(middle_middle)
        middle_right = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, n, -(n-1):, :] + images[:, -n, -(n-1):, :]),
            -3
        )
        #print(middle_right)
        bottom_left = images[:, -(n-1):, :n, :]
        #print(bottom_left)
        bottom_middle = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, -(n-1):, n, :] + images[:, -(n-1):, -n, :]),
            -2
        )
        #print(bottom_middle)
        bottom_right = images[:, -(n-1):, -(n-1):, :]
        #print(bottom_right)
        top_combined = tf.concat(
            [top_left, top_middle, top_right],
            axis=-2
        )
        #print(top_combined)
        middle_combined = tf.concat(
            [middle_left, middle_middle, middle_right],
            axis=-2
        )
        #print(middle_combined)
        bottom_combined = tf.concat(
            [bottom_left, bottom_middle, bottom_right],
            axis=-2
        )
        #print(bottom_combined)
        all_together = tf.concat(
            [top_combined, middle_combined, bottom_combined],
            axis=-3
        )
        #print(all_together)
    return all_together

class spectral_pool_layer(object):
    """Spectral pooling layer."""

    def __init__(
        self,
        input_x,
        filter_size=3,
        freq_dropout_lower_bound=None,
        freq_dropout_upper_bound=None,
        activation=tf.nn.relu,
        m=0,
        train_phase=False
    ):
        """Perform a single spectral pool operation.

        Args:
            input_x: Tensor representing a batch of channels-first images
                shape: (batch_size, num_channels, height, width)
            filter_size: int, the final dimension of the filter required
            freq_dropout_lower_bound: The lowest possible frequency
                above which all frequencies should be truncated
            freq_dropout_upper_bound: The highest possible frequency
                above which all frequencies should be truncated
            train_phase: tf.bool placeholder or Python boolean,
                but using a Python boolean is probably wrong

        Returns:
            An image of similar shape as input after reduction
        """
        # assert only 1 dimension passed for filter size
        assert isinstance(filter_size, int)

        input_shape = input_x.get_shape().as_list()
    
        assert len(input_shape) == 4
        _, H, W, _ = input_shape
        #_, _, H, W = input_shape
        assert H == W
        

        with tf.compat.v1.variable_scope('spectral_pool_layer_{0}'.format(m)):
            # Compute the Fourier transform of the image
            im_fft = tf.compat.v1.fft2d(tf.compat.v1.cast(input_x, tf.compat.v1.complex64))

            # Truncate the spectrum
            im_transformed = _common_spectral_pool(im_fft, filter_size)
            if (
                freq_dropout_lower_bound is not None and
                freq_dropout_upper_bound is not None
            ):
                # If we are in the training phase, we need to drop all
                # frequencies above a certain randomly determined level.
                def true_fn():
                    tf_random_cutoff = tf.compat.v1.random_uniform(
                        [],
                        freq_dropout_lower_bound,
                        freq_dropout_upper_bound
                    )
                    dropout_mask = _frequency_dropout_mask(
                        filter_size,
                        tf_random_cutoff
                    )
                    return im_transformed * dropout_mask

                # In the testing phase, return the truncated frequency
                # matrix unchanged.
                def false_fn():
                    return im_transformed

                im_downsampled = tf.compat.v1.cond(
                    train_phase,
                    true_fn=true_fn,
                    false_fn=false_fn
                )
                im_out = tf.compat.v1.real(tf.compat.v1.ifft2d(im_downsampled))
            else:
                im_out = tf.compat.v1.real(tf.compat.v1.ifft2d(im_transformed))

            if activation is not None:
                cell_out = activation(im_out)
            else:
                cell_out = im_out
            tf.compat.v1.summary.histogram('sp_layer/{}/activation'.format(m), cell_out)

        self.cell_out = cell_out
    def output(self):
        return self.cell_out

def get_spectral_pool_layer_same(input_x):
    channel = int(input_x.shape[-1])
    kernel_size = int(input_x.shape[1])
    spl = spectral_pool_layer(input_x=input_x,
    filter_size=kernel_size,
    freq_dropout_lower_bound=None,
    freq_dropout_upper_bound=None,
    activation=tf.nn.relu,
    m=0,
    train_phase=True)
    return spl.cell_out

def get_spectral_pool_layer_valid(input_x):
    channel = int(input_x.shape[-1])
    kernel_size = int(input_x.shape[1])
    spl = spectral_pool_layer(input_x=input_x,
    filter_size=int(kernel_size/2),
    freq_dropout_lower_bound=None,
    freq_dropout_upper_bound=None,
    activation=tf.nn.relu,
    m=0,
    train_phase=True)
    return spl.cell_out

def hybrid_pool_layer_same(x, pool_size=(2,2)):
    channel = int(x.shape[-1])
    kernel_size = int(x.shape[1])
    return Conv2D(int(x.shape[-1]), (1, 1))(
        concatenate([
            MaxPooling2D(pool_size, padding='same', strides=1)(x),
            Lambda(get_spectral_pool_layer_same, output_shape=(kernel_size, kernel_size, channel)) (x)]))

def hybrid_pool_layer_valid(x, pool_size=(2,2)):
    channel = int(x.shape[-1])
    kernel_size = int(x.shape[1])
    return Conv2D(int(x.shape[-1]), (1, 1))(
        concatenate([
            MaxPooling2D(pool_size)(x),
            Lambda(get_spectral_pool_layer_valid, output_shape=(int(kernel_size/2), int(kernel_size/2), channel)) (x)]))

def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def double_inception_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    axis = 3
    conv1 = SeparableConv2D(size, (filter_size, filter_size), padding='same')(x)
    conv2 = SeparableConv2D(size, (1, 1), padding='same')(x)
    conv3 = SeparableConv2D(size, (5, 5), padding='same')(x)
    pool = hybrid_pool_layer_same(x)
    if batch_norm is True:
        conv1 = BatchNormalization(axis=axis)(conv1)
        conv2 = BatchNormalization(axis=axis)(conv2)
        conv3 = BatchNormalization(axis=axis)(conv3)
        pool = BatchNormalization(axis=axis)(pool)
    conv1 = Activation('relu')(conv1)
    conv2 = Activation('relu')(conv2)
    conv3 = Activation('relu')(conv3)
    pool = Activation('relu')(pool)
    
    concat_1 = concatenate([conv1,conv2,conv3,pool],axis=-1)
    concat_conv = SeparableConv2D(size, (1, 1), padding='same')(concat_1)
    
    conv1_1 = SeparableConv2D(size, (filter_size, filter_size), padding='same')(concat_conv)
    conv2_1 = SeparableConv2D(size, (filter_size, filter_size), padding='same')(concat_conv)
    conv3_1 = SeparableConv2D(size, (filter_size, filter_size), padding='same')(concat_conv)
    pool_1 = hybrid_pool_layer_same(concat_conv)
    if batch_norm is True:
        conv1_1 = BatchNormalization(axis=axis)(conv1_1)
        conv2_1 = BatchNormalization(axis=axis)(conv2_1)
        conv3_1 = BatchNormalization(axis=axis)(conv3_1)
        pool_1 = BatchNormalization(axis=axis)(pool_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv3_1 = Activation('relu')(conv3_1)
    pool_1 = Activation('relu')(pool_1)
    
    concat_2 = concatenate([conv1_1,conv2_1,conv3_1,pool_1],axis=-1)
    conv = SeparableConv2D(size, (1, 1), padding='same')(concat_2)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = SeparableConv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=axis)(shortcut)

    res_path = add([shortcut, conv])
    return res_path


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    axis = 3
    conv = SeparableConv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = SeparableConv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=axis)(shortcut)

    res_path = add([shortcut, conv])
    return res_path

def LSTM_concat(input1, input2):
    shape = input1.shape
    shape2 = input2.shape
    print('Keras shape, ', shape, shape2)
    x1 = Reshape(target_shape=(1, shape[1], shape[2], shape[3]))(input1)
    x2 = Reshape(target_shape=(1, shape2[1], shape2[2], shape2[3]))(input2)
    merge  = concatenate([x1,x2], axis = -1) 
    merge = ConvLSTM2D(filters = int(shape2[3]/2), kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge)
    merge = BatchNormalization()(merge)
    merge = Activation('relu')(merge)
    return merge


def gating_signal(input, out_size, batch_norm=False):
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def cs_attention_block(x, y, z, inter_shape, name): 
    shape_x = x.shape
    shape_g = z.shape

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = theta_x.shape
    
    theta_y = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(y)  # 16
    shape_theta_y = theta_y.shape
    
    phi_z = Conv2D(inter_shape, (1, 1), padding='same')(z)
    phi_z = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_z)  # 16

    concat_xg = add([theta_x, theta_y, phi_z])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = sigmoid_xg.shape
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),
                                       name=name+'_weight')(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])


    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    result_bn = Activation('relu')(result_bn)
    return result_bn

def LSTM_Unet(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs = inputs, outputs = conv9)
    #model = multi_gpu_model(model, gpus=2)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    return model

def RCAIUnet_Unet(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetric(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + RIC
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_inception_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_64 = double_inception_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    
    conv_32 = double_inception_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_16 = double_inception_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = double_inception_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=axis)
    up_conv_16 = double_inception_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=axis)
    up_conv_32 = double_inception_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=axis)
    up_conv_64 = double_inception_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=axis)
    up_conv_128 = double_inception_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetcsa(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + CSA
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating1_16 = gating_signal(pool_16, 8*FILTER_NUM, batch_norm)
    gating2_16 = gating_signal(conv_16, 8*FILTER_NUM, batch_norm)
    gating3_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = cs_attention_block(gating1_16, gating2_16, gating3_16, 8*FILTER_NUM, name='att_16')
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    gating1_32 = gating_signal(pool_32, 4*FILTER_NUM, batch_norm)
    gating2_32 = gating_signal(conv_32, 4*FILTER_NUM, batch_norm)
    gating3_32 = gating_signal(conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = cs_attention_block(gating1_32, gating2_32, gating3_32, 4*FILTER_NUM, name='att_32')
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    gating1_64 = gating_signal(pool_64, 2*FILTER_NUM, batch_norm)
    gating2_64 = gating_signal(conv_64, 2*FILTER_NUM, batch_norm)
    gating3_64 = gating_signal(conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = cs_attention_block(gating1_64, gating2_64, gating3_64, 2*FILTER_NUM, name='att_64')
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    gating1_128 = gating_signal(inputs, FILTER_NUM, batch_norm)
    gating2_128 = gating_signal(conv_128, FILTER_NUM, batch_norm)
    gating3_128 = gating_signal(conv_64, FILTER_NUM, batch_norm)
    att_128 = cs_attention_block(gating1_128, gating2_128, gating3_128, FILTER_NUM, name='att_128')
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetrichp(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + RIC + HP
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_inception_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = hybrid_pool_layer_valid(conv_128)
    
    conv_64 = double_inception_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = hybrid_pool_layer_valid(conv_64)
    
    conv_32 = double_inception_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = hybrid_pool_layer_valid(conv_32)
    
    conv_16 = double_inception_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = hybrid_pool_layer_valid(conv_16)
    
    conv_8 = double_inception_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=axis)
    up_conv_16 = double_inception_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=axis)
    up_conv_32 = double_inception_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=axis)
    up_conv_64 = double_inception_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=axis)
    up_conv_128 = double_inception_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetcsahp(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + CSA + HP
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = hybrid_pool_layer_valid(conv_128)
    
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = hybrid_pool_layer_valid(conv_64)
    
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = hybrid_pool_layer_valid(conv_32)
    
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = hybrid_pool_layer_valid(conv_16)
    
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating1_16 = gating_signal(pool_16, 8*FILTER_NUM, batch_norm)
    gating2_16 = gating_signal(conv_16, 8*FILTER_NUM, batch_norm)
    gating3_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = cs_attention_block(gating1_16, gating2_16, gating3_16, 8*FILTER_NUM, name='att_16')
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    gating1_32 = gating_signal(pool_32, 4*FILTER_NUM, batch_norm)
    gating2_32 = gating_signal(conv_32, 4*FILTER_NUM, batch_norm)
    gating3_32 = gating_signal(conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = cs_attention_block(gating1_32, gating2_32, gating3_32, 4*FILTER_NUM, name='att_32')
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    gating1_64 = gating_signal(pool_64, 2*FILTER_NUM, batch_norm)
    gating2_64 = gating_signal(conv_64, 2*FILTER_NUM, batch_norm)
    gating3_64 = gating_signal(conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = cs_attention_block(gating1_64, gating2_64, gating3_64, 2*FILTER_NUM, name='att_64')
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    gating1_128 = gating_signal(inputs, FILTER_NUM, batch_norm)
    gating2_128 = gating_signal(conv_128, FILTER_NUM, batch_norm)
    gating3_128 = gating_signal(conv_64, FILTER_NUM, batch_norm)
    att_128 = cs_attention_block(gating1_128, gating2_128, gating3_128, FILTER_NUM, name='att_128')
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetcsaric(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + CSA + RIC
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_inception_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_64 = double_inception_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    
    conv_32 = double_inception_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_16 = double_inception_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = double_inception_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating1_16 = gating_signal(pool_16, 8*FILTER_NUM, batch_norm)
    gating2_16 = gating_signal(conv_16, 8*FILTER_NUM, batch_norm)
    gating3_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = cs_attention_block(gating1_16, gating2_16, gating3_16, 8*FILTER_NUM, name='att_16')
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_inception_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    gating1_32 = gating_signal(pool_32, 4*FILTER_NUM, batch_norm)
    gating2_32 = gating_signal(conv_32, 4*FILTER_NUM, batch_norm)
    gating3_32 = gating_signal(conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = cs_attention_block(gating1_32, gating2_32, gating3_32, 4*FILTER_NUM, name='att_32')
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_inception_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    gating1_64 = gating_signal(pool_64, 2*FILTER_NUM, batch_norm)
    gating2_64 = gating_signal(conv_64, 2*FILTER_NUM, batch_norm)
    gating3_64 = gating_signal(conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = cs_attention_block(gating1_64, gating2_64, gating3_64, 2*FILTER_NUM, name='att_64')
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_inception_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    gating1_128 = gating_signal(inputs, FILTER_NUM, batch_norm)
    gating2_128 = gating_signal(conv_128, FILTER_NUM, batch_norm)
    gating3_128 = gating_signal(conv_64, FILTER_NUM, batch_norm)
    att_128 = cs_attention_block(gating1_128, gating2_128, gating3_128, FILTER_NUM, name='att_128')
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_inception_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model

def RCAIUnet_Unetcsarichp(input_size=(256,256,1), dropout_rate=0.0, batch_norm=True):
    # Model: U-Net + RIC + HP + CSA = RCAI-Unet
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    OUTPUT_MASK_CHANNEL = 1
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Downsampling layers
    conv_128 = double_inception_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = hybrid_pool_layer_valid(conv_128)
    
    conv_64 = double_inception_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = hybrid_pool_layer_valid(conv_64)
    
    conv_32 = double_inception_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = hybrid_pool_layer_valid(conv_32)
    
    conv_16 = double_inception_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = hybrid_pool_layer_valid(conv_16)
    
    conv_8 = double_inception_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating1_16 = gating_signal(pool_16, 8*FILTER_NUM, batch_norm)
    gating2_16 = gating_signal(conv_16, 8*FILTER_NUM, batch_norm)
    gating3_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = cs_attention_block(gating1_16, gating2_16, gating3_16, 8*FILTER_NUM, name='att_16')
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_inception_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    gating1_32 = gating_signal(pool_32, 4*FILTER_NUM, batch_norm)
    gating2_32 = gating_signal(conv_32, 4*FILTER_NUM, batch_norm)
    gating3_32 = gating_signal(conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = cs_attention_block(gating1_32, gating2_32, gating3_32, 4*FILTER_NUM, name='att_32')
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_inception_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    gating1_64 = gating_signal(pool_64, 2*FILTER_NUM, batch_norm)
    gating2_64 = gating_signal(conv_64, 2*FILTER_NUM, batch_norm)
    gating3_64 = gating_signal(conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = cs_attention_block(gating1_64, gating2_64, gating3_64, 2*FILTER_NUM, name='att_64')
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_inception_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    gating1_128 = gating_signal(inputs, FILTER_NUM, batch_norm)
    gating2_128 = gating_signal(conv_128, FILTER_NUM, batch_norm)
    gating3_128 = gating_signal(conv_64, FILTER_NUM, batch_norm)
    att_128 = cs_attention_block(gating1_128, gating2_128, gating3_128, FILTER_NUM, name='att_128')
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_inception_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # Output
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)
    
    model = Model(inputs = inputs, outputs = conv_final)
    
    return model