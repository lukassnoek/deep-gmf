import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Add
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Activation

from .resnet import ResBlock, Stem


def StereoNet(input_shape=(112, 112, 3), split_hemi=False, bn_momentum=0.9):

    sl = Input(input_shape, name='layer0_input-left')
    sr = Input(input_shape, name='layer0_input-right')

    conv1 = Conv2D(64, 7, 2, padding='same', kernel_initializer='he_normal', name='layer1_conv')
    bn1 = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')
    relu1 = Activation('relu', name='layer1_relu')
    mp1 = MaxPooling2D(3, 2, padding='same', name='layer1_pool')

    xl, xr = sl, sr
    for lay in [conv1, bn1, relu1, mp1]:
        xl = lay(xl)
        xr = lay(xr)

    x = Concatenate(axis=-1)([xl, xr])

    # if split_hemi:
    #     x_l, x_r = Lambda(tf.split, arguments={'num_or_size_splits': 2, 'axis': 1})(x)
    #     x_l = shared(x_l)
    #     x_r = shared(x_r)
    #     x = Concatenate(axis=1)([x_l, x_r])
    # else:
    #     x = shared(x)

    # Stage 1
    #rb = ResBlock((64,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1, as_model=True, input_shape=xl.shape)
    x = ResBlock((64,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)

    #xl = rb(xl)
    #xr = rb(xr)
    #x = Concatenate(axis=-1)([xl, xr])

    # Stage 2
    x = ResBlock((128,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=4, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=6, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=8, stage=4)(x)

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer9_bnorm')(x)
    x = Activation('relu', name='layer9_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer9_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=[sl, sr], outputs=x, name='StereoResNet10')    
    return model


if __name__ == '__main__':
    mod = StereoNet()
    print(mod.summary())