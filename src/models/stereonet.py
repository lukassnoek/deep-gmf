import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Add, Average, Subtract

from .resnet import ResBlock, Stem, PostActivation


def StereoResNet6(input_shape=(112, 112, 3), fuse_after_stage=0, bn_momentum=0.1):

   # Input layers (sl = stimulus left eye, sr = stimulus right eye)
    sl = Input(input_shape, name='layer0_input-left')
    sr = Input(input_shape, name='layer0_input-right')

    # Stem of ResNet model; downsamples twices (112 -> 56 -> 28)
    stem = Stem(bn_momentum=bn_momentum)
    xl = stem(sl)  # 28 x 28 x 64
    xr = stem(sr)  # 28 x 28 x 64

    # n_input channels at stage `t` (and n_filters at stage `t + 1`)
    n_channels = (64, 128, 256)
    img_size = 28  # image size after stem

    for stage in range(1, 3):
        
        layer_nr = 2 * stage
        n_ch = n_channels[stage - 1]
        n_filt = n_channels[stage]
        
        if fuse_after_stage == (stage - 1):
            # Fuse feature maps of left and right inputs
            x = Concatenate(axis=-1, name=f'layer{layer_nr}_concat')([xl, xr])
            n_ch *= 2  # n_channels is temporarily doubled because of concatenation
            #xm = Average(name=f'layer{layer_nr}_average')([xl, xr])
            #xs = Subtract(name=f'layer{layer_nr}_subtract')([xl, xr])
            #x = Add(name=f'layer{layer_nr}_merge')([xm, xs])

        if stage > 2:
            # In ResNets, the first stage is not spatially downsampled
            img_size = img_size // 2 

        # Define expected input shape for ResBlock        
        rb = ResBlock(n_ch, (n_filt,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=layer_nr,
                      stage=stage)

        if stage <= fuse_after_stage:
            # Process left and right inputs separately
            xl = rb(xl)
            xr = rb(xr)
        else:
            # Process fused feature maps
            x = rb(x)
    
    x = PostActivation(layer_nr=6, bn_momentum=bn_momentum)(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer N - 1, because dense layer (to be added)
    # represents layer N
    model = Model(inputs=[sl, sr], outputs=x, name='StereoResNet6')    
    return model


def StereoResNet10(input_shape=(112, 112, 3), fuse_after_stage=1, bn_momentum=0.9):
    """StereoNet model architecture that processes stereo images as inputs.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input images (height, width, channels)
    fuse_after_stage : int
        After which stage to fuse the feature maps of the left and right inputs
    bn_momentum : float
        Momentum for batch normalization layers

    Returns
    -------
    model : tensorflow.keras.Model
        StereoNet model instance
    """

    # Input layers (sl = stimulus left eye, sr = stimulus right eye)
    sl = Input(input_shape, name='layer0_input-left')
    sr = Input(input_shape, name='layer0_input-right')

    # Stem of ResNet model; downsamples twices (112 -> 56 -> 28)
    stem = Stem(bn_momentum=bn_momentum, as_model=True, input=(sr, sl))
    x = stem([sl, sr])  # 28 x 28 x 64

    # n_input channels at stage `t` (and n_filters at stage `t + 1`)
    n_channels = (64, 64, 128, 256, 512)
    img_size = 28  # image size after stem

    for stage in range(1, 5):
        
        layer_nr = 2 * stage
        n_ch = n_channels[stage - 1]
        n_filt = n_channels[stage]
        
        if fuse_after_stage == (stage - 1):
            # Fuse feature maps of left and right inputs
            x = Concatenate(axis=-1, name=f'layer{layer_nr}_concat')(x)
            n_ch *= 2  # n_channels is temporarily doubled because of concatenation
        
        if stage > 2:
            # In ResNets, the first stage is not spatially downsampled
            img_size = img_size // 2 

        # Define expected input shape for ResBlock        
        x = ResBlock((n_filt,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=layer_nr,
                      stage=stage, as_model=True, inputs=x)

    x = PostActivation(layer_nr=10, bn_momentum=bn_momentum)(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer N - 1, because dense layer (to be added)
    # represents layer N
    model = Model(inputs=[sl, sr], outputs=x, name='StereoResNet10')    
    return model


if __name__ == '__main__':
    mod = StereoResNet10()
    print(mod.summary())