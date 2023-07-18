"""VGG16 implementation, adapted from
https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

But could never make it train properly on GMF data, so switched to ResNet models.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def ConvBlock(filters, n_conv=2, kernel_size=3, block=1, layer=1):
    """ Conv block for VGG16 model. 
    
    Parameters
    ----------
    filters : list
        List of number of filters per block
    n_conv : int
        How many convolution layers this block should do
    kernel_size : int
        Single kernel size (assumed to be square)
    block : int
        Index of block (to be used in names)

    Returns
    -------
    apply : function
        Function that applies the convolutional block
    """
    def apply(x, layer=layer):
        """ Apply operations to input tensor `x`. """        
        for i in range(1, n_conv+1):
            x = Conv2D(filters, kernel_size, activation='relu', padding='same',
                       name=f'layer{layer}_block-{block}_conv')(x)
            layer += 1
        
        x = MaxPooling2D(2, strides=2, name=f'layer{layer-1}_block-{block}_pool')(x)

        return x
    
    return apply


def VGG16(input_shape=(224, 224, 3)):
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    x = s

    filters_ = [64, 128, 256, 512, 512]
    n_conv_ = [2, 2, 3, 3, 3]
    layer = 1
    for block, (filters, n_conv) in enumerate(zip(filters_, n_conv_)):
        x = ConvBlock(filters, n_conv, 3, block=block+1, layer=layer)(x)
        layer += n_conv

    # Classification block
    x = Flatten(name=f'layer{layer-1}_flatten')(x)  # flatten is part of last conv layer
    x = Dense(units=4096, activation='relu', name=f'layer{layer}_fc')(x)
    x = Dense(units=4096, activation='relu', name=f'layer{layer+1}_fc')(x)

    model = Model(inputs=s, outputs=x, name='VGG16')
    return model
