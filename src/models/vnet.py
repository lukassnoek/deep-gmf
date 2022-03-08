from tensorflow.keras import Model
from tensorflow.keras.layers import Resizing, Input, Conv2D, Dropout, Dense
from tensorflow.keras.layers import MaxPooling2D, Flatten, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Add


def ConvPoolNorm(filters, kernel_size=3, dropout=False, pool=True, relu=True, bn_momentum=0.01, layer=1):
    """ Conv (+ Dropout) + Maxpool + Batchnorm layer.
    Very similar to original VNet implementation, except
    that dropout is optional and BatchNorm is used instead
    of GroupNorm (no added benefit if batch size is relatively
    large, I think).

    Parameters
    ----------
    filters : int
        Number of filters for convolution
    kernel_size : int
        Kernel size for convolution (assumed to be square)
    pool : bool
        Whether to perform max-pooling or not
    relu : bool
        Whether to perform ReLU activation (if it's the second
        layer in the block, it should not apply ReLU, because
        it should be done after the skip connection)
    layer : int
        Layer number (to be used in name of operations)
    """
    
    def apply(x):
        """ Apply block to input tensor (`x`). """   
        x = Conv2D(filters, kernel_size, padding='same',
                   kernel_initializer='he_normal',
                   name=f'layer{layer}_conv')(x)
    
        if dropout:
            x = Dropout(rate=0.2, name=f'layer{layer}_dropout')(x)
        
        if pool:
            x = MaxPooling2D(name=f'layer{layer}_maxpool')(x)

        x = BatchNormalization(momentum=bn_momentum, name=f'layer{layer}_bnorm')(x)

        if relu:  # Don't do ReLU when skip
            x = ReLU(name=f'layer{layer}_relu')(x)
        
        return x
    
    return apply


def ConvNorm(filters, kernel_size=3, stride=1, dropout=False, relu=True, bn_momentum=0.01, layer=1):
    """ Conv (+ Dropout) + Batchnorm layer.
    Very similar to ConvPoolNorm, but achieves downsampling
    using the stride hyperparameter in the Conv operation.
    
    Parameters
    ----------
    filters : int
        Number of filters for convolution
    kernel_size : int
        Kernel size for convolution (assumed to be square)
    stride : int
        Stride to using in Conv operation (if 2, then downsample 50%)
    relu : bool
        Whether to perform ReLU activation (if it's the second
        layer in the block, it should not apply ReLU, because
        it should be done after the skip connection)
    layer : int
        Layer number (to be used in name of operations)
    """
    
    def apply(x):
        """ Apply block to input tensor (`x`). """   
        x = Conv2D(filters, kernel_size, stride, padding='same',
                   kernel_initializer='he_normal',
                   name=f'layer{layer}_conv')(x)
    
        if dropout:
            x = Dropout(rate=0.2, name=f'layer{layer}_dropout')(x)
        
        x = BatchNormalization(momentum=bn_momentum, name=f'layer{layer}_bnorm')(x)

        if relu:
            x = ReLU(name=f'layer{layer}_relu')(x)
        
        return x
    
    return apply


def VNet(input_shape=(224, 224, 3), n_classes=4):
    """ VNet, adapted from:
    
    Mehrer, J., Spoerer, C. J., Jones, E. C., Kriegeskorte, N., & Kietzmann, T. C. (2021). 
    An ecologically motivated image dataset for deep learning yields better models of 
    human vision. Proceedings of the National Academy of Sciences, 118(8).
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    n_classes : int
        Number of output classes

    Returns
    -------
    model : Model
        Keras model
    """    

    s = Input(input_shape, name='layer0_input')  # s = stimulus
    x = Resizing(height=128, width=128, name='layer0_resize')(s)
    
    kernel_sizes = [7, 7, 5, 5, 3, 3, 3, 3, 1, 1]
    downsample = [False, False, True, True, False, False, True, True, True, True]
    filters_ = [128, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048]
    
    params = zip(kernel_sizes, downsample, filters_)
    for layer, (ks, pool, filters) in enumerate(params):
        x = ConvPoolNorm(filters, ks, pool=pool, layer=layer+1)(x)

    x = Flatten(name=f'layer{layer+1}_flatten')(x)
    model = Model(inputs=s, outputs=x, name='VNet')
    return model


def VNet_skip(input_shape=(224, 224, 3), n_classes=(4,), targets=('id',), include_top=True):
    """ VNet with skip connections and a substantially reduced
    number of parameters by using fewer filters in each conv layer. 
    Also achieves downsampling through convolution (with stride=2)
    and uses GlobalAveragePooling instead of flatten. But, notably,
    the height and width of the layers are identical to the original
    implementation.

    Network architecture adapted from:
    
    Mehrer, J., Spoerer, C. J., Jones, E. C., Kriegeskorte, N., & Kietzmann, T. C. (2021). 
    An ecologically motivated image dataset for deep learning yields better models of 
    human vision. Proceedings of the National Academy of Sciences, 118(8).
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    n_classes : int
        Number of output classes

    Returns
    -------
    model : Model
        Keras model
    """    

    s = Input(input_shape, name='layer0_input')  # s = stimulus
    x = Resizing(height=128, width=128, name='layer0_resize')(s)
    
    kernel_sizes = [7, 7, 5, 5, 3, 3, 3, 3, 1, 1]
    downsample = [False, False, True, True, False, False, True, True, True, True]
    filters_ = [32, 32, 64, 64, 128, 128, 256, 256, 512, 512]
    
    params = zip(kernel_sizes, downsample, filters_)
    for layer, (ks, ds, filters) in enumerate(params):
        
        relu = True
        if (layer + 1) in [1, 5]:
            sc = x  # shortcut
        elif (layer + 1) in [2, 6]:
            relu = False

        # Downsample using stride (instead of pool)
        stride = 2 if ds else 1
        x = ConvNorm(filters, ks, stride=stride, relu=relu, layer=layer+1)(x)

        if not relu:  # Must be right before skip; add skip!
            sc = Conv2D(filters, ks, padding='same', kernel_initializer='he_normal',
                        name=f'layer{layer+1}_conv_shortcut')(sc)
            x = Add(name=f'layer{layer+1}_add')([sc, x])
            x = ReLU(name=f'layer{layer+1}_relu')(x)

    # Global av pool instead of flatten
    x = GlobalAveragePooling2D(name=f'layer{layer+1}_globalpool')(x)

    model = Model(inputs=s, outputs=x, name='VNet-skip')
    return model


if __name__ == '__main__':

    model = VNet_skip()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())