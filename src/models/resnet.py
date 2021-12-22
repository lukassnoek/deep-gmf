from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Add
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D


def ResBlock(filters, kernel_size=3, stride=2, bn_momentum=0.01, block=1):
    """ Block of Conv2D layers with a skip connection
    at the end (which by itself also contains a conv layer).
    Note that the `stride` param is only used in the first
    conv layer; the second conv layer always uses a stride of 1
    so not to downsample twice in a block.
    
    Parameters
    ----------
    filters : list
        List of number of filters per block
    kernel_size : int
        Single kernel size (assumed to be square)
    stride : int
        Stride to use for convolution
    bn_momentum : float
        Momentum used for batch norm layers
    block : int
        Index of block (to be used in names)

    Returns
    -------
    apply : function
        Function that applies the convolutional block
    """

    def apply(x):
        """ Applies the block to the input x. """
        sc = x  # sc = shortcut
        
        # Block with 2 conv layers
        name = f'layer-{(block-1) * 2 + 2}_block-{block}'
        x = Conv2D(filters, kernel_size, stride, padding='same',
                   kernel_initializer='he_normal',
                   name=f'{name}_conv')(x)
        x = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')(x)
        x = Activation('relu', name=f'{name}_relu')(x)

        # Assume that it has the same number of filters
        # Also, stride is per definition 1 (because we only need to
        # subsample once)
        name = f'layer-{(block-1) * 2 + 3}_block-{block}'
        x = Conv2D(filters, kernel_size, 1, padding='same',
                   kernel_initializer='he_normal',
                   name=f'{name}_conv')(x)
        x = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')(x)

        # Note: you need an additional Conv layer to make sure the shortcut has
        # the same number of filters as the previous conv layers; same for stride
        sc = Conv2D(filters, kernel_size, stride, padding='same',
                    kernel_initializer='he_normal',
                    name=f'{name}_conv_shortcut')(sc)

        x = Add(name=f'{name}_add')([x, sc])
        x = Activation('relu', name=f'{name}_relu')(x)

        return x

    return apply


def ResNet10(input_shape=(224, 224, 3), n_classes=4, bn_momentum=0.01):
    """ ResNet10 model.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    n_classes : int
        Number of output classes
    bn_momentum : float
        Momentum used for batch norm layers

    Returns
    -------
    model : Model
        Keras model
    """    
    
    s = Input(input_shape, name='layer-0_input')  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer-1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer-1_bnorm')(x)
    x = Activation('relu', name='layer-1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer-1_pool')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128, 256, 512]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlock(nf, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer-9_globalpool')(x)

    # Add classification head
    if n_classes == 2:
        y = Dense(1, activation='sigmoid', name='layer-10_fc')(x)
    else:
        y = Dense(n_classes, activation='softmax', name='layer-10_fc')(x)

    # Create model
    model = Model(inputs=s, outputs=y, name='ResNet10')
    return model


def ResNet6(input_shape=(224, 224, 3), n_classes=4, bn_momentum=0.01):
    """ ResNet6 model.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    n_classes : int
        Number of output classes
    bn_momentum : float
        Momentum used for batch norm layers

    Returns
    -------
    model : Model
        Keras model
    """    
    
    s = Input(input_shape, name='layer-0_input')  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer-1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer-1_bnorm')(x)
    x = Activation('relu', name='layer-1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer-1_pool')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlock(nf, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer-5_globalpool')(x)

    # Add classification head
    if n_classes == 2:
        y = Dense(1, activation='sigmoid', name='layer-6_fc')(x)
    else:
        y = Dense(n_classes, activation='softmax', name='layer-6_fc')(x)

    # Create model
    model = Model(inputs=s, outputs=y, name='ResNet6')
    return model


if __name__ == '__main__':

    # Initialize and compile model
    model = ResNet10(n_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())