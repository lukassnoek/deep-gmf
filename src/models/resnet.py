from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Add
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D


def ResBlock(x, filters, kernel_size=3, stride=2, block=1):
    """ Block of Conv2D layers with a skip connection
    at the end (which by itself also contains a conv layer).
    Note that the `stride` param is only used in the first
    conv layer; the second conv layer always uses a stride of 1
    so not to downsample twice in a block.
    
    Parameters
    ----------
    x : Tensor
        Input to block
    filters : list
        List of number of filters per block
    kernel_size : int
        Single kernel size (assumed to be square)
    stride : int
        Stride to use for 
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
        x = Conv2D(filters, kernel_size, stride, padding='same',
                   kernel_initializer='he_normal',
                   name=f'conv1_bl{block}')(x)
        x = BatchNormalization(name=f'bn_conv1_bl{block}')(x)
        x = Activation('relu')(x)

        # Assume that it has the same number of filters
        # Also, stride is per definition 1 (because we only need to
        # subsample once)
        x = Conv2D(filters, kernel_size, 1, padding='same',
                   kernel_initializer='he_normal',
                   name=f'conv2_bl{block}')(x)
        x = BatchNormalization(name=f'bn_conv2_bl{block}')(x)

        # Note: you need an additional Conv layer to make sure the shortcut has
        # the same number of filters as the previous conv layers; same for stride
        sc = Conv2D(filters, kernel_size, stride, padding='same',
                    kernel_initializer='he_normal',
                    name=f'convsc_bl{block}')(sc)
        
        x = Add()([x, sc])
        x = Activation('relu')(x)

        return x

    return apply


def ResNet10(input_shape=(224, 224, 3), n_classes=4):
    """ ResNet10 model.
    
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
    
    s = Input(input_shape)  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='conv_init')(s)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, 2, padding='same')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128, 256, 512]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlock(x, nf, kernel_size=3, stride=stride, block=i+1)(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D()(x)

    # Add classification head
    if n_classes == 2:
        y = Dense(1, activation='sigmoid', name='fc')(x)
    else:
        y = Dense(n_classes, activation='softmax', name='fc')(x)

    # Create model
    model = Model(inputs=s, outputs=y, name='ResNet10')
    return model


def ResNet6(input_shape=(224, 224, 3), n_classes=4):
    """ ResNet6 model.
    
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
    
    s = Input(input_shape)  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='conv_init')(s)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, 2, padding='same')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlock(x, nf, kernel_size=3, stride=stride, block=i+1)(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D()(x)

    # Add classification head
    if n_classes == 2:
        y = Dense(1, activation='sigmoid', name='fc')(x)
    else:
        y = Dense(n_classes, activation='softmax', name='fc')(x)

    # Create model
    model = Model(inputs=s, outputs=y, name='ResNet6')
    return model


if __name__ == '__main__':

    import numpy as np
    from tensorflow.keras.utils import to_categorical

    # Simulate some data from a linear model, Y = argmax(XW)
    N, C = 512, 4
    X = np.random.normal(0, 1, size=(N, 224, 224, 3)).astype('float32')
    W = np.random.normal(0, 0.005, size=(C, 224, 224, 3)).astype('float32')
    Z = X.reshape((N, 224 * 224 * 3)) @ W.reshape((224 * 224 * 3, C))
    Y = to_categorical(Z.argmax(axis=1))

    # Initialize and compile model
    model = ResNet10(n_classes=C)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    # Fit!    
    model.fit(X, Y, batch_size=64, epochs=5, validation_split=0.2)

    X = np.random.normal(0, 1, size=(N, 224, 224, 3)).astype('float32')
    Z = X.reshape((N, 224 * 224 * 3)) @ W.reshape((224 * 224 * 3, C))
    Y = to_categorical(Z.argmax(axis=1))
    model.evaluate(x=X, y=Y)
