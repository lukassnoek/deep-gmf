from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense


def ResBlockStack(filters, n_layers=2, kernel_size=3, stride=2, bn_momentum=0.01, block=1):
    """ Stack of ResNet (V2!) blocks with Conv2D layers and skip connections.

    * block = [layer, layer]
    * stack = [block, block, ... , block]  (depends on `n_layers`)
    
    Each stack consists of `n_layers` conv layers, which contain skip connections
    after every 2 layers (i.e., after each 'block'). For simplicity, this
    implementation assumes that the same number of filters is used throughout the
    stack! Note that the `stride` parameter is only used in the first conv layer;
    the subsequent conv layers always use a stride of 1, as to not downsample more
    than once within the stack.
    
    Parameters
    ----------
    filters : list
        List of number of filters per block
    n_layers : int
        Number of convolution layers per stack (default: 2)
    kernel_size : int
        Single kernel size (assumed to be square)
    stride : int
        Stride to use for convolution
    bn_momentum : float
        Momentum used for batch norm layers
    block : int
        Index of block (to be used in layer names)

    Returns
    -------
    apply : function
        Function that applies the convolutional block
    """

    if n_layers < 2:
        raise ValueError("Number of layers should be 2 or more!")

    def apply(x):
        """ Applies the block to the input x. """

        sc = x  # always define the shortcut at the start of the block

        # First conv layer usually downsamples using a stride of 2
        name = f'layer{(block-1) * n_layers + 2}_block-{block}'
        x = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')(x)
        x = Activation('relu', name=f'{name}_relu')(x)
        x = Conv2D(filters, kernel_size, stride, padding='same',
                   kernel_initializer='he_normal',
                   name=f'{name}_conv')(x)

        # Subsequent layers use the same number of filters, but a stride of 1        
        for i in range(1, n_layers):
            
            # Every even layer, define a shortcut (skip connection)
            if i % 2 == 0:
                sc = x  # sc = shortcut
            
            # Regular conv layer with a stride of 1 (no downsampling)
            name = f'layer{(block-1) * n_layers + 2 + i}_block-{block}'
            x = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')(x)
            x = Activation('relu', name=f'{name}_relu')(x)
            x = Conv2D(filters, kernel_size, 1, padding='same',
                    kernel_initializer='he_normal',
                    name=f'{name}_conv')(x)
            
            # At odd layers, add the shortcut!
            if i % 2 != 0:
                # In the first skip connection within the stack, we need to
                # downsample the shortcut because of the downsampling happening
                # in the first conv layer
                if i == 1:
                    # Note to self: in this implementation of ResNet, the number of filters
                    # are assumed to always be the same *within a block* 
                    # Also: kernel size is always 1 when downsampling!
                    sc = Conv2D(filters, 1, stride, padding='same',
                                kernel_initializer='he_normal',
                                name=f'{name}_conv_shortcut')(sc)

                # Add skip connection
                x = Add(name=f'{name}_add')([x, sc])

        return x

    return apply


def ResNet6(input_shape=(224, 224, 3), bn_momentum=0.01):
    """ ResNet6 model.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    bn_momentum : float
        Momentum used for batch norm layers

    Returns
    -------
    model : Model
        Keras model
    """    
    
    s = Input(input_shape, name='layer0_input')  # s = stimulus

    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')(x)
    x = Activation('relu', name='layer1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer1_pool')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlockStack(nf, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer5_bnorm')(x)
    x = Activation('relu', name='layer5_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer5_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet6')

    return model


def ResNet10(input_shape=(224, 224, 3), bn_momentum=0.9):
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
    
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')(x)
    x = Activation('relu', name='layer1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer1_pool')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128, 256, 512]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlockStack(nf, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer9_bnorm')(x)
    x = Activation('relu', name='layer9_relu')(x)

    x = GlobalAveragePooling2D(name='layer9_globalpool')(x)
    # Note to self: use regular max pooling instead of
    # global max pooling, because the latter results in
    # too few features (512) for shape decoding
    #x = MaxPooling2D(strides=2, padding='valid', name='layer9_maxpool')(x)
    #x = Flatten(name='layer9_flatten')(x)
    
    # Create model
    model = Model(inputs=s, outputs=x, name='ResNet10')
    
    return model


def ResNet18(input_shape=(224, 224, 3), bn_momentum=0.01):
    """ ResNet18 model.
    
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
    
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')(x)
    x = Activation('relu', name='layer1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer1_pool')(x)

    # Add four residual conv blocks with an increasing number of filters,
    n_filters = [64, 128, 256, 512]
    for i, nf in enumerate(n_filters):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlockStack(nf, n_layers=4, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer17_bnorm')(x)
    x = Activation('relu', name='layer17_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer17_globalpool')(x)
    
    # Create model
    model = Model(inputs=s, outputs=x, name='ResNet18')
    
    return model


def ResNet34(input_shape=(224, 224, 3), bn_momentum=0.01):
    """ ResNet34 model.
    
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
    
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    
    # ResNet initial block: Conv block + Maxpool
    x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
               kernel_initializer='he_normal',
               name='layer1_conv')(s)
    x = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')(x)
    x = Activation('relu', name='layer1_relu')(x)
    x = MaxPooling2D(3, 2, padding='same', name='layer1_pool')(x)

    n_filters = [64, 128, 256, 512]
    n_layers = [6, 8, 12, 6]
    for i, (nf, nl) in enumerate(zip(n_filters, n_layers)):
        # Use a stride of 1 if it's the first block, else 2 for downsampling
        stride = 1 if i == 0 else 2
        x = ResBlockStack(nf, n_layers=nl, kernel_size=3, stride=stride, bn_momentum=bn_momentum, block=i+1)(x)

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer33_bnorm')(x)
    x = Activation('relu', name='layer33_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer33_globalpool')(x)
    
    # Create model
    model = Model(inputs=s, outputs=x, name='ResNet34')
    
    return model
    


if __name__ == '__main__':

    # Initialize and compile model
    model = ResNet10()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    
    # from tensorflow.keras.utils import plot_model
    # plot_model(
    #     model,
    #     to_file="model.png",
    #     show_shapes=False,
    #     show_dtype=False,
    #     show_layer_names=True,
    #     rankdir="TB",
    #     expand_nested=False,
    #     dpi=96,
    #     layer_range=None,
    #     show_layer_activations=False,
    # )