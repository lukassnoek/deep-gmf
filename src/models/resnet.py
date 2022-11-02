from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Add
from tensorflow.keras.layers import GlobalAveragePooling2D


def ResBlock(filters, kernel_sizes=(3, 3), n_blocks=2, bn_momentum=0.01, layer_nr=1, stage=1, as_model=False, input_shape=None):
    """ Stack of ResNet (V2!) blocks with Conv2D layers and skip connections.

    * block = [layer, layer]
    * stack = [block, block, ... , block]  (depends on `n_blocks`)
    
    Each stack consists of `n_blocks` of a number of conv layers, which depends on the 
    length of `filters` and `kernel_sizes`. For example, if we use `filters=(64, 64, 256)`
    and `kernel_sizes=(1, 3, 1)` and `n_blocks=3` (like for ResNet50), then the structure
    of this stack is as follows:
    
    [conv64-1x1, conv64-3x3, conv256-1x1, conv64-1x1, conv64-3x3, conv256-1x1, conv64-1x1, conv64-3x3, conv256-1x1]
    
    There's a skip connection after each block, which contains a Conv2D operation
    if the height or width does not match between the shortcut (`sc`) and current
    feature map or when the numfer of filters doesn't match.
   
    Parameters
    ----------
    filters : list
        List of number of filters per block
    kernel_sizes : list
        Kernel sizes per block (assumed to be square)
    n_blocks : int
        Number of blocks (repetitions)
    bn_momentum : float
        Momentum used for batch norm layers
    layer_nr : int
        Current layer number (used in name only)
    stage : int
        Current stage

    Returns
    -------
    apply : function
        Function that applies the convolutional block
    """

    if len(kernel_sizes) != len(filters):
        raise ValueError("Nr of kernel sizes should be equal to nr of filters!")

    def apply(x, layer_nr=layer_nr):
        """ Applies the block to the input x. """

        for block_nr in range(n_blocks):

            sc = x  # always define the shortcut at the start of the block
        
            for i, (nf, ks) in enumerate(zip(filters, kernel_sizes)):
                
                if block_nr == 0 and i == 0:
                    if stage == 1:
                        stride = 1
                    else:
                        stride = 2
                else:
                    stride = 1           

                name = f'layer{layer_nr}_block-{block_nr+1}_stage-{stage}'
                x = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')(x)
                x = Activation('relu', name=f'{name}_relu')(x)
                x = Conv2D(nf, ks, stride, padding='same', kernel_initializer='he_normal',
                           name=f'{name}_conv')(x)
                layer_nr += 1

            mismatch_nf = sc.shape[-1] != nf
            mismatch_hw = block_nr == 0 and stage != 1
            
            if mismatch_nf or mismatch_hw:
                if mismatch_hw:
                    stride = 2
                else:
                    stride = 1

                sc = Conv2D(nf, 1, stride, padding='same',
                            kernel_initializer='he_normal',
                            name=f'{name}_conv_shortcut')(sc)

            x = Add(name=f'{name}_add')([x, sc])

        return x

    if as_model:
        inp = Input(input_shape)
        x = apply(inp, layer_nr=layer_nr)
        return Model(inp, x, name='ResBlock')
    else:
        return apply


def Stem(bn_momentum=0.01, as_model=False, input_shape=None):
    
    def apply(s):
        
        # ResNet initial block: Conv block + Maxpool
        x = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
                   kernel_initializer='he_normal',
                   name='layer1_conv')(s)
        x = BatchNormalization(momentum=bn_momentum, name='layer1_bnorm')(x)
        x = Activation('relu', name='layer1_relu')(x)
        x = MaxPooling2D(3, 2, padding='same', name='layer1_pool')(x)

        return x
    
    if as_model:
        inp = Input(input_shape, name='input_stem')
        x = apply(inp)
        return Model(inp, x, name='model_stem')
    else:
        return apply


def ResNet6(input_shape=(224, 224, 3), bn_momentum=0.01, blocks_only=False):
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=4, stage=2)(x)
    
    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet6')

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


def ResNet10(input_shape=(224, 224, 3), bn_momentum=0.9, blocks_only=False):
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=4, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=6, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=8, stage=4)(x)

    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet10')

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer9_bnorm')(x)
    x = Activation('relu', name='layer9_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer9_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet10')    
    return model


def ResNet18(input_shape=(224, 224, 3), bn_momentum=0.01, blocks_only=False):
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64, 64), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128, 128), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=6, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256, 256), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=10, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512, 512), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=14, stage=4)(x)

    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet10')

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer17_bnorm')(x)
    x = Activation('relu', name='layer17_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer17_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet18')    
    
    return model


def ResNet34(input_shape=(224, 224, 3), bn_momentum=0.01, blocks_only=False):
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64, 64), (3, 3), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128, 128), (3, 3), n_blocks=4, bn_momentum=bn_momentum, layer_nr=8, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256, 256), (3, 3), n_blocks=6, bn_momentum=bn_momentum, layer_nr=16, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512, 512), (3, 3), n_blocks=3, bn_momentum=bn_momentum, layer_nr=28, stage=4)(x)

    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet34')

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer33_bnorm')(x)
    x = Activation('relu', name='layer33_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer33_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet34')    
       
    return model
    

def ResNet50(input_shape=(224, 224, 3), bn_momentum=0.01, blocks_only=False):
    """ ResNet50 model.
    
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64, 64, 256), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128, 128, 512), (1, 3, 1), n_blocks=4, bn_momentum=bn_momentum, layer_nr=11, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256, 256, 1024), (1, 3, 1), n_blocks=6, bn_momentum=bn_momentum, layer_nr=23, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512, 512, 2048), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=41, stage=4)(x)

    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet50')

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer49_bnorm')(x)
    x = Activation('relu', name='layer49_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer49_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet50')    
       
    return model


def ResNet101(input_shape=(224, 224, 3), bn_momentum=0.01, blocks_only=False):
    """ ResNet101 model.
    
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
    x = Stem()(s)

    # Stage 1
    x = ResBlock((64, 64, 256), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock((128, 128, 512), (1, 3, 1), n_blocks=4, bn_momentum=bn_momentum, layer_nr=11, stage=2)(x)
    
    # Stage 3
    x = ResBlock((256, 256, 1024), (1, 3, 1), n_blocks=23, bn_momentum=bn_momentum, layer_nr=23, stage=3)(x)
    
    # Stage 4
    x = ResBlock((512, 512, 2048), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=92, stage=4)(x)

    if blocks_only:
        return Model(inputs=s, outputs=x, name='ResNet101')

    # Post-activation
    x = BatchNormalization(momentum=bn_momentum, name='layer100_bnorm')(x)
    x = Activation('relu', name='layer100_relu')(x)

    # Average feature maps per filter (resulting in 512 values)
    x = GlobalAveragePooling2D(name='layer100_globalpool')(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet101')    
       
    return model


if __name__ == '__main__':

    # Initialize and compile model
    model = ResNet6(input_shape=(112, 112, 3))
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