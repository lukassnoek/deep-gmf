"""Implementation of different ResNet (v2) architectures."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten


def ResBlock(input_channels, filters, kernel_sizes=(3, 3), n_blocks=2, bn_momentum=0.9, layer_nr=1, stage=1):
    """ Stack of ResNet (v2!) blocks with Conv2D layers and skip connections.

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

    # First define all layers such that they're not defined twice when calling the
    # returned function multiple times (which causes errors)
    layers = {}
    for block_nr in range(n_blocks):

        layers[block_nr] = {}

        for i, (nf, ks) in enumerate(zip(filters, kernel_sizes)):
            layers[block_nr]['main'] = []

            if block_nr == 0 and i == 0:
                if stage == 1:
                    stride = 1
                else:
                    stride = 2
            else:
                stride = 1           

            name = f'layer{layer_nr}_block-{block_nr+1}_stage-{stage}'
            bn = BatchNormalization(momentum=bn_momentum, name=f'{name}_bnorm')
            act = Activation('relu', name=f'{name}_relu')
            conv = Conv2D(nf, ks, stride, padding='same', kernel_initializer='he_normal',
                          name=f'{name}_conv')

            layers[block_nr]['main'].append((bn, act, conv))
            layer_nr += 1

        mismatch_nf = input_channels != nf
        mismatch_hw = block_nr == 0 and stage != 1
        
        if mismatch_nf or mismatch_hw:
            # Add a convolution to the shortcut if the number of filters or height/width
            # does not match between the current data (x) and the shortcut (sc)
            if mismatch_hw:
                # For downsampling height/width
                stride = 2
            else:
                # For downsampling number of channels
                stride = 1

            conv_sc = Conv2D(nf, 1, stride, padding='same',
                             kernel_initializer='he_normal',
                             name=f'{name}_conv_shortcut')
            
            layers[block_nr]['shortcut'] = conv_sc

        layers[block_nr]['add'] = Add(name=f'{name}_add')

    def apply(x):
        """Applies the layers (`layers` dictionary) to the input `x`, which might be
        a single input or multiple inputs (for `StereoResNet` models)."""

        for block_nr in range(n_blocks):
            sc = x
            for layer in layers[block_nr]['main']:
                for operation in layer:
                    x = operation(x)    

            if 'shortcut' in layers[block_nr]:
                sc = layers[block_nr]['shortcut'](sc)

            x = layers[block_nr]['add']([x, sc])

        return x

    return apply


def Stem():
    """ResNet "stem" set of layers, to be applied before the ResNet blocks.
    
    Returns
    -------
    apply : function
        Function that applies the stem layers
    """
    
    conv = Conv2D(64, 7, 2, padding='same',  # 64 filters, kernel size 7, stride 2
                   kernel_initializer='he_normal',
                   name='layer1_conv')
    mp = MaxPooling2D(3, 2, padding='same', name='layer1_pool')
    
    def apply(s):
        
        # ResNet initial block: Conv block + Maxpool
        x = mp(conv(s))

        return x

    return apply


def PostActivation(layer_nr, bn_momentum=0.9):
    """Post-activation layers, to be applied after the ResNet blocks.
    
    Parameters
    ----------
    layer_nr : int
        Current layer number (used in name only)
    bn_momentum : float
        Momentum used for batch norm layers

    Returns
    -------
    apply : function
        Function that applies the post-activation layers
    """
    bn = BatchNormalization(momentum=bn_momentum, name=f'layer{layer_nr}_bnorm')
    act = Activation('relu', name=f'layer{layer_nr}_relu')
    gap = GlobalAveragePooling2D(name=f'layer{layer_nr}_globalpool')

    def apply(x):
        x = gap(act(bn(x)))
        return x

    return apply


def ResNet6(input_shape=(112, 112, 3), bn_momentum=0.9):
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

    Notes
    -----
    I changed the number of filters in the first two blocks from 64 to 128 and 128 to 256
    to increase the model's representational power, which helps in training models on
    data with lots of different face identities (in my experience)
    """    
    
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    x = Stem()(s)
    
    # Stage 1 (Lukas: changed 64 filters to 128)
    x = ResBlock(64, (128,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2 (Lukas: changed 128 filters to 256)
    x = ResBlock(128, (256,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=4, stage=2)(x)

    # Post-activation
    x = PostActivation(layer_nr=6)(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet6')

    return model


def ResNet10(input_shape=(112, 112, 3), bn_momentum=0.9):
    """ ResNet10 model.
    
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
    x = ResBlock(64, (64,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock(64, (128,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=4, stage=2)(x)
    
    # Stage 3
    x = ResBlock(128, (256,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=6, stage=3)(x)
    
    # Stage 4
    x = ResBlock(256, (512,), (3,), n_blocks=2, bn_momentum=bn_momentum, layer_nr=8, stage=4)(x)

    # Post-activation
    x = PostActivation(layer_nr=10)(x)
    
    # There is no classification top/head by default!
    # Note: ends at layer 5, because dense layer (to be added)
    # represents layer 6
    model = Model(inputs=s, outputs=x, name='ResNet10')    
    return model


def ResNet18(input_shape=(224, 224, 3), bn_momentum=0.9):
    """ ResNet18 model.
    
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
    x = Stem(b)(s)

    # Stage 1
    x = ResBlock(64, (64, 64), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock(64, (128, 128), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=6, stage=2)(x)
    
    # Stage 3
    x = ResBlock(128, (256, 256), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=10, stage=3)(x)
    
    # Stage 4
    x = ResBlock(256, (512, 512), (3, 3), n_blocks=2, bn_momentum=bn_momentum, layer_nr=14, stage=4)(x)

    # Post-activation
    x = PostActivation(layer_nr=18)(x)
    
    model = Model(inputs=s, outputs=x, name='ResNet18')    
    
    return model


def ResNet34(input_shape=(112, 112, 3), bn_momentum=0.9):
    """ ResNet34 model.
    
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
    x = ResBlock(64, (64, 64), (3, 3), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock(64, (128, 128), (3, 3), n_blocks=4, bn_momentum=bn_momentum, layer_nr=8, stage=2)(x)
    
    # Stage 3
    x = ResBlock(128, (256, 256), (3, 3), n_blocks=6, bn_momentum=bn_momentum, layer_nr=16, stage=3)(x)
    
    # Stage 4
    x = ResBlock(256, (512, 512), (3, 3), n_blocks=3, bn_momentum=bn_momentum, layer_nr=28, stage=4)(x)
    
    # Post-activation
    x = PostActivation(layer_nr=34)(x)
    model = Model(inputs=s, outputs=x, name='ResNet34')    
       
    return model
    

def ResNet50(input_shape=(112, 112, 3), bn_momentum=0.9):
    """ ResNet50 model.
    
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
    x = ResBlock(64, (64, 64, 256), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock(256, (128, 128, 512), (1, 3, 1), n_blocks=4, bn_momentum=bn_momentum, layer_nr=11, stage=2)(x)
    
    # Stage 3
    x = ResBlock(1024, (256, 256, 1024), (1, 3, 1), n_blocks=6, bn_momentum=bn_momentum, layer_nr=23, stage=3)(x)
    
    # Stage 4
    x = ResBlock(2048, (512, 512, 2048), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=41, stage=4)(x)

    # Post-activation
    x = PostActivation(layer_nr=50)(x)

    model = Model(inputs=s, outputs=x, name='ResNet50')    
       
    return model


def ResNet101(input_shape=(224, 224, 3), bn_momentum=0.9):
    """ ResNet101 model.
    
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
    x = ResBlock(64, (64, 64, 256), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=2, stage=1)(x)
    
    # Stage 2
    x = ResBlock(256, (128, 128, 512), (1, 3, 1), n_blocks=4, bn_momentum=bn_momentum, layer_nr=11, stage=2)(x)
    
    # Stage 3
    x = ResBlock(512, (256, 256, 1024), (1, 3, 1), n_blocks=23, bn_momentum=bn_momentum, layer_nr=23, stage=3)(x)
    
    # Stage 4
    x = ResBlock(1024, (512, 512, 2048), (1, 3, 1), n_blocks=3, bn_momentum=bn_momentum, layer_nr=92, stage=4)(x)

    # Post-activation
    x = PostActivation(layer_nr=101)(x)

    model = Model(inputs=s, outputs=x, name='ResNet101')    
       
    return model


if __name__ == '__main__':

    # Initialize and compile model
    model = ResNet6(input_shape=(112, 112, 3))
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
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
    # )tgy6