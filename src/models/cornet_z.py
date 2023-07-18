from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D


def CORnet_Z(input_shape=(224, 224, 3)):
    """ CORnet_Z model (Kubilius et al., 2018, BioRxiv, 
    https://doi.org/10.1101/408385).

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
    
    # V1: 64 filters, kernel size 7x7, stride 2, glorot uniform init (default)
    # 224 x 224 (conv) -> 112 x 112 (maxpool) -> 56 x 56 
    x = Conv2D(64, 7, 2, activation='relu', padding='same', name='layer1_conv')(s)
    x = MaxPooling2D(3, strides=2, padding='same', name='layer1_pool')(x)

    # V2: 128 filters, kernel size 3x3, stride 1
    # 56 x 56 (conv) -> 56 x 56 (maxpool) -> 28 x 28
    x = Conv2D(128, 3, 1, activation='relu', padding='same', name='layer2_conv')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='layer2_pool')(x)

    # V4: 256 filters, kernel size 3x3, stride 1
    # 28 x 28 (conv) -> 28 x 28 (maxpool) -> 14 x 14
    x = Conv2D(256, 3, 1, activation='relu', padding='same', name='layer3_conv')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='layer3_pool')(x)
    
    # IT: 512 filters, kernel size 3x3
    # 14 x 14 (conv) -> 14 x 14 (maxpool) -> 7 x 7
    x = Conv2D(512, 3, 1, activation='relu', padding='same', name='layer4_conv')(x)
    x = MaxPooling2D(3, strides=2, padding='same', name='layer4_pool')(x)
    
    # Decoder: average per channel, dense
    # 7 x 7 x 512 (globalavpool) -> 512 -> C
    x = GlobalAveragePooling2D(name='layer4_globalpool')(x)
    
    # Create model (no head; use `src.models.utils.add_head` for this)
    model = Model(inputs=s, outputs=x, name='CORnet-Z')
    return model
