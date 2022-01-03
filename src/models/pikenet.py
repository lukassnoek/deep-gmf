from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D


def PikeNet(input_shape=(256, 256, 3), filter_mult=2):
    """ PikeNet.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple with length 3 (x, y, channels)
    filter_mult : int
        Multiplier for number of (default) filters

    Returns
    -------
    model : Model
        Keras model
    """    

    s = Input(input_shape, name='layer-0_input')  # s = stimulus
    x = Conv2D(16 * filter_mult, 7, strides=2, padding='same', activation='relu', name='layer-1_conv')(s)
    x = Conv2D(16 * filter_mult, 7, strides=2, padding='same', activation='relu', name='layer-2_conv')(x)
    x = Conv2D(32 * filter_mult, 5, strides=2, padding='same', activation='relu', name='layer-3_conv')(x)
    x = Conv2D(32 * filter_mult, 5, strides=2, padding='same', activation='relu', name='layer-4_conv')(x)
    x = Conv2D(64 * filter_mult, 3, strides=2, padding='same', activation='relu', name='layer-5_conv')(x)
    x = Conv2D(64 * filter_mult, 3, strides=2, padding='same', activation='relu', name='layer-6_conv')(x)
    x = Conv2D(128 * filter_mult, 1, strides=2, padding='same', activation='relu', name='layer-7_conv')(x)
    x = Conv2D(128 * filter_mult, 1, strides=2, padding='same', activation='relu', name='layer-8_conv')(x)
    x = Flatten(name='layer-8_flatten')(x)
    #x = GlobalAveragePooling2D(name='layer-7_globalpool')(x)
    model = Model(inputs=s, outputs=x, name='PikeNet')

    return model


if __name__ == '__main__':

    model = PikeNet(filter_mult=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())