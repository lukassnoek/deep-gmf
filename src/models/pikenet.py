from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D


def PikeNet(input_shape=(224, 224, 3), n_classes=4):
    """ PikeNet.
    
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
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(s)
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    x = GlobalAveragePooling2D()(x)
    y = Dense(units=n_classes, activation='softmax')(x)
    
    model = Model(inputs=s, outputs=y, name='PikeNet')
    return model


if __name__ == '__main__':

    model = PikeNet(n_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())