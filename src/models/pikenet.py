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

    s = Input(input_shape, name='input0')  # s = stimulus
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu', name='conv1')(s)
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv3')(x)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv4')(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv5')(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv6')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu', name='conv7')(x)

    x = GlobalAveragePooling2D(name='globalpooling8')(x)
    y = Dense(units=n_classes, activation='softmax', name='dense9')(x)
    
    model = Model(inputs=s, outputs=y, name='PikeNet')
    return model


if __name__ == '__main__':

    model = PikeNet(n_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())