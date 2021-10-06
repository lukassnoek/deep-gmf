from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D


def CORnet_Z(input_shape=(224, 224, 3), n_classes=4):
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

    s = Input(input_shape)  # s = stimulus
    
    # V1: 64 filters, kernel size 7x7, stride 2, glorot uniform init (default)
    # 224 x 224 (conv) -> 112 x 112 (maxpool) -> 56 x 56 
    x = Conv2D(64, 7, 2, activation='relu', padding='same')(s)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # V2: 128 filters, kernel size 3x3, stride 1
    # 56 x 56 (conv) -> 56 x 56 (maxpool) -> 28 x 28
    x = Conv2D(128, 3, 1, activation='relu', padding='same')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # V4: 256 filters, kernel size 3x3, stride 1
    # 28 x 28 (conv) -> 28 x 28 (maxpool) -> 14 x 14
    x = Conv2D(256, 3, 1, activation='relu', padding='same')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # IT: 512 filters, kernel size 3x3
    # 14 x 14 (conv) -> 14 x 14 (maxpool) -> 7 x 7
    x = Conv2D(512, 3, 1, activation='relu', padding='same')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Decoder: average per channel, dense
    # 7 x 7 x 512 (globalavpool) -> 512 -> C
    x = GlobalAveragePooling2D()(x)
    if n_classes == 2:
        y = Dense(1, activation='sigmoid', name='fc')(x)
    else:
        y = Dense(n_classes, activation='softmax', name='fc')(x)

    # Create model
    model = Model(inputs=s, outputs=y, name='CORnet-Z')
    return model


if __name__ == '__main__':

    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

    ### Create random data
    N, C = 1028, 4
    batch_size = 256  # original cornet paper

    # Simulate some data from a linear model, Y = argmax(XW)
    X = np.random.normal(0, 1, size=(N, 224, 224, 3)).astype('float32')
    W = np.random.normal(0, 0.005, size=(C, 224, 224, 3)).astype('float32')
    Z = X.reshape((N, 224 * 224 * 3)) @ W.reshape((224 * 224 * 3, C))
    Y = to_categorical(Z.argmax(axis=1))

    ### Create model, compile and fit
    model = CORnet_Z()

    # LR schedule: divide lr by 10 every 10 epochs
    decay_steps = int(round((N / batch_size) * 10))
    lr_schedule = ExponentialDecay(0.01, decay_steps=decay_steps,
                                   decay_rate=0.1, staircase=True)
    opt = SGD(learning_rate=lr_schedule, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
    model.fit(X, Y, batch_size=256, epochs=25, validation_split=0.2)