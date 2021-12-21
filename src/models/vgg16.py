# https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def ConvBlock(filters, n_conv=2, kernel_size=3, block=1):
    
    def apply(x):
        
        for i in range(1, n_conv+1):
            x = Conv2D(filters, kernel_size, activation='relu', padding='same', name=f'conv{block}_{i}')(x)
    
        x = MaxPooling2D(2, strides=2, name=f'pool{block}')(x)

        return x
    
    return apply


def VGG16(input_shape=(256, 256, 3), n_classes=4):
    s = Input(input_shape, name='input0')  # s = stimulus

    # filters_ = [64, 128, 256, 512, 512]
    # n_conv_ = [2, 2, 3, 3, 3]
    # for block, (filters, n_conv) in enumerate(zip(filters_, n_conv_)):
    #     if block == 0:
    #         x = ConvBlock(filters, n_conv, 3, block=block+1)(s)
    #     else:
    #         x = ConvBlock(filters, n_conv, 3, block=block+1)(x)

    # # Classification block
    x = Conv2D(64, 3, padding='same', activation='relu')(s)
    x = Flatten()(x)
    # #x = GlobalAveragePooling2D()(x)
    # x = Dense(units=4096, activation='relu', name='fc6')(x)
    # x = Dense(units=4096, activation='relu', name='fc7')(x)
    y = Dense(units=n_classes, activation='softmax', name='fc8')(x)
    model = Model(inputs=s, outputs=y, name='VGG16')
    return model


if __name__ == '__main__':
    
    model = VGG16(n_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())