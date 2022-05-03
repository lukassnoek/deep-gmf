import tensorflow as tf
from coral_ordinal import CornOrdinal
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Lambda
from tensorflow.keras.layers import MaxPooling2D, Dense, Subtract


def StereoNet(input_shape=(128, 128, 1), split_hemi=False, target='shape'):

    img_l = Input(input_shape, name='layer0_input-left')
    img_r = Input(input_shape, name='layer0_input-right')
    
    #retina = Conv2D(32, 3, 2, padding='same', activation='relu', name='retina')
    #x_l = retina(img_l)
    #x_r = retina(img_r)    
    #x = Concatenate(axis=-1)([x_l, x_r])
    #x = Conv2D(32, 1, 1, padding='same', activation='relu')(x)
    x = Subtract()([img_l, img_r])

    shared = Sequential([
        MaxPooling2D(),
        Conv2D(64, 3, 2, padding='same', activation='relu'),
        Conv2D(64, 3, 2, padding='same', activation='relu'),
        Conv2D(128, 3, 2, padding='same', activation='relu'),
    ])

    if split_hemi:
        x_l, x_r = Lambda(tf.split, arguments={'num_or_size_splits': 2, 'axis': 1})(x)
        x_l = shared(x_l)
        x_r = shared(x_r)
        x = Concatenate(axis=1)([x_l, x_r])
    else:
        x = shared(x)
    
    x = Flatten()(x)
    if target == 'shape':
        y = Dense(units=2, activation='softmax')(x)
    elif target == 'disparity':
        y = CornOrdinal(num_classes=10)(x)#Dense(units=1, activation=None)(x)
    elif target == 'radius':
        y = Dense(units=1, activation=None)(x)
    
    return Model(inputs=[img_l, img_r], outputs=y, name='test')


if __name__ == '__main__':
    mod = StereoNet()
    print(mod.summary())