import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import ResNet50


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""

    def apply(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=tf.keras.regularizers.l2(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    
    return apply


def ArcFaceModel(size=None, channels=3, name='arcface_model',
                 embd_shape=512, w_decay=5e-4):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = ResNet50(input_shape=x.shape[1:], include_top=False,
                 weights=None)(x)
    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)
    model = Model(inputs, embds, name=name)
    ckpt_path = tf.train.latest_checkpoint('./pretrained_models/arc_res50')
    model.load_weights(ckpt_path)
    return model