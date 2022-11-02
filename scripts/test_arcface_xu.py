import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization


# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """Make trainable=False freeze BN for real (the og version is sad).
#        ref: https://github.com/zzh8829/yolov3-tf2
#     """
#     def call(self, x, training=False):
#         if training is None:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=tf.keras.regularizers.l2(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    
    return output_layer


def ArcFaceModel(size=(112, 112, 3), num_classes=None, margin=0.5, logist_scale=64,
                 embd_shape=512, w_decay=5e-4, use_pretrain=False):
    """Arc Face Model"""
    
    weights = 'imagenet' if use_pretrain else None
    backbone = ResNet50(input_shape=size, include_top=False, weights=weights)
    output_layer = OutputLayer(embd_shape, w_decay=w_decay)
    arc_head = ArcHead(num_classes=num_classes, margin=margin, logist_scale=logist_scale)

    x = inputs = Input(size, name='input_image')
    labels = Input([], name='label')

    x = backbone(x)
    x = output_layer(x)
    logist = arc_head(x, labels)
    y = Softmax()(logist)

    return Model([inputs, labels], y, name='arcface_model')


NUM_CLASSES = 6144
model = ArcFaceModel(size=(112, 112, 3), num_classes=NUM_CLASSES, embd_shape=512, w_decay=5e-4, use_pretrain=False)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')

import sys
sys.path.append('.')
from src.io import create_dataset
import pandas as pd

info = pd.read_csv(f'/analyse/Project0257/lukas/data/gmf_112x112.csv')

train, val = create_dataset(info, Y_col='id', target_size=(112, 112, 3),
                            batch_size=1024, n_id_train=NUM_CLASSES, arcface=True)

model.fit(train, epochs=50)
