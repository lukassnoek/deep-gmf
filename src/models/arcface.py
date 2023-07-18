import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Softmax, GlobalAveragePooling2D

from . import resnet
from ..layers import ArcMarginPenaltyLogits


def ArcFace(input_shape=(112, 112, 3), backbone_name='ResNet10', num_classes=32, n_embed=512, w_decay=5e-4, bn_momentum=0.9):
    """ArcFace Model, based on an implementation by Kuan-Yu Huang
    (https://github.com/peteryuX/arcface-tf2).
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input images, by default (112, 112, 3)
    backbone_name : str
        Name of the backbone network, by default 'ResNet10' (from models.resnet)
    num_classes : int
        Number of classes, by default 32
    n_embed : int
        Dimensionality of the embedding space, by default 512
    w_decay : float
        Weight decay, by default 5e-4
    bn_momentum : float
        Batch normalization momentum, by default 0.9
    """

    # Actually uses labels in arcmarginpenaltylogits
    s = Input(input_shape, name='layer0_input')  # s = stimulus
    labels = Input([], name='layer0_labels')

    backbone = getattr(resnet, backbone_name)
    x = backbone(input_shape, bn_momentum, blocks_only=True)(s)
    #from tensorflow.keras.applications import ResNet50
    #x = ResNet50(input_shape=s.shape[1:], include_top=False, weights='imagenet')(s)
    
    x = BatchNormalization(momentum=bn_momentum)(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(n_embed, kernel_regularizer=tf.keras.regularizers.l2(w_decay))(x)
    x = BatchNormalization(momentum=bn_momentum)(x)
    x = ArcMarginPenaltyLogits(num_classes=num_classes)(x, labels)  # logits
    y = Softmax(name='id')(x)
    #x = GlobalAveragePooling2D()(x)
    #y = Dense(num_classes, activation='softmax', name='id')(x)
    
    model = Model([s, labels], y, name=f'arcface_{backbone_name}')
    return model