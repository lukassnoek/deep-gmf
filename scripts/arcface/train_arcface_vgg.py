import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.data import Dataset
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

sys.path.append('.')
from src.models.arcface import ArcFaceModel


### Parameters
batch_size = 64#512
n_val = 128#2**15
shuffle_buffer = 128#1024

def _get_class(f):
    substr = tf.strings.split(f, '/')[-2]
    substr = tf.strings.substr(substr, 1, 6)
    return tf.strings.to_number(substr, tf.int32)

def _load_image(f):
    img = tf.io.read_file(f)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.random_crop(img, size=(112, 112, 3))
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img

fn = list(Path('/analyse/Project0257/lukas/data/VGGFACE2_Cleandata/train').glob('n00000?'))
classes = np.unique([int(str(f.stem)[1:]) for f in fn])
n_classes = classes.size

dset = Dataset.list_files('/analyse/Project0257/lukas/data/VGGFACE2_Cleandata/train/n00000?/*.jpg')
dset = dset.shuffle(len(fn), reshuffle_each_iteration=False)  # shuffle once
X = dset.map(_load_image, num_parallel_calls=10)  # get images

# Infer classes and map to integers 0, 1, ... `n_classes`
raw_classes = dset.map(_get_class)
ilu = IntegerLookup(vocabulary=classes, num_oov_indices=0, output_mode='int')
labels = raw_classes.map(ilu)

# Get one-hot encoded version
Y = labels.map(lambda l: tf.one_hot(l, depth=n_classes))
dset = Dataset.zip(((X, labels), Y))

# Split in train and validation set; only shuffle train every time
val = dset.take(n_val)
val = val.batch(batch_size)
val = val.prefetch(1)
train = dset.skip(n_val)
train = train.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
train = train.batch(batch_size, drop_remainder=True)
train = train.prefetch(1)

n_batch_per_epoch = train.cardinality().numpy()
n_epochs = int(np.ceil(32_000 / n_batch_per_epoch))

# Same opt parameters as original arcface model
model = ArcFaceModel(num_classes=n_classes, bn_momentum=0.9)
lr = 0.1#PiecewiseConstantDecay(boundaries=[10_000, 20_000, 28_000], values=[0.001, 0.01, 0.001, 0.0001])
opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
model.fit(train, validation_data=val, epochs=n_epochs)
