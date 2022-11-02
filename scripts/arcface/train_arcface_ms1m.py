import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.data import Dataset
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

sys.path.append('.')
from src.models.arcface import ArcFace

### Parameters
batch_size = 512
n_val = 2**15
shuffle_buffer = 1024

dirs = list(Path('/analyse/Project0257/lukas/data/ms1m_align_112').glob('*'))
classes = np.unique([int(d.stem) for d in dirs if str(d.stem).isdigit()])
n_classes = classes.size
print(f"Working on {n_classes} classes!")

def _load_image(f):

    X = tf.io.read_file(f)
    X = tf.io.decode_jpeg(X, channels=3)
    X = tf.image.random_flip_left_right(X)
    X = tf.cast(X, tf.float32)
    X = X / 255.

    return X  

def _get_class(f):
    # Assuming that the class label is in the penultimate subfolder
    substr = tf.strings.split(f, '/')[-2]
    return tf.strings.to_number(substr, tf.int32)

# Set shuffle to False, because otherwise it will shuffle *every time* you call `map``
dset = Dataset.list_files('/analyse/Project0257/lukas/data/ms1m_align_112/*/*.jpg', shuffle=False)
dset = dset.shuffle(dset.cardinality(), reshuffle_each_iteration=False)  # shuffle once
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
n_epochs = 32#int(np.ceil(180_000 / n_batch_per_epoch))

# Same opt parameters as original arcface model
model = ArcFace(num_classes=n_classes, bn_momentum=0.9)
#lr = PiecewiseConstantDecay(boundaries=[100_000, 160_000], values=[0.1, 0.01, 0.001])
lr = 0.01
opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
history = model.fit(train, validation_data=val, epochs=n_epochs)
Path(f'trained_models/ResNet50_dataset-ms1m_loss-arcface').mkdir(exist_ok=True)
model.save(f'trained_models/ResNet50_dataset-ms1m_loss-arcface/epoch-{n_epochs:03}')
pd.DataFrame(history.history).to_csv('trained_models/Resnet51_dataset-ms1m_loss-arcface/history.csv', index=False)
# sys.path.append('/home/lukass/arcface-tf2')
# from modules.models import ArcFaceModel
# model = ArcFaceModel(size=112,
#                      num_classes=85742,
#                      backbone_type='ResNet50',
#                      training=True)
# ckpt_path = tf.train.latest_checkpoint('/home/lukass/arcface-tf2/checkpoints/arc_res50')
# model.load_weights(ckpt_path)
# opt = tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
# model.fit(train)
