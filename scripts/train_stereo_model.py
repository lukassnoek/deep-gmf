import sys
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import StringLookup
from tensorflow.keras.optimizers import Adam
from coral_ordinal import OrdinalCrossEntropy, MeanAbsoluteErrorLabels

sys.path.append('.')
from src.models import StereoNet
from scripts.generate_stimuli.old.random_dot_stereogram import rds

N = 2**14
N_val = 512
BS = 512
TARGET = 'radius'

rds_params = {
    'radius': np.random.randint(5, 26, size=N) * 2,
    'shape': np.random.choice(['square', 'circle'], size=N),
    'disparity': np.random.randint(1, 11, size=N),
    'x': np.random.randint(-50, 51, size=N),
    'y': np.random.randint(-50, 51, size=N)
}
dset_params = Dataset.from_tensor_slices(rds_params)
dset_img = dset_params.map(lambda row: tf.py_function(
        rds, inp=[row['radius'], row['shape'], row['disparity'], row['x'], row['y']],
        Tout=(tf.int8, tf.int8)
    ), num_parallel_calls=tf.data.AUTOTUNE
)

if TARGET == 'shape':
    slu = StringLookup(output_mode='one_hot', num_oov_indices=0)
    slu.adapt(rds_params['shape'])
    dset_target = dset_params.map(lambda x: slu(x['shape']),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    loss = 'binary_crossentropy'
    metric = 'accuracy'
elif TARGET == 'radius':
    dset_target = dset_params.map(lambda x: tf.cast(x['radius'], tf.float32))
    loss = 'mse'
    metric = 'cosine_similarity'
else:
    dset_target = dset_params.map(lambda x: tf.cast(x['disparity'], tf.float32))
    loss = OrdinalCrossEntropy(),
    metric = [MeanAbsoluteErrorLabels()]
    #loss = 'mse'
    #metric = 'cosine_similarity'

dset = Dataset.zip((dset_img, dset_target))
val = dset.take(N_val)
val = val.batch(BS).prefetch(buffer_size=tf.data.AUTOTUNE)
train = dset.skip(N_val)
train = train.batch(BS).prefetch(buffer_size=tf.data.AUTOTUNE)

model = StereoNet(target=TARGET)
model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=metric)
model.fit(train, epochs=100, validation_data=val)

print(model.predict(val))