import tensorflow_datasets as tfds

ds = tfds.load('mnist', split=['train', 'validation'], shuffle_files=True)