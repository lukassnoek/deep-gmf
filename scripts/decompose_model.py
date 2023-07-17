"""Extracts N batches of activations from a trained model and stores them in a single
array (after PCA compression) per layer; separately for the training, validation and
testing splits."""

# Set maximum number of CPUs to 10 otherwise will use all available CPUs on the server
import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['NUMEXPR_MAX_THREADS'] = '10'

import sys
import random
import click
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

from tqdm import tqdm
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.io import create_test_dataset
from src.models.utils import loop_over_layers


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--batch-size', default=1024)
@click.option('--n-id-train', type=click.INT, default=None)
@click.option('--n-id-val', type=click.INT, default=None)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--n-per-id', type=click.INT, default=None)
@click.option('--gpu-id', default=1)
def main(model_path, batch_size, n_id_train, n_id_val, n_id_test, n_per_id, gpu_id):
    """ Decomposes model activations for `n_samples` of stimuli into
    a (optionally PCA-compressed) single array of activations per layer,
    in order to speed up the decoding analyses. """

    model_path = Path(model_path)
    # Infer dataset the model was trained on from `model_path``
    model_name = str(model_path.parent.name)
    dataset = model_name.split('dataset-')[1].split('_target')[0]

    # Load model and filter layers
    model = load_model(model_path)
    factors = ['id', 'ethn', 'gender', 'age', 'bg', 'emo', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']

    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')

    # Open hdf5 file to store PCA params and some meta-data
    f_out_path = f'{model_path}_decomposed.h5'
    comp_params = h5py.File(f'{str(model_path)}_compressed.h5', 'r')

    f_out = h5py.File(f_out_path, 'w')
    binocular = 'StereoResNet' in str(model_path)

    for split, n in [('training', n_id_train), ('validation', n_id_val), ('testing', n_id_test)]:

        if n == 0:
            continue

        info_ = info.query("split == @split")
        if n is not None:
            ids = info_['id'].unique().tolist()
            ids = random.sample(ids, n_id_test)
            info_ = info_.query('id in @ids')

        if n_per_id is not None:
            info_ = info_.groupby('id').sample(n=n_per_id)

        n_samples = info_.shape[0]
        print(f"Split: {split}, total samples: {n_samples}")

        if n_samples % batch_size != 0:
            raise ValueError("`n_samples` is not a multiple of `batch_size`!")

        h5_split = f_out.create_group(split)

        with tf.device(f'/gpu:{gpu_id}'):

            # Get a single batch of stimuli
            dataset_tf = create_test_dataset(info_, batch_size=batch_size, n_shape=100, n_tex=100,
                                             binocular=binocular)

            for fac in factors:
                h5_split.create_dataset(fac, data=info_.loc[:, fac].to_numpy())
 
            shape, tex = [], []
            a_N_all = defaultdict(list)
            n_batches = n_samples // batch_size
            h5_a_N = h5_split.create_group('a_N')
            
            for X, shape_, tex_ in tqdm(dataset_tf, total=n_batches):
                shape.append(shape_.numpy())
                tex.append(tex_.numpy())
            
                for layer in loop_over_layers(model):
                    
                    # Zero-pad layer name
                    layer_nr = int(layer.name.split('layer')[1].split('_')[0])
                    layer_name = layer.name.replace(f'layer{layer_nr}', f'layer{layer_nr:02d}')

                    # Note to self: predict_step avoids warning that you get
                    # when calling predict (but does the same)
                    extractor = Model(inputs=model.inputs, outputs=[layer.output])
                    a_N = extractor.predict(X, verbose=0)

                    # Reshape to obs x features array (and cast to numpy)
                    a_N = np.reshape(a_N, (a_N.shape[0], -1))

                    # apply PCA transformation
                    mu = comp_params[layer.name]['mu'][:]
                    W = comp_params[layer.name]['W'][:]
                    a_N = (a_N - mu) @ W  # n_samples x 500
                    a_N_all[layer_name].append(a_N)

            for layer_name, a_N in a_N_all.items():
                h5_a_N.create_dataset(layer_name, data=np.vstack(a_N))

            h5_split.create_dataset('shape', data=np.vstack(shape))
            h5_split.create_dataset('tex', data=np.vstack(tex))
            
    comp_params.close()        
    f_out.close()

if __name__ == '__main__':
    main()
