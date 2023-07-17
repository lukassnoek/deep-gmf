"""Compression of model representations using PCA."""

# Set maximum number of CPUs to 10 otherwise will use all available CPUs on the server
import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['NUMEXPR_MAX_THREADS'] = '10'

import sys
import click
import h5py
import pandas as pd
import tensorflow as tf
from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.io import create_dataset
from src.models.embedding import TFIncrementalPCA
from src.models.utils import loop_over_layers


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--n-comp', default=2048)
@click.option('--n-batches', default=64)
@click.option('--batch-size', default=2048)
@click.option('--gpu-id', default=1)
def main(model_path, n_comp, n_batches, batch_size, gpu_id):
    """ Learn a (linear PCA) compression of the feature representations
    in each layer of a given model (or, at least, the input/conv/globalpool layers.)
    """

    # Load dataset info to pass to `create_dataset`
    model_name = str(Path(model_path).parent.name)
    dataset = model_name.split('dataset-')[1].split('_target')[0]
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    binocular = 'StereoResNet' in model_path
    train_data, _ = create_dataset(info, Y_col='id', batch_size=batch_size, binocular=binocular)

    # Load model
    model = load_model(model_path)

    # Open hdf5 file to store PCA params and some meta-data    
    f_out = h5py.File(f'{model_path}_compressed.h5', 'w')
    f_out.attrs['n_comp'] = n_comp   
    f_out.attrs['n_batches'] = n_batches

    layers = list(loop_over_layers(model))
    for layer in layers:
        # Create separate group for every layer        
        grp = f_out.create_group(layer.name)
        extractor = Model(model.input, layer.output)
        pca = TFIncrementalPCA(n_components=n_comp)

        for i, (X, _) in enumerate(train_data):

            # Only process `n_batches`
            if i == n_batches:
                break

            # Extract activations (a_N, N = neural)
            with tf.device(f'/gpu:{gpu_id}'):
                a_N = extractor.predict(X, verbose=0)
                a_N = tf.reshape(a_N, (batch_size, -1))            
                
                if a_N.dtype == tf.float16:
                    a_N = tf.cast(a_N, tf.float32)

                pca.partial_fit(a_N)

        pca.mean_ = pca.mean_.numpy()
        pca.components_ = pca.components_.numpy()
        exp_var = pca.explained_variance_ratio_.numpy().sum()
        print(f"Layer: {layer.name}, explained var.: {exp_var:.3f}")

        # Save PCA mean and component parameters (and expl. variance)       
        grp.create_dataset('mu', data=pca.mean_)
        grp.create_dataset('W', data=pca.components_)
        grp.attrs['explained_variance'] = exp_var

    f_out.close()


if __name__ == '__main__':
    main()