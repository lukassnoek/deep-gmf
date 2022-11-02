import os
os.environ['OPENBLAS_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_MAX_THREADS'] = '20'

import sys
import random
import click
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.io import create_test_dataset
from src.losses import AngleLoss


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--batch-size', default=512)
@click.option('--n-samples', type=click.INT, default=None)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--gpu-id', default=1)
@click.option('--binocular', is_flag=True)
def main(model_path, batch_size, n_samples, n_id_test, gpu_id, binocular):
    """ Decomposes model activations for `n_samples` of stimuli into
    a (optionally PCA-compressed) single array of activations per layer,
    in order to speed up the decoding analyses. """

    model_path = Path(model_path)
    # Infer dataset the model was trained on from `model_path``
    model_name = str(model_path.parent.name)
    dataset = model_name.split('dataset-')[1].split('_target')[0]
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    df_test = info.query("split == 'testing'")  # only use test-set stimuli!

    if n_id_test is not None:
        # Select a subset of face IDs
        ids = df_test['id'].unique().tolist()
        ids = random.sample(ids, n_id_test)
        df_test = df_test.query('id in @ids')
    else:
        n_id_test = df_test['id'].nunique()

    if n_samples is None:
        n_samples = df_test.shape[0]
    
    print(f"Number of IDs: {n_id_test}, total samples: {n_samples}")
    if n_id_test > n_samples:
        raise ValueError(f"Cannot have more face IDs ({n_id_test}) than samples ({n_samples})!")

    if n_samples % batch_size != 0:
        raise ValueError("`n_samples` is not a multiple of `batch_size`!")

    # Get an equal number of samples per `n_id_test`
    df_test = df_test.groupby('id').sample(n_samples // n_id_test)
    df_test = df_test.sample(frac=1)

    # Open hdf5 file to store PCA params and some meta-data    
    f_out = h5py.File(f'{model_path}_decomposed.h5', 'w')
    f_out.attrs['n_samples'] = n_samples   
    f_out.attrs['n_id_test'] = n_id_test

    with tf.device(f'/gpu:{gpu_id}'):

        # Get a single batch of stimuli
        dataset_tf = create_test_dataset(df_test, batch_size=batch_size, binocular=binocular)

        # Load model and filter layers
        model = load_model(model_path, custom_objects={'AngleLoss': AngleLoss})
        layers2analyze = ['input', 'conv', 'flatten', 'globalpool']    
        layers = [l for l in model.layers if 'shortcut' not in l.name]
        layers = [l for l in layers if any([l2a in l.name for l2a in layers2analyze])]
        factors = ['id', 'ethn', 'gender', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']

        for fac in factors:
            f_out.create_dataset(fac, data=df_test.loc[:, fac].to_numpy())

        comp_params = h5py.File(f'{str(model_path)}_compressed.h5', 'r')

        shape, tex, bg = [], [], []
        n_batches = n_samples // batch_size
        for _, shape_, tex_ in tqdm(dataset_tf.as_numpy_iterator(), desc='a_F', total=n_batches):
            shape.append(shape_)
            tex.append(tex_)
            #bg.append(bg_)
        
        f_out.create_dataset('shape', data=np.vstack(shape))
        f_out.create_dataset('tex', data=np.vstack(tex))
        #f_out.create_dataset('bg', data=np.vstack(bg).astype(np.uint8))[]

        grp = f_out.create_group('a_N')
        pbar = tqdm(layers)
        for layer in layers:
            
            # Zero-pad layer name
            layer_nr = int(layer.name.split('layer')[1].split('_')[0])
            layer_name = layer.name.replace(f'layer{layer_nr}', f'layer{layer_nr:02d}')
            pbar.set_description(layer_name)           
            
            # Note to self: predict_step avoids warning that you get
            # when calling predict (but does the same)
            extractor = Model(inputs=model.inputs, outputs=[layer.output])
            a_N = []
            for batch in dataset_tf:
                X_ = batch[0]
                a_N.append(extractor.predict(X_))
            
            a_N = np.vstack(a_N)

            # Reshape to obs x features array (and cast to numpy)
            a_N = np.reshape(a_N, (a_N.shape[0], -1))

            # apply PCA transformation
            mu = comp_params[layer.name]['mu'][:]
            W = comp_params[layer.name]['W'][:]
            a_N = (a_N - mu) @ W  # n_samples x 500

            grp.create_dataset(layer_name, data=a_N.copy())

    comp_params.close()        
    f_out.close()

if __name__ == '__main__':
    main()