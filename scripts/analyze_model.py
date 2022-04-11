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
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.io import create_test_dataset
from src.losses import AngleLoss
from src.decoding import decode_fracridge, decode_regular


F_NAMES = {
    'bg': 'background',
    'l': 'lights',
    'xr': 'rot-x',
    'yr': 'rot-y',
    'zr': 'rot-z',
    'xt': 'trans-x',
    'yt': 'trans-y',
    'zt': 'trans-z',
    'xr_yr_zr': 'rotation',
    'xt_yt': 'translation',
    'ethn': 'ethnicity',
    'age': 'age',
    'id': 'Face ID'
}


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--compress', is_flag=True)
@click.option('--batch-size', default=512)
@click.option('--n-samples', default=1024)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--use-fracridge', is_flag=True)
@click.option('--gpu-id', default=1)
def main(model_path, compress, batch_size, n_samples, n_id_test, use_fracridge, gpu_id):

    model_path = Path(model_path)
    # Infer dataset the model was trained on from `model_path``
    model_name = str(model_path.parent.name)
    epoch_id = str(model_path.name)
    dataset = model_name.split('dataset-')[1].split('_')[0]
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    df_test = info.query("split == 'testing'")  # only use test-set stimuli!

    if n_id_test is not None:
        # Select a subset of face IDs
        ids = df_test['id'].unique().tolist()
        ids = random.sample(ids, n_id_test)
        df_test = df_test.query('id in @ids')
    else:
        n_id_test = df_test['id'].nunique()

    if n_id_test > n_samples:
        raise ValueError(f"Cannot have more face IDs ({n_id_test}) than samples ({n_samples})!")

    if use_fracridge:
        f_decode = decode_fracridge
    else:
        f_decode = decode_regular

    # Get an equal number of samples per `n_id_test`
    df_test = df_test.groupby('id').sample(n_samples // n_id_test)
    df_test = df_test.sample(frac=1)

    #df_test = df_test.sort_values(by=['id', 'ethn', 'gender', 'age'], axis=0)
    with tf.device(f'/gpu:{gpu_id}'):

        # Get a single batch of stimuli
        dataset_tf = create_test_dataset(df_test, batch_size=batch_size)

        # Load model and filter layers
        model = load_model(model_path, custom_objects={'AngleLoss': AngleLoss})
        layers2analyze = ['input', 'conv', 'flatten', 'globalpool']    
        layers = [l for l in model.layers if 'shortcut' not in l.name]
        layers = [l for l in layers if any([l2a in l.name for l2a in layers2analyze])]
        factors = ['id', 'ethn', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']

        results = defaultdict(list)
        if compress:
            comp_params = h5py.File(f'{str(model_path)}_compressed.h5', 'r')

        groups = df_test.loc[:, 'id'].to_numpy()        
        shape, tex, bg = [], [], []
        for _, shape_, tex_, bg_ in dataset_tf:
            shape.append(shape_.numpy())
            tex.append(tex_.numpy())
            bg_ = tf.image.resize(tf.reshape(bg_, (batch_size, 256, 256, 3)), size=(112, 112))
            bg.append(tf.reshape(bg_, (batch_size, -1)).numpy())

        shape = np.vstack(shape)
        tex = np.vstack(tex)
        bg = np.vstack(bg)
        
        for op_nr, layer in enumerate(tqdm(layers)):

            if 'layer' in layer.name:
                layer_nr = layer.name.split('_')[0].split('layer')[1]
                op_type = layer.name.split('_')[-1]
            else:
                # Only relevant when also analyzing logits (which is not
                # very informative)
                layer_nr = int(layer_nr) + 1
                op_type = 'logits'

            # Note to self: predict_step avoids warning that you get
            # when calling predict (but does the same)
            extractor = Model(inputs=model.inputs, outputs=[layer.output])
            a_N = []
            for X_, _, _, _ in dataset_tf:
                a_N.append(extractor.predict_step(X_).numpy())
            
            a_N = np.vstack(a_N)

            # Reshape to obs x features array (and cast to numpy)
            a_N = tf.reshape(a_N, (a_N.shape[0], -1)).numpy()

            if compress:
                # apply PCA transformation
                mu = comp_params[layer.name]['mu'][:]
                W = comp_params[layer.name]['W'][:]
                a_N = (a_N - mu) @ W  # n_samples x 500

            # Remove zero ("dead") units
            nonzero = a_N.sum(axis=0) != 0
            a_N = a_N[:, nonzero]
            
            for v in factors + [('shape', shape), ('tex', tex), ('background', bg)]:
                if v in factors:
                    # regular parameter
                    a_F = df_test.loc[:, v].to_numpy()
                else:
                    # It's already a n_samples x features array
                    v, a_F = v

                    nonzero = a_F.sum(axis=0) != 0
                    a_F = a_F[..., nonzero].squeeze()

                    if a_F.ndim == 2:
                        if a_F.shape[1] == 0:
                            print(f"WARNING: feature {v} is constant!")
                            continue

                if v in ['id', 'gender', 'ethn']:
                    groups_ = None if v == 'id' else groups
                    score = decode_regular(a_N, a_F, groups=groups_, classification=True)
                else:
                    score = f_decode(a_N, a_F, groups=groups)

                # Average across folds
                score_av = np.median(score.squeeze(), axis=0)

                if v == 'background':
                    # Do not save r2 values of *all* background pixels
                    score_av = score_av.mean()
                    print(f"Layer {layer.name}, {v}: {score_av:.2f}")
                    score_av = [score_av]
                elif score_av.ndim == 0:
                    print(f"Layer {layer.name}, {v}: {score_av:.2f}")
                    score_av = [score_av]
                else:
                    score_av_ = score_av.mean()
                    s_max = np.max(score_av)
                    s_min = np.min(score_av)
                    s_amax = np.argmax(score_av)
                    print(f"Layer {layer.name}, {v}: {score_av_:.2f}, max: {s_max:.3f} ({s_amax}), min: {s_min:.3f}")

                for i, sc_ in enumerate(score_av):
                    results['corr'].append(sc_)
                    results['factor'].append(v)
                    results['feature_nr'].append(i+1)
                    results['layername'].append(layer.name)
                    results['layer'].append(layer_nr)
                    results['operation'].append(op_type)
                    results['op_nr'].append(op_nr)

        df = pd.DataFrame(results)
        
        f_out = f"results/{model_name}_{epoch_id}.tsv"
        df.to_csv(f_out, sep='\t', index=False)
 
        if compress:
            comp_params.close()

if __name__ == '__main__':
    main()