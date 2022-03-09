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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import cross_val_score

sys.path.append('.')
from src.models import MODELS
from src.io import create_test_dataset
from src.layers import CKA, OneMinCorrRdm, EuclideanRdm
from src.losses import AngleLoss


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
@click.option('--compressed', is_flag=True)
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--n-samples', default=512)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--cpu', is_flag=True)
@click.option('--get-background', is_flag=True)
def main(model_path, method, rsa_type, compressed, target, n_samples, n_id_test, cpu, get_background):

    # Infer dataset the model was trained on from `model_path`
    dataset = model_path.split('dataset-')[1].split('_')[0]
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    df_test = info.query("split == 'testing'")  # only use test-set stimuli!

    if n_id_test is not None:
        # Select a subset of face IDs
        ids = df_test['id'].unique().tolist()
        ids = random.sample(ids, n_id_test)
        df_test = df_test.query('id in @ids')
    else:
        n_id_test = df_test['id'].nunique()

    # Get an equal number of samples per `n_id_test`
    df_test = df_test.groupby('id').sample(n_samples // n_id_test)
    df_test = df_test.sort_values(by=['id', 'ethn', 'gender', 'age'], axis=0)

    ctx = '/cpu:0' if cpu else '/gpu:0'
    with tf.device(ctx):

        # Get a single batch of stimuli
        dataset_tf = create_test_dataset(df_test, batch_size=n_samples)
        X, shape, tex, bg = dataset_tf.__iter__().get_next()

        # Load model and filter layers
        model = load_model(model_path, custom_objects={'AngleLoss': AngleLoss})
        
        layers2analyze = ['input', 'conv', 'globalpool']    
        layers = [l for l in model.layers if 'shortcut' not in l.name]
        layers = [l for l in layers if any([l2a in l.name for l2a in layers2analyze])]
        factors = ['id', 'xr', 'yr', 'zr', 'xl', 'yl']

        results = defaultdict(list)
        if method == 'rsa':
            rdms = dict(feature=dict(), neural=dict())
            if rsa_type == 'OneMinCorr':
                rdm_comp = OneMinCorrRdm()
            else:
                rdm_comp = CKA()
            
            id_ohe = pd.get_dummies(df_test.loc[:, 'id']).to_numpy()

        if compressed:
            comp_file = f'results/{full_model_name}_epoch-{epoch}_compression.h5'
            comp_params = h5py.File(comp_file, 'r')

        for op_nr, layer in enumerate(tqdm(layers)):

            if 'layer' in layer.name:
                layer_nr = layer.name.split('_')[0].split('layer')[1]
                op_type = layer.name.split('_')[-1]
            else:
                layer_nr = int(layer_nr) + 1
                op_type = 'logits'

            # Note to self: predict_step avoids warning that you get
            # when calling predict (but does the same)
            extractor = Model(inputs=model.inputs, outputs=[layer.output])
            a_N = extractor.predict_step(X)
            a_N = tf.reshape(a_N, (tf.shape(a_N)[0], -1)).numpy()

            if compressed:
                a_N = a_N
                mu = comp_params[layer.name]['mu'][:]
                W = comp_params[layer.name]['W'][:]
                a_N = (a_N - mu) @ W.T

            if method == 'rsa':
                a_N = tf.convert_to_tensor(a_N, dtype=tf.float32)
                rdms['neural'][layer.name] = rdm_comp.get_rdm(a_N).numpy()
                
            for v in factors + [('shape', shape), ('tex', tex), ('background', bg)]:
                
                if v in factors:
                    a_F = df_test.loc[:, v]
                    if v in ['id', 'gender', 'ethn']:
                        a_F = pd.get_dummies(a_F)
                    a_F = a_F.to_numpy()
                else:
                    v, a_F = v
                    a_F = a_F.numpy()

                if v in ['shape', 'tex']:
                    a_F = a_F[:, 1]

                if method == 'rsa':
                    if v in ['shape', 'tex']:
                        n_per_id = id_ohe.sum(axis=0)[:, None]
                        a_F = (id_ohe.T @ a_F) / n_per_id
                        this_a_N = (id_ohe.T @ a_N) / n_per_id
                        this_a_N = tf.convert_to_tensor(this_a_N, dtype=tf.float32)
                    else:
                        this_a_N = a_N
               
                    a_F = tf.convert_to_tensor(a_F, dtype=tf.float32)
                    r = np.round(rdm_comp(this_a_N, a_F).numpy(), 4)
                else:
                    if v in ['id', 'gender', 'ethn']:
                        mod = LogisticRegression()
                        a_F = a_F.argmax(axis=1)
                    else:
                        mod = Ridge(fit_intercept=False)
                        a_F = (a_F - a_F.mean(axis=0)) / a_F.std(axis=0)

                    this_a_N = StandardScaler().fit_transform(a_N)
                    r = cross_val_score(mod, this_a_N, a_F, cv=5).mean()
                
                print(layer.name, v, np.round(r, 3))                
                results['corr'].append(r)
                results['factor'].append(v)
                results['layername'].append(layer.name)
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
                results['op_nr'].append(op_nr)

                if method == 'rsa':
                    if v not in rdms['feature'].keys():
                        rdms['feature'][v] = rdm_comp.get_rdm(a_F).numpy()

        df = pd.DataFrame(results)
        f_out = f"results/{full_model_name}_epoch-{epoch}.tsv"
        df.to_csv(f_out, sep='\t', index=False)
 
        if method == 'rsa':    
            for tpe in ['feature', 'neural']:
                this_dict = rdms[tpe]
                np.savez(f_out.replace('.tsv', f'_rdm-{tpe}.npz'), **this_dict)

        if compressed:
            comp_params.close()

if __name__ == '__main__':

    main()