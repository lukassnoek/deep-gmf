import sys
import random
import click
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, balanced_accuracy_score

sys.path.append('.')
from src.io import create_test_dataset
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
@click.option('--compress', is_flag=True)
@click.option('--n-samples', default=512)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--cpu', is_flag=True)
def main(model_path, compress, n_samples, n_id_test, cpu):

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
        if compress:
            comp_params = h5py.File(f'{str(model_path)}_compressed.h5', 'r')

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
            a_N = extractor.predict_step(X)
            
            # Reshape to obs x features array (and cast to numpy)
            a_N = tf.reshape(a_N, (tf.shape(a_N)[0], -1)).numpy()

            if compress:
                # apply PCA transformation
                mu = comp_params[layer.name]['mu'][:]
                W = comp_params[layer.name]['W'][:]
                a_N = (a_N - mu) @ W.T  # n_samples x 500
            
            # Important for Ridge
            a_N = (a_N - a_N.mean(axis=0)) / a_N.std(axis=0)
            
            for v in factors + [('shape', shape), ('tex', tex)]:#, ('background', bg)]:
                
                if v in factors:
                    # regular parameter
                    a_F = df_test.loc[:, v]
                    if v in ['id', 'gender', 'ethn']:
                        # if categorical, one-hot encode variable
                        a_F = pd.get_dummies(a_F)
                        cv = StratifiedKFold(n_splits=4)
                    else:
                        cv = GroupKFold(n_splits=4)

                    a_F = a_F.to_numpy()                    
                else:
                    # It's already a n_samples x features array
                    v, a_F = v
                    a_F = a_F.numpy()
                    cv = GroupKFold(n_splits=4)
                
                nonzero = a_F.sum(axis=0) != 0
                a_F = a_F[..., nonzero].squeeze()
                if v in ['id', 'gender', 'ethn']:
                    mod = LogisticRegression()
                    a_F = a_F.argmax(axis=1)
                else:
                    mod = Ridge(fit_intercept=False)
                    a_F = (a_F - a_F.mean(axis=0)) / a_F.std(axis=0)
                    
                # Important to scale when using Ridge
                pipe = mod#make_pipeline(StandardScaler(), mod)
                if v in ['id', 'gender', 'ethn'] or a_F.ndim == 1:
                    r = np.zeros(5)
                else:
                    r = np.zeros((5, a_F.shape[1]))

                for i, (train_idx, test_idx) in enumerate(cv.split(a_N, a_F, groups=df_test.loc[:, 'id'].to_numpy())):
                    pipe.fit(a_N[train_idx], a_F[train_idx])
                    preds = pipe.predict(a_N[test_idx])
                    if v in ['id', 'gender', 'ethn']:
                        r[i] = balanced_accuracy_score(a_F[test_idx], preds)
                    elif v in ['shape', 'tex', 'background']:
                        r[i, :] = r2_score(a_F[test_idx], preds, multioutput='raw_values')
                    else:
                        r[i] = r2_score(a_F[test_idx], preds)

                r_av = r.mean(axis=0)

                if r.ndim == 1:
                    print(layer.name, v, np.round(r_av, 3))
                    r_av = [r_av]
                elif v == 'background':
                    r_av = [r_av.mean()]
                else:
                    print(layer.name, v, np.round(np.mean(r_av), 3), np.argmax(r_av))

                for i, r_ in enumerate(r_av):
                    results['corr'].append(r_)
                    results['factor'].append(v)
                    results['feature_nr'].append(i)
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