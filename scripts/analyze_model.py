from email.headerregistry import Group
import os
os.environ['OPENBLAS_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_MAX_THREADS'] = '20'

import sys
import click
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from functools import partial
from collections import defaultdict
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.linear_model import RidgeCV, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, r2_score

sys.path.append('.')
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
@click.option('--use-fracridge', is_flag=True)
def main(model_path, use_fracridge):

    model_path = Path(model_path)
    # Infer dataset the model was trained on from `model_path``
    model_name = str(model_path.parent.name)
    epoch_id = str(model_path.name)

    f_in = h5py.File(str(model_path) + '_decomposed.h5', 'r')

    factors = ['ethn', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']
    groups = f_in['id'][:]
    results = defaultdict(list)    

    for op_nr, l_name in enumerate(tqdm(f_in['a_N'].keys())):
        layer_nr = l_name.split('_')[0].split('layer')[1]
        op_type = l_name.split('_')[-1]
        a_N = f_in['a_N'][l_name][:]

        # Remove zero ("dead") units
        nonzero = a_N.sum(axis=0) != 0
        a_N = a_N[:, nonzero]
        a_N = (a_N - a_N.mean(axis=0)) / a_N.std(axis=0)

        for v in ['shape', 'tex']:#factors + ['shape', 'tex']:#, 'bg']:
            a_F = f_in[v][:]
            if v in ['gender', 'ethn']:
                a_F = a_F.astype(str)
            else:
                nonzero = a_F.sum(axis=0) != 0
                a_F = a_F[..., nonzero].squeeze()
                a_F = (a_F - a_F.mean(axis=0)) / a_F.std(axis=0)

            if a_F.ndim == 2:
                if a_F.shape[1] == 0:
                    print(f"WARNING: feature {v} is constant!")
                    continue

            if v in ['gender', 'ethn']:
                #model = LogisticRegression(max_iter=500)
                model = RidgeClassifier()
                metric = balanced_accuracy_score
            else:
                scaler = StandardScaler()
                alphas = np.logspace(-3, 4, num=8, endpoint=True)
                model = RidgeCV(alphas=alphas, alpha_per_target=True, fit_intercept=False)
                #model = PLSRegression(n_components=10)
                #model = make_pipeline(scaler, model)
                metric = partial(r2_score, multioutput='raw_values')

            cv = GroupKFold(n_splits=4)
            out = cross_val_predict(model, a_N, a_F, groups=groups, cv=cv, n_jobs=4)
            score = metric(a_F, out)
            if not isinstance(score, np.ndarray):
                score = [score]
            print(l_name, np.max(score))
            # Average across folds
            #score_av = np.median(score.squeeze(), axis=0)
            
            # if v == 'background':
            #     # Do not save r2 values of *all* background pixels
            #     score_av = score_av.mean()
            #     print(f"Layer {l_name}, {v}: {score_av:.2f}")
            #     score_av = [score_av]
            # elif score_av.ndim == 0:
            #     print(f"Layer {l_name}, {v}: {score_av:.2f}")
            #     score_av = [score_av]
            # else:
            #     score_av_ = score_av.mean()
            #     s_max = np.max(score_av)
            #     s_min = np.min(score_av)
            #     s_amax = np.argmax(score_av)
            #     print(f"Layer {l_name}, {v}: {score_av_:.2f}, max: {s_max:.3f} ({s_amax}), min: {s_min:.3f}")

            for i, sc_ in enumerate(score):
                results['corr'].append(sc_)
                results['factor'].append(v)
                results['feature_nr'].append(i+1)
                results['layername'].append(l_name)
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
                results['op_nr'].append(op_nr)

    df = pd.DataFrame(results)
    print(df.groupby(['layer', 'factor']).mean()['corr'])
    f_out = f"results/{model_name}_{epoch_id}.tsv"
    df.to_csv(f_out, sep='\t', index=False)

    f_in.close()

if __name__ == '__main__':
    main()