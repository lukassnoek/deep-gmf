# Set maximum number of CPUs to 10 otherwise will use all available CPUs on the server
import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['NUMEXPR_MAX_THREADS'] = '10'

import click
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from functools import partial
from collections import defaultdict
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline  


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--n-id-train', type=int, default=None)
@click.option('--n-id-val', type=int, default=None)
@click.option('--n-id-test', type=int, default=None)
def main(model_path, n_id_train, n_id_val, n_id_test):
    """Analyzes generative feature representations in a DNN model.
    
    Parameters
    ----------
    model_path : str
        Path to the model to be analyzed (should be in ../trained_models)
    n_id_train : int
        Number of identities to use for training. If None, all identities are used
    n_id_val : int
        Number of identities to use for validation. If None, all identities are used
    n_id_test : int
        Number of identities to use for testing. If None, all identities are used
    """
    model_path = Path(model_path)
    model_name = str(model_path.parent.name)
    epoch_id = str(model_path.name)
    f_in = h5py.File(str(model_path) + '_decomposed.h5', 'r')
    
    # Define data across splits from h5 file
    train_data = f_in['training']
    val_data = f_in['validation']
    test_data = f_in['testing']
    
    # If a subset of identities should be used for any split, select them here
    idx = []
    for data, n in zip([train_data, val_data, test_data], [n_id_train, n_id_val, n_id_test]):

        if n is not None:
            all_ids = data['id'][:]
            selected_ids = np.random.choice(np.unique(all_ids), size=n, replace=False)
            idx.append(np.isin(all_ids, selected_ids))
        else:
            idx.append(np.ones(data['id'].shape, dtype=bool))

    train_idx, val_idx, test_idx = idx

    all_factors = ['id', 'bg', 'ethn', 'gender', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'emo']
    cat_factors = ['id', 'bg', 'ethn', 'gender', 'emo']  # categorical factors

    # To be saved for later
    df = {}
    for factor in all_factors:
        df[factor] = test_data[factor][test_idx]

    df = pd.DataFrame(df)
    results = defaultdict(list)    

    # Loop over layers
    for op_nr, l_name in enumerate(tqdm(train_data['a_N'].keys())):

        if 'softmax' in l_name:
            # Not interesting, so skip
            continue

        layer_nr = l_name.split('_')[0].split('layer')[1]
        op_type = l_name.split('_')[-1]  # operation type, e.g., 'conv'
        
        X_train = train_data['a_N'][l_name][train_idx]
        X_val = val_data['a_N'][l_name][val_idx]
        X_test = test_data['a_N'][l_name][test_idx]

        if op_nr == 0:
            print(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")

        out_df = df.copy()

        # Loop over all generative stimulus features (called 'factors' here, don't know why I did that)
        for v in all_factors + ['shape', 'tex']:
            Y_train = train_data[v][train_idx]
            Y_val = val_data[v][val_idx]
            Y_test = test_data[v][test_idx]

            if v in cat_factors:
                # Encode strings as integers for categorical target variables
                label_enc = LabelEncoder()
                Y_test = label_enc.fit_transform(Y_test)

                if v != 'id':
                    # Do not do this when predicting identity, because we cannot cross-validate
                    # across train and test set (as these contain unique identities) and the LabelEncoder
                    # will fail when it encounters a new identity in the val/test set
                    Y_val = label_enc.transform(Y_val)
                    Y_train = label_enc.transform(Y_train)

                # Set ROC-AUC score to 'ovr' to deal with multi-class classification
                metric = partial(roc_auc_score, multi_class='ovr')
                # Use LDA because it's fast (and not much worse than logistic regression in my experience)
                model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            else:
                # For continuous target variables, use penalized ridge regression, in which
                # the regularization parameter is determined by cross-validation
                alphas = np.logspace(-3, 4, num=8, endpoint=True)
                metric = partial(r2_score, multioutput='raw_values')
                if v in ['shape', 'tex']:
                    # Pick the best alpha for each target variable separately
                    cv_scores = np.zeros((alphas.size, Y_train.shape[1]))
                else:
                    cv_scores = np.zeros_like(alphas)

                for i, alpha in enumerate(alphas):
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
                    model.fit(X_train, Y_train)
                    preds = model.predict(X_val)
                    cv_scores[i] = metric(Y_val, preds)

                # Determine best alpha (per target variable) and create new pipeline
                alpha = alphas[cv_scores.argmax(axis=0)]
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

            if v == 'id':
                # Do not cross-validate from train to test set when predicting identity,
                # because the splits contain unique identities; do 4-fold cv within the test set instead
                preds = cross_val_predict(model, X_test, Y_test, cv=4, method='predict_proba')
            else:
                # Cross-validate from train to test set
                model.fit(X_train, Y_train)
                if v in cat_factors:
                    preds = model.predict_proba(X_test)
                else:
                    preds = model.predict(X_test)

            if v in cat_factors:
                # For binary classification, select p(y=1) because otherwise the ROC-AUC score will error
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
                
            score = metric(Y_test, preds)            
            
            if v in cat_factors:
                # Normalize ROC-AUC to 0-1 range
                score = (score - .5) / 0.5

                if preds.ndim == 1:
                    # For binary classification, add p(y=0) as well
                    preds = np.stack([1 - preds, preds], axis=1)

                # Save all predictions and true feature values
                out_df[v + '_truepred'] = preds[np.arange(len(Y_test)), Y_test]
                preds = pd.DataFrame(preds, columns=[f'{v}_pred{i:03}' for i in range(preds.shape[1])])
                out_df = pd.concat([out_df, preds], axis=1)
                out_df[v] = Y_test
            elif v in ['shape', 'tex']:
                # Save all predictions and true feature values as well as best predicted feature within all shape/tex comps
                best_idx = score.argmax()
                best = pd.DataFrame(np.stack([Y_test[:, best_idx],preds[:, best_idx]], axis=1), columns=[v, v + '_pred'])
                Y_test = pd.DataFrame(Y_test, columns=[f'{v}{i:03}' for i in range(Y_test.shape[1])])
                preds = pd.DataFrame(preds, columns=[f'{v}_pred{i:03}' for i in range(preds.shape[1])])
                out_df = pd.concat([out_df, Y_test, preds, best], axis=1)
            else:
                out_df[v + '_pred'] = preds


            if not isinstance(score, np.ndarray):
                score = [score]

            # Just for printing            
            agg_s = 'max' if v in ['shape', 'tex'] else 'mean'
            agg_score = np.max(score) if v in ['shape', 'tex'] else np.mean(score)
            print(f"target = {v}, layer = {l_name}, {agg_s} score: {agg_score:.3f}")

            # Save results into defaultdict
            for i, sc_ in enumerate(score):
                results['corr'].append(sc_)
                results['factor'].append(v)
                results['feature_nr'].append(i+1)
                results['layername'].append(l_name)
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
                results['op_nr'].append(op_nr)

        # Done processing all features for layer, save preds
        out_df.to_csv(f'results/{model_name}_{epoch_id}_layer-{l_name}_preds.csv', index=False)

    df = pd.DataFrame(results)
    f_out = f"results/{model_name}_{epoch_id}_perf.tsv"
    df.to_csv(f_out, sep='\t', index=False)

    f_in.close()

if __name__ == '__main__':
    main()