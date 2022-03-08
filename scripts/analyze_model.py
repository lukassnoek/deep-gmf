import sys
import random
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

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
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--epoch', default=None)
@click.option('--dataset', default='manyIDmanyIMG')
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--n-samples', default=512)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--cpu', is_flag=True)
@click.option('--get-background', is_flag=True,)
def main(model_name, epoch, dataset, target, n_samples, n_id_test, cpu, get_background):

    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    df_test = info.query("split == 'testing'")
    if n_id_test is not None:
        # Select a subset of face IDs
        ids = df_test['id'].unique().tolist()
        ids = random.sample(ids, n_id_test)
        df_test = df_test.query('id in @ids')
    else:
        n_id_test = df_test['id'].nunique()

    df_test = df_test.groupby('id').sample(n_samples // n_id_test)

    df_test = df_test.sort_values(by=['id', 'ethn', 'gender', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl'],
                                  axis=0)  # sort to get nice looking RDMs
    
    ctx = '/cpu:0' if cpu else '/gpu:0'
    with tf.device(ctx):
        
        dataset_tf = create_test_dataset(df_test, batch_size=n_samples, get_background=get_background)
        if get_background:
            X, shape, tex, bg = dataset_tf.__iter__().get_next()
        else:
            X, shape, tex = dataset_tf.__iter__().get_next()

        # Load model and filter layers
        full_model_name = f"{model_name}_dataset-{dataset}_target-{'+'.join(target)}"
        if epoch is None:
            all_models = Path(f'trained_models/{full_model_name}').glob('epoch-*')
            epoch = str(sorted(list(all_models))[-1]).split('epoch-')[1]

        f_in = f'trained_models/{full_model_name}/epoch-{epoch}'
        model = load_model(f_in, custom_objects={'AngleLoss': AngleLoss})
        
        layers = [l for l in model.layers if 'shortcut' not in l.name]
        #factors = ['id', 'ethn', 'gender', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']
        factors = ['id', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl']

        results = defaultdict(list)
        rdms = dict(feature=dict(), neural=dict())
        cka = CKA()
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
            a_N = tf.reshape(a_N, (n_samples, -1))
            #a_N = a_N.numpy()
            #a_N -= a_N.mean(axis=0)

            rdms['neural'][layer.name] = cka.get_rdm(a_N).numpy()

            for v in factors + [('shape', shape), ('tex', tex)]:#, ('background', bg)]:
                if isinstance(v, str):
                    a_F = pd.get_dummies(df_test.loc[:, v]).to_numpy()
                else:
                    v, a_F = v
                    a_F = a_F.numpy()

                #mod = Ridge()
                #a_F = a_F - a_F.mean(axis=0)
                #a_N_train, a_N_test, a_F_train, a_F_test = train_test_split(a_N, a_F, test_size=64)
                #mod.fit(a_N_train, a_F_train)
                #r = mod.score(a_N_test, a_F_test)
                #print(v, r)
                a_F = tf.convert_to_tensor(a_F, dtype=tf.float32)
                r = np.round(cka(a_N, a_F).numpy(), 4)
                results['corr'].append(r)
                results['factor'].append(v)
                results['layername'].append(layer.name)
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
                results['op_nr'].append(op_nr)

                if v not in rdms['feature'].keys():
                    rdms['feature'][v] = cka.get_rdm(a_F).numpy()

        df = pd.DataFrame(results)
        f_out = f"results/{full_model_name}_epoch-{epoch}.tsv"
        df.to_csv(f_out, sep='\t', index=False)
        
        for tpe in ['feature', 'neural']:
            this_dict = rdms[tpe]
            np.savez(f_out.replace('.tsv', f'_rdm-{tpe}.npz'), **this_dict)


if __name__ == '__main__':

    main()