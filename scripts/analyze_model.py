import sys
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append('.')
from src.models import MODELS
from src.io import create_dataset, DATASETS
from src.layers import CKA


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
@click.argument('target', type=click.STRING, default='id')
@click.option('-n', '--n-samples', type=click.INT, default=512)
def main(model_name, target, n_samples):

    info = pd.read_csv(Path(DATASETS['gmf']) / 'dataset_info.csv')
    info = info.sample(n=n_samples, replace=False)
    info = info.sort_values(by=['bg', 'id', 'ethn', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'l'],
                            axis=0)

    dataset = create_dataset(info, Y_col=target, Z_col=['shape', 'tex'], validation_split=None,
                             target_size=(224, 224), batch_size=n_samples, shuffle=False)

    (X, shape, tex), _ = dataset.__iter__().get_next()
    
    # Load model and filter layers
    full_model_name = f'{model_name}_dataset-gmf_target-{target}'
    model = load_model(f'models/{full_model_name}')
    layers = [l for l in model.layers if 'shortcut' not in l.name]
    factors = ['id', 'ethn', 'age', 'bg', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'l']

    results = defaultdict(list)
    for layer in tqdm(layers):

        extractor = Model(inputs=model.inputs, outputs=[layer.output])
        a_N = extractor.predict(X, batch_size=n_samples)
        a_N = tf.reshape(a_N, (n_samples, -1))

        for v in factors + [('shape', shape), ('tex', tex)]:
            if isinstance(v, str):
                a_F = pd.get_dummies(info.loc[:, v]).to_numpy()
            else:
                v, a_F = v
 
            a_F = tf.convert_to_tensor(a_F, dtype=tf.float32)
            cka = CKA()
            r = np.round(cka(a_N, a_F).numpy(), 4)
            results['corr'].append(r)
            results['factor'].append(v)
            if 'layer' in layer.name:
                layer_nr = layer.name.split('_')[0].split('-')[1]
                op_type = layer.name.split('_')[-1]
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
            else:
                # Must be logits
                results['layer'].append(int(layer_nr) + 1)
                results['operation'].append('logits')            

    df = pd.DataFrame(results)
    df.to_csv(f'results/{full_model_name}.tsv', sep='\t', index=False)
    


if __name__ == '__main__':

    main()