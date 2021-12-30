import sys
import click
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from scipy.spatial.distance import pdist, squareform

sys.path.append('.')
from src.models import MODELS
from src.io import create_data_generator, DATASETS
from src.layers import EuclideanRdm, OneMinCorrRdm, CKA


F_NAMES = {
    'bg': 'background',
    'l': 'lights',
    'xr': 'rot x',
    'yr': 'rot y',
    'zr': 'rot z',
    'xt': 'trans x',
    'yt': 'trans y',
    'xr_yr_zr': 'rotation',
    'xt_yt': 'translation',
    'id': 'Face ID'
}


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('-n', '--n-samples', type=click.INT, default=512)
def main(model_name, n_samples):

    info = pd.read_csv(Path(DATASETS['gmfmini']) / 'dataset_info.csv')
    info = info.query("id < 2")
    info = info.astype({'id': str})
    
    train_gen, train_df = create_data_generator(
        info, y='id', n=n_samples, batch_size=n_samples, return_df=True,
        shuffle=False, target_size=(256, 256)
    )  # set shuffle to False so train_gen and train_df have the same order
    
    # Get data and reorder according to y (nice for RDM viz)
    X, y = next(train_gen)
    reord = np.lexsort([train_df['l'], train_df['zr'], train_df['yr'],train_df['xr'],
                        train_df['yt'], train_df['xt'], train_df['bg'], train_df['id']
                       ])

    # Load model and filter layers
    model = load_model(f'models/{model_name}_dataset-gmfmini')
    layers = [l for l in model.layers if 'shortcut' not in l.name]
    layers = [l for l in layers if l.name.split('_')[-1] in ('conv', 'fc', 'mean')]

    factors = ['id', 'bg', 'xr', 'yr', 'zr', 'xt', 'yt', 'l']

    # Init master figure    
    fig = plt.figure(constrained_layout=True, figsize=(len(layers) + 1, len(factors) + 1))
    gs = fig.add_gridspec(nrows=len(factors) + 1, ncols=len(layers) + 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    rs = defaultdict(list)
    for i, layer in enumerate(layers):

        extractor = Model(inputs=model.inputs, outputs=[layer.output])
        a_N = extractor.predict(X[reord, ...], batch_size=256)
        a_N = tf.reshape(a_N, (n_samples, -1))
        
        cka = CKA()
        rdm_N = cka.get_rdm(a_N).numpy()
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(rdm_N)
        title = '\n'.join([s for s in layer.name.split('_') if 'block' not in s])
        if 'mean' in title:
            title = 'global\npool'

        ax.set_title(title)
        ax.axis('off')

        for ii, v in enumerate(factors):
            a_F = pd.get_dummies(train_df.iloc[reord, :].loc[:, v]).to_numpy()
            a_F = tf.convert_to_tensor(a_F, dtype=tf.float32)

            r = np.round(cka(a_N, a_F).numpy(), 3)
            key = v if isinstance(v, str) else '_'.join(v)
            rs[key].append(r)

            print(f"Layer {layer.name}, feature {v}: {r:.3f}")
            ax = fig.add_subplot(gs[ii + 1, 0])
            rdm_F = cka.get_rdm(a_F)
            ax.imshow(rdm_F)
            ax.axis('off')
            title = F_NAMES[key]
            ax.set_title(title, fontsize=12)

    for i, (name, r) in enumerate(rs.items()):        
        ax = fig.add_subplot(gs[i+1, 1:])
        ax.bar(np.arange(len(layers)), r, width=0.45)
        ax.set_ylim(0, 1)
        ax.set_xlim(-.25, len(layers) - .75)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('CKA', fontsize=12)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    fig.savefig(f'figures/{model_name}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    main()