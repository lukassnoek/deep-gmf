import os
import sys
import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.models import MODELS
from src.io import create_dataset
from src.losses import AngleLoss


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('-t', '--target', default=['shape'], multiple=True)
@click.option('--epoch', default=None)
@click.option('--dataset', default='manyIDmanyIMG')
@click.option('--n-comp', default=500)
@click.option('--n-batches', default=2)
@click.option('--batch-size', default=512)
@click.option('--query', default=None)
def main(model_name, target, epoch, dataset, n_comp, n_batches, batch_size, query):
    
    # Load dataset info to pass to `create_dataset`
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    train_data, _ = create_dataset(info, Y_col='shape', batch_size=batch_size, query=query)
    
    # Load model and filter layers
    full_model_name = f"{model_name}_dataset-{dataset}_target-{'+'.join(target)}"
    if epoch is None:
        all_models = Path(f'trained_models/{full_model_name}').glob('epoch-*')
        epoch = str(sorted(list(all_models))[-1]).split('epoch-')[1]

    f_in = f'trained_models/{full_model_name}/epoch-{epoch}'
    model = load_model(f_in, custom_objects={'AngleLoss': AngleLoss})
    layers = [l for l in model.layers if 'shortcut' not in l.name]
    
    f_out_path = Path(f'results/{full_model_name}_epoch-{epoch}_compression.h5')
    if f_out_path.exists():
        os.remove(f_out_path)

    f_out = h5py.File(f_out_path, 'w')
    f_out.attrs['n_comp'] = n_comp   
    f_out.attrs['n_batches'] = n_batches
    
    for layer in tqdm(layers):
        grp = f_out.create_group(layer.name)
        extractor = Model(inputs=model.inputs, outputs=[layer.output])
        pca = IncrementalPCA(n_components=n_comp)
        for i, (X, _) in enumerate(tqdm(train_data, total=n_batches)):

            if i == n_batches:
                break

            a_N = extractor.predict_step(X).numpy()
            a_N = a_N.reshape((batch_size, -1))
            pca.partial_fit(a_N)
        
        grp.create_dataset('mu', data=pca.mean_, compression='gzip', compression_opts=9)
        grp.create_dataset('W', data=pca.components_, compression='gzip', compression_opts=9)
        grp.attrs['explained_variance'] = pca.explained_variance_ratio_.sum()
        print(grp.attrs['explained_variance'])

    f_out.close()


if __name__ == '__main__':
    main()