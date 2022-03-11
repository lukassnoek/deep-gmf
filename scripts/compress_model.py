import sys
import click
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.io import create_dataset
from src.losses import AngleLoss


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--n-comp', default=500)
@click.option('--n-batches', default=2)
@click.option('--batch-size', default=512)
def main(model_path, n_comp, n_batches, batch_size):
    """ Learn a (linear PCA) compression of the feature representations
    in each layer of a given model (or, at least, the input/conv/globalpool layers.)
    """
    # Load dataset info to pass to `create_dataset`
    model_name = str(Path(model_path).parent.name)
    dataset = model_name.split('dataset-')[1].split('_')[0]
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    train_data, _ = create_dataset(info, Y_col='shape', batch_size=batch_size)
        
    # Load model
    model = load_model(model_path, custom_objects={'AngleLoss': AngleLoss})

    # Analyze only relevant layers, to speed up compression
    layers2analyze = ['conv', 'globalpool', 'input']    
    layers = [l for l in model.layers if 'shortcut' not in l.name]
    layers = [l for l in layers if any([l2a in l.name for l2a in layers2analyze])]

    # Open hdf5 file to store PCA params and some meta-data    
    f_out = h5py.File(f'{model_path}_compressed.h5', 'w')
    f_out.attrs['n_comp'] = n_comp   
    f_out.attrs['n_batches'] = n_batches

    for layer in tqdm(layers):
        # Create separate group for every layer
        grp = f_out.create_group(layer.name)
        extractor = Model(inputs=model.inputs, outputs=[layer.output])
        pca = IncrementalPCA(n_components=n_comp)

        for i, (X, _) in enumerate(train_data):
            # Only process `n_batches`
            if i == n_batches:
                break

            # Extract activations (a_N, N = neural)
            a_N = extractor.predict_step(X).numpy()
            a_N = a_N.reshape((batch_size, -1))
            pca.partial_fit(a_N)  # fit PCA

        # Save PCA mean and component parameters (and expl. variance)       
        grp.create_dataset('mu', data=pca.mean_, compression='gzip', compression_opts=9)
        grp.create_dataset('W', data=pca.components_, compression='gzip', compression_opts=9)
        grp.attrs['explained_variance'] = pca.explained_variance_ratio_.sum()

    f_out.close()


if __name__ == '__main__':
    main()