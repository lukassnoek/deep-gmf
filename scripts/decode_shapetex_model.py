import sys
import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten

sys.path.append('.')
from src.io import create_dataset
from src.losses import AngleLoss
from src.metrics import RSquared
from src.layers import PCATransform


@click.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=False))
@click.option('--n-shape', default=300)
@click.option('--n-tex', default=300)
@click.option('--n-id-train', type=click.INT, default=None)
@click.option('--n-id-val', type=click.INT, default=None)
@click.option('--n-var-per-id', type=click.INT, default=None)
@click.option('--batch-size', default=512)
@click.option('--epochs', default=10)
@click.option('--lr', default=0.01)
def main(model_path, n_shape, n_tex, n_id_train, n_id_val, n_var_per_id, batch_size, epochs, lr):
    """ Learn a (linear PCA) compression of the feature representations
    in each layer of a given model (or, at least, the input/conv/globalpool layers.)
    """
        
    # Load full model (input --> target)
    model = load_model(model_path, custom_objects={'AngleLoss': AngleLoss})

    # Analyze only relevant layers, to speed up analysis
    layers2analyze = ['conv', 'flatten', 'globalpool', 'input']    
    layers = [l for l in model.layers if 'shortcut' not in l.name]
    layers = [l for l in layers if any([l2a in l.name for l2a in layers2analyze])]

    # Open PCA file with PCA parameters per layer
    pca_in = h5py.File(f'{str(model_path)}_compressed.h5', 'r')

    # Fit a shape and tex model separately
    for param, n_coeff in zip(['shape', 'tex'], [n_shape, n_tex]):
    
        # Load dataset info to pass to `create_dataset`
        model_name = str(Path(model_path).parent.name)
        dataset = model_name.split('dataset-')[1].split('_')[0]
        info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
        train, val = create_dataset(info, Y_col=param, batch_size=batch_size, n_id_train=n_id_train,
                                    n_id_val=n_id_val, n_var_per_id=n_var_per_id,
                                    n_shape=n_shape, n_tex=n_tex)

        if param == 'tex':
            n_coeff *= 5

        # Fit a new model per layer
        for layer in tqdm(layers):

            extractor = Model(inputs=model.input, outputs=layer.output)
            extractor.trainable = False
            
            # Add regression head!
            head = _create_head(extractor, param, n_coeff)

            # Create full model by concatenating extractor and head            
            full_model = Model(inputs=extractor.input, outputs=head.layers[-1].output, name='full_model')
            
            # Load PCA weights a set in full_model
            mu = pca_in[layer.name]['mu'][:]
            W = pca_in[layer.name]['W'][:]           
            full_model.get_layer('pca_transform').set_weights([mu, W])  # set weights!

            # Compile and fit!
            opt = Adam(learning_rate=lr)
            full_model.compile(optimizer=opt, loss='mse', metrics=[RSquared(reduce='median')], run_eagerly=True)
            full_model.fit(train, validation_data=val, epochs=epochs)                

            # Save full model (extractor + head)
            f_out = model_path + f'_target-{param}_layer-{layer.name}'
            full_model.save(f_out)
            
            #full_model = load_model(f_out, custom_objects={'PCATransform': PCATransform, 'r_squared': RSquared()})
            #full_model.predict(val, batch_size=batch_size)

    pca_in.close()
    #f_out.close()


def _create_head(extractor, param, n_coeff):
    
    x = Flatten(name='head_start')(extractor.layers[-1].output)
    x = PCATransform(name='pca_transform')(x)
    #x = BatchNormalization(momentum=0.5)(x)
    
    #if param == 'tex':
    #    kernel_init = TruncatedNormal(mean=0.0, stddev=0.001)
    #else:
    #    kernel_init = TruncatedNormal(mean=0.0, stddev=0.05)
    kernel_init = 'glorot_uniform'
    y = Dense(units=n_coeff, kernel_initializer=kernel_init, name='final_dense')(x)
    return Model(inputs=extractor.output, outputs=y)


if __name__ == '__main__':
    main()