import sys
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from scipy.spatial.distance import pdist, squareform

sys.path.append('.')
from src.utils.io import create_data_generator
from src.models import MODELS


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('-n', '--n-samples', type=click.INT, default=512)
def main(model_name, n_samples):

    model = load_model(f'models/{model_name}')
    extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers[::5]])

    train_gen, train_df = create_data_generator(
        'data/human_exp/dataset_info.csv', 'face_id',
        n=n_samples, batch_size=n_samples, return_df=True, shuffle=False
    )  # set shuffle to False so train_gen and train_df have the same order

    # Get data and reorder according to y (nice for RDM viz)
    X, y = next(train_gen)
    reord = np.argmax(y, axis=1).argsort()

    # Extract layer representations (Z is a list with layer maps)
    Z = extractor(X[reord, ...])

    # Plot RDM per layer and correlation (DNN, faceID)
    fig = plt.figure(constrained_layout=False, figsize=(len(Z) * 2, 5))
    gs = fig.add_gridspec(nrows=2, ncols=len(Z))
    r_ax = fig.add_subplot(gs[1, :])
    for y_name in ['face_id', 'sex']:

        y = pd.get_dummies(train_df.loc[:, y_name].iloc[reord]).to_numpy()
        y_rdv = pdist(y, metric='euclidean')
        y_rdm = squareform(y_rdv)

        r = np.zeros(len(Z))

        for i, z in enumerate(Z):
            z_rdm = 1 - np.corrcoef(z.numpy().reshape(X.shape[0], -1))
            z_rdv = squareform(z_rdm.round(3))
            r[i] = pearsonr(y_rdv, z_rdv)[0]

            ax = fig.add_subplot(gs[0, i])
            ax.imshow(z_rdm)
            ax.axis('off')
            ax.set_title(f'DNN layer {i+1}')

        r_ax.plot(r)

    r_ax.spines['right'].set_visible(False)
    r_ax.spines['top'].set_visible(False)
    r_ax.set_ylabel('Corr(DNN, target)', fontsize=15)
    r_ax.set_xlabel('Layer nr.', fontsize=15)
    r_ax.set_xticks(range(len(Z)))
    r_ax.set_xticklabels(range(1, len(Z) + 1))
    fig.tight_layout()
    plt.savefig(f'figures/model-{model_name}_target-faceID_rdms.png')


if __name__ == '__main__':

    main()