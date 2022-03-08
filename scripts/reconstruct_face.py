import os
os.environ['DISPLAY'] = ':0.0'

import sys
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from GFG.core import Ctx
from GFG.identity import IDModel, Nf
from GFG.core import Camera
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.models import MODELS
from src.io import create_test_dataset
from src.losses import AngleLoss

ST = np.load('data/idm_St.npy')
SV = np.load('data/idm_Sv.npy')


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--epoch', default=None)
@click.option('--dataset', default='manyIDmanyIMG')
@click.option('--n-samples', default=4)
def main(model_name, epoch, dataset, n_samples):
    
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    df_test = info.query("split == 'testing'")
    df_test = df_test.sample(n=n_samples)
    dataset_tf = create_test_dataset(df_test, batch_size=n_samples)
    X, shape, tex, bg = dataset_tf.__iter__().get_next()
    shape = shape.numpy()
    tex = tex.numpy()
    
    # Load model and filter layers
    full_model_name = f"{model_name}_dataset-{dataset}_target-shape"
    if epoch is None:
        all_models = Path(f'trained_models/{full_model_name}').glob('epoch-*')
        epoch = str(sorted(list(all_models))[-1]).split('epoch-')[1]

    f_in = f'trained_models/{full_model_name}/epoch-{epoch}'
    model = load_model(f_in, custom_objects={'AngleLoss': AngleLoss})
    shape_hat = model(X).numpy()
    X = X.numpy()
    
    IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
    idm = IDModel.load(IDM_PATH)
    tdet = np.load('/analyse/Project0294/GFG_data/tdet.npy')

    ctx = Ctx(hidden=True)

    # To init a camera, a model should exist, so using base_nf
    base_nf = Nf.from_default()
    base_nf.attach(ctx)
    ctx._camera[0] = Camera(
        ctx.win, (224, 224), 4.,
        target=[-11.5644, -13.0381, 0],
        eye = [-11.5644, -13.0381, 300],
        up = [0, 1, 0],
        FOV = 50,
        near = 100.,
        far = 1000.
    )
    ctx.assign_camera(0)
    base_nf.detach()

    fig, axes = plt.subplots(ncols=3, nrows=n_samples, figsize=(9, n_samples * 3))
    for i in range(shape_hat.shape[0]):
        
        shape_est = shape_hat[0, ...].squeeze() / SV
        shape_real = shape[0, ...].squeeze() / SV
        shape_random = np.random.normal(0, 1, size=394).astype(np.float16)
        this_tex = tex[0, ...].squeeze().reshape((5, 394)) / ST
        
        for ii, (suffix, sh) in enumerate([('est', shape_est), ('real', shape_real), ('random', shape_random)]):
            nf = idm.generate(
                sh, this_tex,
                ethnicity=df_test.iloc[i, :].loc['ethn'],
                gender=df_test.iloc[i, :].loc['gender'],
                age=df_test.iloc[i, :].loc['age'],
                basenf=base_nf, tdet=tdet
            )
            nf.attach(ctx)  # attach to openGL context
            img = ctx.render(dest='image')
            axes[i, ii].imshow(img)
            axes[i, ii].axis('off')
            axes[i, ii].set_title(suffix)           

            nf.detach()

    fig.savefig('test.png')


if __name__ == '__main__':
    main()