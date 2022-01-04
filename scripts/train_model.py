import sys
import click
import os.path as op
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

sys.path.append('.')
from src.models import MODELS
from src.models.utils import add_head
from src.losses import AngleLoss
from src.io import create_dataset, DATASETS


TARGET_INFO = {
    # some info for loss function / metric; if the target is not listed
    # here, it is assumed to be categorical (so loss = cat. cross-ent)
    'tex': {'n_out': 1970, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    'shape': {'n_out': 394, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    '3d': {'n_out': 2364, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    'xr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'yr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'zr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'xt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'yt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
}


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--dataset', type=click.STRING, default='gmfmini')
@click.option('--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--validation-split', type=click.FLOAT, default=0.1)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--image-size', type=click.INT, default=224)
@click.option('--epochs', type=click.INT, default=10)
@click.option('--lr', type=click.FLOAT, default=0.001)
@click.option('--n-gpu', type=click.INT, default=1)
def main(model_name, dataset, target, validation_split, batch_size, image_size, epochs, lr, n_gpu):
    """ Main training function. """

    # Load dataset info to pass to `create_dataset`
    info = pd.read_csv(op.join(DATASETS[dataset], 'dataset_info.csv'))
    #info = info.query("id < 2")#sample(n=20_000)

    # Infer number of output variables, loss function(s), and metric(s)
    # for each target (may be >1)
    n_out, losses, metrics, weights = (), {}, {}, {}
    for t in target:
        if t not in TARGET_INFO:  # assume it's categorical!
            # Explicitly cast to str (so object -> categorical)
            info = info.astype({t: str})
            n_out += (info[t].nunique(),)
            losses[t] = 'categorical_crossentropy'
            metrics[t] = 'accuracy'
            weights[t] = 1.
        else:
            # Get n_out/loss/metrics from TARGET_INFO
            n_out += (TARGET_INFO[t]['n_out'],)
            losses[t] = TARGET_INFO[t]['loss']
            metrics[t] = TARGET_INFO[t]['metric']
            weights[t] = TARGET_INFO[t]['weight']

    # Manage multi GPU training
    devices = [f"/gpu:{g}" for g in range(n_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    # Create training and validation set with a specific target variable(s)
    target_size = (image_size, image_size, 3)
    train, val = create_dataset(info, Y_col=target, validation_split=validation_split,
                                target_size=target_size, batch_size=batch_size)

    # For multiple GPU training!
    with strategy.scope():  
        # Create 'body' and add head to it, which may have
        # multiple outputs (depends on `target`)
        body = MODELS[model_name]()
        model = add_head(body, target, n_out)

        # Compile with pre-specified losses/metrics
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=weights)
 
    # Train and save both model and history (loss/accuracy across epochs)
    steps_per_epoch = train.cardinality().numpy()  # need for nice stdout
    history = model.fit(train, validation_data=val, epochs=epochs, steps_per_epoch=steps_per_epoch)
    target = '+'.join(list(target))
    f_out = f'models/{model.name}_dataset-{dataset}_target-{target}'
    model.save(f_out)
    pd.DataFrame(history.history).to_csv(f'{f_out}_history.csv', index=False)


if __name__ == '__main__':

    main()
