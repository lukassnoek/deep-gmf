import sys
import click
import os.path as op
import pandas as pd
import tensorflow as tf

sys.path.append('.')
from src.models import MODELS
from src.io import create_dataset, DATASETS
from tensorflow.keras.optimizers import Adam


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--dataset', type=click.STRING, default='gmfmini')
@click.option('--target', type=click.STRING, default='id')
@click.option('--validation-split', type=click.FLOAT, default=0.1)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--epochs', type=click.INT, default=10)
@click.option('--lr', type=click.FLOAT, default=0.001)
@click.option('--gpu', type=click.INT, default=[0], multiple=True)
def main(model_name, dataset, target, validation_split, batch_size, epochs, lr, gpu):

    # Peek at data to determine number of output classes
    info = pd.read_csv(op.join(DATASETS[dataset], 'dataset_info.csv'))
    info = info.astype({target: str})#.sample(frac=0.5)
    n_classes = info[target].nunique()
    loss_f = 'categorical_crossentropy'

    devices = [f"/gpu:{g}" for g in gpu]
    batch_size = batch_size * len(devices)
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    
    train, val = create_dataset(info, Y_col=target, validation_split=validation_split, batch_size=batch_size)
    with strategy.scope():
    
        model = MODELS[model_name](n_classes=n_classes)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss_f, metrics=None)

    steps_per_epoch = train.cardinality().numpy()
    history = model.fit(train, validation_data=val, epochs=epochs, steps_per_epoch=steps_per_epoch)
    f_out = f'models/{model.name}_dataset-{dataset}-target-{target}'
    model.save(f_out)
    pd.DataFrame(history.history).to_csv(f'{f_out}_history.csv', index=False)


if __name__ == '__main__':

    main()
