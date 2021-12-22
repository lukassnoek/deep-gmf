import sys
import click
import os.path as op
import pandas as pd
import tensorflow as tf

sys.path.append('.')
from src.models import MODELS
from src.utils.io import create_data_generator, DATASETS
from tensorflow.keras.optimizers import Adam


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--dataset', type=click.STRING, default='gmfmini')
@click.option('--target', type=click.STRING, default='id')
@click.option('--n-train', type=click.INT, default=None)
@click.option('--n-val', type=click.INT, default=None)
@click.option('--validation-split', type=click.FLOAT, default=0.1)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--epochs', type=click.INT, default=10)
@click.option('--lr', type=click.FLOAT, default=0.001)
@click.option('--gpu', type=click.INT, default=[0], multiple=True)
def main(model_name, dataset, target, n_train, n_val, validation_split, batch_size, epochs, lr, gpu):

    # Peek at data to determine number of output classes
    info = pd.read_csv(op.join(DATASETS[dataset], 'dataset_info.csv'))
    info = info.astype({target: str})
    n_classes = info[target].unique().size

    devices = [f"/gpu:{g}" for g in gpu]
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    
    with strategy.scope():
        model = MODELS[model_name](n_classes=n_classes)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

    train_gen, val_gen = create_data_generator(
        info, target, n=n_train, n_validation=n_val,
        validation_split=validation_split, batch_size=batch_size
    )
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(f'models/{model.name}_dataset-{dataset}')
    pd.DataFrame(history.history).to_csv(f'models/{model.name}_dataset-{dataset}_history.csv', index=False)


if __name__ == '__main__':

    main()
