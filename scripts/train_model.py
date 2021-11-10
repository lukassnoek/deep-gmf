import sys
import click
import os.path as op
import pandas as pd

sys.path.append('.')
from src.models import MODELS
from src.utils.io import create_data_generator, DATASETS
from tensorflow.keras.optimizers import Adam


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--dataset', type=click.STRING, default='human_exp')
@click.option('--target', type=click.STRING, default='face_id')
@click.option('--n-train', type=click.INT, default=16384)
@click.option('--n-val', type=click.INT, default=2048)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--epochs', type=click.INT, default=10)
@click.option('--lr', type=click.FLOAT, default=0.001)
def main(model_name, dataset, target, n_train, n_val, batch_size, epochs, lr):

    # Peek at data to determine number of output classes
    info = pd.read_csv(op.join(DATASETS[dataset], 'dataset_info.csv'))
    n_classes = info[target].unique().size

    model = MODELS[model_name](n_classes=n_classes)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

    train_gen, val_gen = create_data_generator(
        info, target, n=n_train, n_validation=n_val, batch_size=batch_size
    )
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(f'models/{model.name}_dataset-{dataset}')


if __name__ == '__main__':

    main()
