import sys
import click
import shutil
import atexit
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.losses import TripletSemiHardLoss

sys.path.append('.')
from src.models import MODELS
from src.losses import AngleLoss
from src.io import create_dataset
from src.models.utils import add_prediction_head, add_embedding_head


TARGET_INFO = {
    # some info for loss function / metric; if the target is not listed
    # here, it is assumed to be categorical (so loss = cat. cross-ent)
    'tex': {'n_out': 1970, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    'shape': {'n_out': 394, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    'xr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'yr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'zr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'xt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'yt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'zt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'xl': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'yl': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'zl': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
}


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('--dataset', default='manyIDfewIMG')
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--n-id-train', type=click.INT, default=None)
@click.option('--n-id-val', type=click.INT, default=None)
@click.option('--n-var-per-id', type=click.INT, default=None)
@click.option('--n-shape', type=click.INT, default=None)
@click.option('--n-tex', type=click.INT, default=None)
@click.option('--query', type=str, default=None)
@click.option('--image-size', default=224)
@click.option('--epochs', default=10)
@click.option('--lr', default=1e-4)
@click.option('--gpu-id', default=0)
@click.option('--save-every-x-epochs', default=10)
@click.option('--use-triplet-loss', is_flag=True)
def main(model_name, dataset, target, batch_size, n_id_train, n_id_val,
         n_var_per_id, n_shape, n_tex, query, image_size, epochs,
         lr, gpu_id, save_every_x_epochs, use_triplet_loss):
    """ Main training function.
    
    Parameters
    ----------
    model_name : str
        Name of the model (architecture) to be used. Check src.models.__init__.py for the different
        model names
    dataset : str
        Name of dataset (like `manyIDfewIMG`)
    target : tuple
        Tuple with one or more strings indicating the target(s)
        to predict
    batch_size : int
        Batch size (used both for training and validation)
    n_id_train : int
        How many face IDs should be used for training (default: all);
        nice for quick testing
    n_id_val : int
        How many face IDs should be used for validation (default: all);
        nice for quick testing
    n_var_per_id : int
        How many images ("variations") per face ID should be used (default: all);
        nice for quick testing
    n_coeff : int
        Pick the first `n_coeff` coefficents to predict when the target is
        'tex' or 'shape' (otherwise it's ignored; default: all)
    query : str
        Query string to filter the feature dataframe; for example, "age < 50" or
        "(xr > -30) & (xr < 30)"
    image_size : int
        What the image size should be (assumes a square image)
    epochs : int
        How many epochs to train the model for; this is ignored when
        `target_val_loss` is set
    lr : float
        Learning rate for Adam optimizer
    gpu_id : int
        Which GPU to train the model on
    save_every_x_epochs : int
        After how many epochs the model should be saved
    """

    if use_triplet_loss and 'id' not in target:
        raise ValueError("Triplet loss can only be used with target 'id'!")

    # Load dataset info to pass to `create_dataset`
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')

    # Infer number of output variables, loss function(s), and metric(s)
    # for each target (may be >1)
    n_out, losses, metrics, weights = (), {}, {}, {}
    for t in target:
        if t not in TARGET_INFO:  # assume it's categorical!
            # Explicitly cast to str (so object -> categorical)
            info = info.astype({t: str})
            if t == 'id' and n_id_train is not None:
                n_out += (n_id_train,)
            else:
                n_out += (info.query("split != 'testing'")[t].nunique(),)
    
            losses[t] = 'categorical_crossentropy'
            metrics[t] = 'accuracy'
            weights[t] = 1.
        else:
            # Get n_out/loss/metrics from TARGET_INFO
            if t == 'shape' and n_shape is not None:
                n_out += (n_shape,)
            elif t == 'tex' and n_tex is not None:
                n_out += (n_tex * 5,)
            else:
                n_out += (TARGET_INFO[t]['n_out'],)

            losses[t] = TARGET_INFO[t]['loss']
            metrics[t] = TARGET_INFO[t]['metric']
            weights[t] = TARGET_INFO[t]['weight']

    if use_triplet_loss:
        losses = TripletSemiHardLoss()
        metrics = None
        weights = None

    # Create training and validation set with a specific target variable(s)
    target_size = (image_size, image_size, 3)
    train, val = create_dataset(info, Y_col=target, target_size=target_size,
                                batch_size=batch_size, n_id_train=n_id_train,
                                n_id_val=n_id_val, n_var_per_id=n_var_per_id,
                                n_shape=n_shape, n_tex=n_tex, query=query,
                                use_triplet_loss=use_triplet_loss)

    # Create 'body' and add head to it, which may have
    # multiple outputs (depends on `target`)
    body = MODELS[model_name]()
    
    if use_triplet_loss:
        model = add_embedding_head(body, n_embedding=512)
    else:
        model = add_prediction_head(body, target, n_out)

    # Compile with pre-specified losses/metrics
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=weights)
  
    target = '+'.join(list(target))
    if use_triplet_loss:
        target += 'triplet'
    
    model_dir = Path(f'trained_models/{model.name}_dataset-{dataset}_target-{target}')       
    model_dir.mkdir(parents=True, exist_ok=True)

    # Note that cardinality refers to number of steps per epoch when
    # the dataset is batched (like it is here)!
    steps_per_epoch = train.cardinality().numpy()  # for nice stdout
    
    callbacks = [ModelCheckpoint(
        filepath=str(model_dir) + '/epoch-{epoch:03d}',
        save_best_only=False,
        save_weights_only=False,
        # Save every x epochs
        save_freq=int(steps_per_epoch * save_every_x_epochs),
    )]

    # Save untrained model as well!
    model.save(f"{model_dir}/epoch-{'0'.zfill(3)}")

    # Train and save both model and history (loss/accuracy across epochs)
    with tf.device(f'/gpu:{gpu_id}'):
        history = model.fit(train, validation_data=val, epochs=epochs,
                            callbacks=callbacks)

    epochs = len(history.history['loss'])  # check actual number of epochs!
    model.save(f"{model_dir}/epoch-{epochs:03d}")
    pd.DataFrame(history.history).to_csv(f'{model_dir}/history.csv', index=False)


if __name__ == '__main__':
    main()