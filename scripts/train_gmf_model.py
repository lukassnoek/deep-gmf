import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import sys
import time
import click
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

sys.path.append('.')
from src.models import MODELS
from src.losses import AngleLoss
from src.io import create_dataset
from src.models.utils import EarlyStoppingOnLoss
from src.models.utils import add_prediction_head


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
@click.option('--dataset', default='gmf_112x112')
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--n-id-train', type=click.INT, default=None)
@click.option('--n-id-val', type=click.INT, default=None)
@click.option('--n-var-per-id', type=click.INT, default=None)
@click.option('--n-shape', type=click.INT, default=None)
@click.option('--n-tex', type=click.INT, default=None)
@click.option('--query', type=str, default=None)
@click.option('--image-size', default=112)
@click.option('--epochs', default=10)
@click.option('--lr', default=1e-4)
@click.option('--gpu-id', default=0)
@click.option('--n-cpu', default=32)
@click.option('--stop-on-val-loss', default=None, type=click.FLOAT)
@click.option('--binocular', is_flag=True)
def main(model_name, dataset, target, batch_size, n_id_train, n_id_val,
         n_var_per_id, n_shape, n_tex, query, image_size, epochs,
         lr, gpu_id, n_cpu, stop_on_val_loss, binocular):
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
    n_shape : int
        Number of shape components to load. Default is to load all (394)
    n_tex : int
        Number of texture components to load. Note that indicates `n_tex`
        *per spatial frequency band*, so `n_tex = 5` gives (5 * 5 = ) 25
        components
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
    """

    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", '/gpu:3'])
    strategy = tf.distribute.OneDeviceStrategy(f'/gpu:{gpu_id}')
    batch_size *= strategy.num_replicas_in_sync
    
    # Load dataset info to pass to `create_dataset`
    info = pd.read_csv(f'/analyse/Project0257/lukas/data/{dataset}.csv')
    # Create training and validation set with a specific target variable(s)
    target_size = (image_size, image_size, 3)
    train, val = create_dataset(info, Y_col=target, target_size=target_size,
                                batch_size=batch_size, n_id_train=n_id_train,
                                n_id_val=n_id_val, n_var_per_id=n_var_per_id,
                                n_shape=n_shape, n_tex=n_tex, query=query,
                                n_cpu=n_cpu, arcface=model_name == 'ArcFace', binocular=binocular)

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

    mixed_precision.set_global_policy('mixed_float16')
    
    with strategy.scope():
        # Create 'body' and add head to it, which may have
        # multiple outputs (depends on `target`)
        body_class = MODELS[model_name]
        if model_name == 'ArcFace':
            model = body_class(input_shape=target_size, num_classes=n_out[-1])
        else:
            body = body_class(input_shape=target_size)
            model = add_prediction_head(body, target, n_out)

        # Compile with pre-specified losses/metrics
        #opt = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=weights,
                    steps_per_execution=32, jit_compile=True)

    target = '+'.join(list(target))
    model_dir = Path(f'trained_models/{model.name}_dataset-{dataset}_target-{target}')       
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if stop_on_val_loss is not None:
        callbacks.append(EarlyStoppingOnLoss(value=stop_on_val_loss, verbose=0))
        epochs = 100

    callbacks.append(ReduceLROnPlateau('loss', patience=5))
    callbacks.append(TensorBoard(log_dir='./tf_logs', update_freq='batch'))
    
    # Train and save both model and history (loss/accuracy across epochs)
    with tf.device(f'/gpu:{gpu_id}'):
        history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)

    epochs = len(history.history['loss'])  # check actual number of epochs!
    model.save(f"{model_dir}/xepoch-{epochs:03d}")
    pd.DataFrame(history.history).to_csv(f'{model_dir}/history.csv', index=False)


if __name__ == '__main__':
    start = time.time()
    main()
    duration = time.time() - start
    print(f"Training took {duration} seconds!")