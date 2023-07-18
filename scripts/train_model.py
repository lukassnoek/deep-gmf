"""CLI tool to train a DNN on a dataset generated with the GMF, potentially with
multiple-target objective functions."""

# Set maximum number of CPUs to 10 otherwise will use all available CPUs on the server
import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['NUMEXPR_MAX_THREADS'] = '10'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import sys
import click
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

sys.path.append('.')
from src.models import MODELS
from src.io import create_dataset
from src.models.utils import EarlyStoppingOnLoss, add_prediction_head, loop_over_layers


TARGET_INFO = {
    # some info for loss function / metric; if the target is not listed
    # here, it is assumed to be categorical (so loss = cat. cross-ent)
    'tex': {'n_out': 1970, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    'shape': {'n_out': 394, 'loss': 'mse', 'metric': 'cosine_similarity', 'weight': 1},
    # 'xr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    # 'yr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    # 'zr': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    'xt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'yt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    'zt': {'n_out': 1, 'loss': 'mse', 'metric': 'mae', 'weight': 0.001},
    # 'xl': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
    # 'yl': {'n_out': 1, 'loss': AngleLoss(), 'metric': None, 'weight': 1},
}


@click.command()
@click.option('--model', default='ResNet6', type=click.Choice(MODELS.keys()))
@click.option('--dataset', default='/analyse/Project0257/lukas/data/gmf_112x112_emo')
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('--batch-size', type=click.INT, default=256)
@click.option('--n-id-train', type=click.INT, default=None)
@click.option('--n-id-val', type=click.INT, default=None)
@click.option('--n-var-per-id', type=click.INT, default=None)
@click.option('--n-shape', type=click.INT, default=100)
@click.option('--n-tex', type=click.INT, default=100)
@click.option('--query', type=str, default=None)
@click.option('--epochs', default=10)
@click.option('--lr', default=1e-4)
@click.option('--gpu-id', default=0)
@click.option('--n-cpu', default=10)
@click.option('--stop-on-val', default=None, type=str)
@click.option('--jit-compile', is_flag=True)
@click.option('--prefix', default=None, type=click.STRING)
def main(model_name, dataset, target, batch_size, n_id_train, n_id_val,
         n_var_per_id, n_shape, n_tex, query, epochs, lr, gpu_id, n_cpu,
         stop_on_val, jit_compile, prefix):
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
    info = pd.read_csv(f'{dataset}.csv')
    # Create training and validation set with a specific target variable(s)

    binocular = model_name in ['StereoResNet10', 'StereoResNet6']
    target_size = (112, 112, 3)  # hard-coded for now
    train, val = create_dataset(info, Y_col=target, batch_size=batch_size, n_id_train=n_id_train,
                                n_id_val=n_id_val, n_var_per_id=n_var_per_id, n_shape=n_shape,
                                n_tex=n_tex, query=query, n_cpu=n_cpu, binocular=binocular)
    
    # Infer number of output variables, loss function(s), and metric(s)
    # for each target (may be >1)
    n_out, losses, metrics, weights = (), (), (), ()
    for t in target:
        if t not in TARGET_INFO:  # assume it's categorical!
            # Explicitly cast to str (so object -> categorical)
            info = info.astype({t: str})
            if t == 'id' and n_id_train is not None:
                n_out += (n_id_train,)
            else:
                n_out += (info.query("split == 'training'")[t].nunique(),)

            losses += ('categorical_crossentropy',)
            metrics += ('accuracy',)
            weights += (1.,)
        else:
            # Get n_out/loss/metrics from TARGET_INFO
            if t == 'shape' and n_shape is not None:
                n_out += (n_shape,)
            elif t == 'tex' and n_tex is not None:
                n_out += (n_tex * 5,)
            else:
                n_out += (TARGET_INFO[t]['n_out'],)

            losses += TARGET_INFO[t]['loss'],
            metrics += TARGET_INFO[t]['metric'],
            weights += TARGET_INFO[t]['weight'],

    with strategy.scope():
        # Create 'body' and add classification/regression head to it, which may have
        # multiple outputs (depends on `target` arg)
        body_class = MODELS[model_name]
        body = body_class(input_shape=target_size)
        n_layers = len(list(loop_over_layers(body))) - 1
        model = add_prediction_head(body, target, n_out, layer_nr=n_layers)

        # Compile with pre-specified losses/metrics
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=weights,
                      steps_per_execution=16, jit_compile=jit_compile)
        
    target = '+'.join(list(target))
    model_dir = Path(f'trained_models/{model.name}_dataset-{dataset}_target-{target}')       
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if stop_on_val is not None:
        # stop_on_val should be a string like "val_loss=0.1"
        metric, threshold = stop_on_val.split('=')
        callbacks.append(EarlyStoppingOnLoss(monitor=metric, value=float(threshold), verbose=0))
        epochs = 100

    # ReduceLRonPlateau substantially improves both train and val performance
    callbacks.append(ReduceLROnPlateau('loss', patience=2))
    callbacks.append(TensorBoard(log_dir='./tf_logs', update_freq='batch'))

    # Train and save both model and history (loss/accuracy across epochs)
    with tf.device(f'/gpu:{gpu_id}'):
        history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)

    epochs = len(history.history['loss'])  # check actual number of epochs!
    f_out = f'epoch-{epochs:03d}'
    if prefix is not None:
        f_out = f'{prefix}_{f_out}'
    
    # Save both and model and metrics per epoch as CSV file
    model.save(f"{model_dir}/{f_out}")
    pd.DataFrame(history.history).to_csv(f'{model_dir}/{f_out}_history.csv', index=False)


if __name__ == '__main__':
    main()
