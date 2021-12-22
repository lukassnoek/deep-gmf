import pandas as pd
import os.path as op
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASETS = {
    'mini_gmf_dataset': '/analyse/Project0257/lukas/data/mini_gmf_dataset',
    'gmfmini': '/analyse/Project0257/lukas/data/gmfmini'
}


def create_data_generator(df_path, y, n=None, n_validation=None, validation_split=0.0, 
                          rescale=1./255, target_size=(256, 256), batch_size=128, seed=42,
                          return_df=False, **kwargs):
    """ Creates a ImageDataGenerator object from a dataframe with
    filenames and target variable(s). 
    
    Note to self: make sure n (and maybe also n_validation) is a multiple of batch_size.
    
    Parameters
    ----------
    df_path : str / DataFrame
        Either a path to CSV file with dataset information
        or the DataFrame itself with the information
    y : str
        Name of column to use for `y`
    n : int
        Number of observations for train
    n_validation : int
        Number of observations for validation (if None, no validation)
     
    """

    # Load and shuffle
    if isinstance(df_path, str):
        df = pd.read_csv(df_path)
    else:
        df = df_path

    if n_validation is not None:
        if validation_split != 0.0:
            validation_split = 0.0

    # Always shuffle!
    df = df.sample(frac=1)
    
    if n is not None:
        if 'split' in df.columns:
            df_train = df.query("split == 'train'").sample(n=n)
        else:
            df_train = df.sample(n=n, replace=False)
    else:
        df_train = df

    if n_validation is not None:
        if 'split' in df.columns:
            df_val = df.query("split == 'val'").sample(n=n_validation)
        else:
            df_val = df.drop(df_train.index, axis=0)
            df_val = df_val.sample(n=n_validation, replace=False)

    # Initialize data generator
    data_gen = ImageDataGenerator(
        rescale=rescale, validation_split=validation_split
    )

    subset = 'training' if validation_split != 0.0 else None
    train_gen = data_gen.flow_from_dataframe(
        df_train, x_col='filename', y_col=y, subset=subset,
        target_size=target_size, batch_size=batch_size, seed=seed,
        **kwargs
    )

    to_return = (train_gen,)
    if n_validation is not None:
        val_gen = data_gen.flow_from_dataframe(
            df_val, x_col='filename', y_col=y,
            target_size=target_size, batch_size=batch_size, seed=seed, **kwargs
        )
        to_return += (val_gen,)
        if return_df:
            to_return += (df_train, df_val) 
    elif validation_split != 0.0:
        val_gen = data_gen.flow_from_dataframe(
            df_train, x_col='filename', y_col=y, subset='validation',
            target_size=target_size, batch_size=batch_size, seed=seed, **kwargs
        )
        to_return += (val_gen,)
    else:
        if return_df:
            to_return += (df_train,)

    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return