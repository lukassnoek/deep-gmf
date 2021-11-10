import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generator(df_path, y, n=None, n_validation=None,
                          rescale=1./255, target_size=(224, 224), batch_size=128, seed=42,
                          return_df=False, **kwargs):
    """ Creates a ImageDataGenerator object from a dataframe with
    filenames and target variable(s). 
    
    Note to self: make sure n (and maybe also n_validation) is a multiple of batch_size
    """

    # Load and shuffle
    df = pd.read_csv(df_path).sample(frac=1, replace=False)
    if n is not None:  # `n` takes precendence over `frac`
        df_train = df.sample(n=n, replace=False)
    else:
        df_train = df

    if n_validation is not None:
        df_val = df.drop(df_train.index, axis=0)
        df_val = df_val.sample(n_validation, replace=False)

    # Initialize data generator
    data_gen = ImageDataGenerator(
        rescale=rescale, 
    )

    train_gen = data_gen.flow_from_dataframe(
        df_train, x_col='file', y_col=y, target_size=target_size,
        batch_size=batch_size, seed=seed, **kwargs
    )

    to_return = (train_gen,)
    if n_validation is not None:
        
        val_gen = data_gen.flow_from_dataframe(
            df_val, x_col='file', y_col=y, target_size=target_size,
            batch_size=batch_size, seed=seed, **kwargs
        )
        to_return += (val_gen,)
        if return_df:
            to_return += (df_train, df_val) 
    else:
        if return_df:
            to_return += (df_train,)

    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return