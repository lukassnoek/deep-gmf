import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import StringLookup, CategoryEncoding

DATASETS = {
    'gmf': Path('/analyse/Project0257/lukas/data/gmf'),
    'gmf_random': Path('/analyse/Project0257/lukas/data/gmf_random'),    
}

# CATegorical variables & CONTinuous variables
CAT_COLS = ['id', 'ethn', 'age', 'gender', 'bg', 'l']
CONT_COLS = ['xr', 'yr', 'zr', 'xt', 'yt', 'zt']


def create_dataset_test(df, X_col='filename', n_samples=512, batch_size=256, target_size=(224, 224, 3), n_id_test=None):
    """ Load test dataset. """

    df_test = df.query("split == 'testing'")
    if n_id_test is not None:
        ids = df_test['id'].unique().tolist()
        ids = random.sample(ids, n_id_test)
        df_test = df_test.query('id in @ids')

    df_test = df_test.sample(n=n_samples)

    files = df_test[X_col].to_numpy()
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = files_dataset.map(lambda f: _preprocess_img(f, target_size),
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    shape_tex = []
    for col in ('tex', 'shape'):
        ds = files_dataset.map(
            lambda f: tf.py_function(
                func=_load_3d_data, inp=[f, col], Tout=tf.float32,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        shape_tex.append(ds)
    
    dataset = tf.data.Dataset.zip((dataset, *shape_tex))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, df_test


def create_dataset(df, X_col='filename', Y_col='id', Z_col=None, batch_size=256,
                   n_id_train=None, n_id_val=None, target_size=(224, 224, 3),
                   shuffle=True, cache=False):

    # Make sure Y_col and Z_col are lists by default
    if isinstance(Y_col, (type(None), str)):
        Y_col = [Y_col]

    if isinstance(Z_col, (type(None), str)):
        Z_col = [Z_col]
    
    # Shuffle rows of dataframe; much faster than
    # shuffling the Dataset object!
    if shuffle:
        df = df.sample(frac=1)

    # Note: split created here instead of using the 
    # Dataset object, because using the DataFrame we
    # can stratify the split according to ID
    df_train = df.query("split == 'training'")

    if 'id' in Y_col:
        # If 'id' is one of the targets, we cannot
        # stratify according to 'id', because that 
        # will lead to unique 'id' values in the val set!
        if n_id_val is not None:
            # We'll 'abuse' the n_id_val parameter to now 
            # sample the number of observations for the val set!
            df_val = df_train.sample(n=n_id_val)
            df_train = df_train.drop(df_val.index, axis=0)
        else:
            # If there's no explicit number of obs for the val
            # set, sample 10% of the train set (arbitrary frac)
            df_val = df_train.sample(frac=0.1)
            df_train = df_train.drop(df_val.index, axis=0)

        if n_id_train is not None:
            # Again, abose the n_id_train parameter
            df_train = df_train.sample(n=n_id_train)

        # Make sure the dfs are not again subsampled
        n_id_train = None
        n_id_val = None
    else:
        # If 'id' is not a target, we can just nicely
        # stratify by ID (as done before in the script
        # `aggregate_dataset_info.py`) using the existing
        # split
        df_val = df.query("split == 'validation'")

    # Only relevant if 'id' is not a target!
    if n_id_train is not None:
        # Pick a subset of train IDs (for quick testing)
        ids = df_train['id'].unique().tolist()
        ids = random.sample(ids, n_id_train)
        df_train = df_train.query('id in @ids')

    if n_id_val is not None:
        # Pick a subset of val IDs (for quick testing)
        ids = df_val['id'].unique().tolist()
        ids = random.sample(ids, n_id_val)
        df_val = df_val.query('id in @ids')

    # For categorical features (in Y or Z), we need to infer
    # the categories *on the full dataset* (here: `df_comb`),
    # otherwise we risk that the train and val set are encoded
    # differently (e.g., 'M' -> 0 in train, 'M' -> in val)
    df_comb = pd.concat((df_train, df_val), axis=0)
    cat_encoders = {}  # we'll store them for later
    for col in list(Y_col) + list(Z_col):
        if col is not None:
            if col in CAT_COLS:
                df = df.astype({col: str})
                # StringLookup converts strings to integers and then
                # (with output_mode='one_hot') to a dummy representation
                slu = StringLookup(output_mode='one_hot', num_oov_indices=0)
                slu.adapt(df_comb[col].to_numpy())
                cat_encoders[col] = slu

    train_val_datasets = []
    for df in [df_train, df_val]:
        # `files` is a list with paths to images
        files = df[X_col].tolist()
        
        # Note to self: Dataset.list_files automatically sorts the input,
        # undoing the df.sample randomization! So use from_tensor_slices instead
        files_dataset = tf.data.Dataset.from_tensor_slices(files)

        # Load img + resize + normalize (/255)
        # Do not overwrite files_dataset, because we need it later
        X_dataset = files_dataset.map(lambda f: _preprocess_img(f, target_size),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        # Extract and preprocess other input (Z) and output (Y) vars
        ds_tmp = {'Y': (), 'Z': ()}
        for name, cols in {'Y': Y_col, 'Z': Z_col}.items():
                        
            for col in cols:

                if col is None:
                    continue

                # shape/tex/3d are loaded from identity-specific
                # npz file using a custom (non-tf) function
                if col in ['shape', 'tex', '3d']:
                    ds = files_dataset.map(
                        lambda f: tf.py_function(
                            func=_load_3d_data, inp=[f, col], Tout=tf.float32,
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                    ds_tmp[name] += (ds,)
                    continue  # skip rest of block
                
                # Below is necessary because sometimes pandas
                # treats numeric colums as object/str ...
                if col in CAT_COLS:
                    df = df.astype({col: str})
                else:
                    df = df.astype({col: float})

                # Use df column as tensor slices, to be preprocessed
                # in a way depending on whether it's a categorical or
                # continuous variable
                v = df[col].to_numpy()  
                ds = tf.data.Dataset.from_tensor_slices(v)

                if col in CAT_COLS:
                    # Use previously created StringLookup layers
                    # to one-hot encode string values
                    ds = ds.map(lambda x: cat_encoders[col](x),
                                num_parallel_calls=tf.data.AUTOTUNE)

                elif col in CONT_COLS:
                    # Cast to float32 (necessary for e.g. rotation params)
                    ds = ds.map(lambda x: tf.cast(x, tf.float32))
                else:
                    raise ValueError("Not sure whether {col} is categorical or continuous!")

                # Append to dict of Datasets
                ds_tmp[name] += (ds,)
        
        if Z_col is not None:
            # Merge Z into inputs (X, images); for now,
            # do not nest Z (hence the *ds_tmp; might change later)
            X_dataset = tf.data.Dataset.zip((X_dataset, *ds_tmp['Z']))

        if len(ds_tmp['Y']) == 1:
            # If only one output, it should not be nested!
            ds_tmp['Y'] = ds_tmp['Y'][0]

        # Merge inputs (X, Z) with outputs (Y)
        dataset = tf.data.Dataset.zip((X_dataset, ds_tmp['Y']))
    
        # Optimization tricks (drop_reminder is necessary for multi-GPU
        # training, don't know why)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Get rid of tf stdout vomit
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        
        if cache:
            dataset = dataset.cache()

        train_val_datasets.append(dataset)

    # Return as tuple    
    return train_val_datasets[0], train_val_datasets[1]


def _preprocess_img(file, target_size):
    """ Image-to-tensor (plus some preprocessing).
    To be used in dataset.map(). """
    img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [*target_size[:2]])
    img = img / 255.  # rescale
    return img


def _load_3d_data(f, col):
    f = f.numpy().decode('utf-8')
    col = col.numpy().decode('utf-8')
    data = np.load(str(Path(f).parent) + '.npz')
    if col == '3d':
        return np.r_[data['v_coeff'], data['t_coeff'].flatten()].astype(np.float32)
    elif col == 'shape':
        return data['v_coeff'].astype(np.float32)
    else:
        return data['t_coeff'].flatten().astype(np.float32)


def create_mock_model(n_inp, n_out, in_shape=None, n_y=30):
    # For testing purposes
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
    
    inp = []
    for i in range(n_inp):
        inp.append(Input(shape=in_shape[i], name=f'inp{i}'))
    
    out = []
    for i in range(n_out):
        x = GlobalAveragePooling2D()(inp[0])
        y = Dense(units=n_y)(x)
        out.append(y)

    if len(inp) == 1:
        inp = inp[0]
        
    if len(out) == 1:
        out = out[0]
    
    return Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    
    df = pd.read_csv('/analyse/Project0257/lukas/data/gmfmini/dataset_info.csv')
    df = df.query("id == 1").astype({'id': str, 'age': str, 'bg': str, 'l': str})
    train = create_dataset(df, Y_col='tex', validation_split=None)
    model = create_mock_model(n_inp=1, n_out=1, in_shape=[(224, 224, 3)], n_y=1970)
    model.compile(optimizer='adam', loss='mse', metrics='cosine_similarity')
    model.fit(train, epochs=5)    
    # import sys
    # sys.path.append('.')
    # from src.models import ResNet6
    # model = ResNet6(n_classes=(30,))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    # model.fit(train, epochs=5)