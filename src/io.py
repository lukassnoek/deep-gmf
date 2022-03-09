import h5py
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import StringLookup


DATA_PATH = Path('/analyse/Project0257/lukas/data')

# CATegorical variables & CONTinuous variables
CAT_COLS = ['id', 'ethn', 'gender']
CONT_COLS = ['age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl']


def create_test_dataset(df_test, X_col='image_path', batch_size=256, target_size=(224, 224, 3)):
    
    # Note to self: no reason to shuffle dataset for testing!
    files = df_test[X_col].to_numpy()
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = files_dataset.map(lambda f: _preprocess_img(f, target_size),
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    shape_tex_bg = []
    for feat in ('shape', 'tex', 'background'):
        # by default, we want all shape/tex coefficients
        ds = files_dataset.map(
            lambda f: tf.py_function(
                func=_load_hdf_features, inp=[f, feat, 0], Tout=tf.float32,
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        shape_tex_bg.append(ds)

    dataset = tf.data.Dataset.zip((dataset, *shape_tex_bg))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    # Get rid of tf stdout vomit
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    return dataset


def create_dataset(df, X_col='image_path', Y_col='id', Z_col=None, batch_size=256,
                   n_id_train=None, n_id_val=None, n_var_per_id=None, n_coeff=None,
                   query=None, target_size=(224, 224, 3), shuffle=True, cache=False):

    # Never train a model on test-set stimuli
    df = df.query("split != 'testing'")

    # Make sure Y_col and Z_col are lists by default
    if isinstance(Y_col, (type(None), str)):
        Y_col = [Y_col]

    if isinstance(Z_col, (type(None), str)):
        Z_col = [Z_col]
    
    if 'id' in Y_col:
        # If we're classifying face ID, the subsample of
        # face IDs should be the same in the train and val set
        if n_id_train is not None:
            n_id_val = n_id_train
        
        if n_id_val is not None:
            n_id_train = n_id_val

    if query is not None:
        df = df.query(query)
    
    if 'id' in Y_col:
        # If 'id' is one of the targets, we cannot stratify according
        # to 'id', because that will lead to unique 'id' values in the val set!
        if n_id_val is not None:  # `n_id_val`` always is the same as `n_id_train`
            # Pick a subset of IDs
            ids = df['id'].unique().tolist()
            ids = random.sample(ids, n_id_val)
            df = df.query('id in @ids')

        # Frac is hardcoded-for now, maybe not a good idea
        df_val = df.groupby('id').sample(frac=0.125)
        df_train = df.drop(df_val.index, axis=0)

        # Make sure the dfs are not again subsampled
        n_id_train = None
        n_id_val = None
    else:
        # Note: split created here instead of using the 
        # Dataset object, because using the DataFrame we
        # can stratify the split according to ID
        df_train = df.query("split == 'training'")

        # If 'id' is not a target, we can just nicely
        # stratify by ID (as done before in the script
        # `aggregate_dataset_info.py`) using the existing split
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

    if n_var_per_id is not None:
        # Select, per face ID, a specific number of images;
        # this is useful for quick testing!
        min_var_per_id = df_train.groupby('id').size().min()
        if min_var_per_id < n_var_per_id:
            raise ValueError(f"Cannot sample {n_var_per_id} per ID because "
                             f"minimum number of variations is {min_var_per_id}!")

        df_train = df_train.groupby('id').sample(n=n_var_per_id)
        
        # Only subsample the validation data if the number of images
        # per ID is larger than `n_var_per_id` (not always the case when
        # target is `id`!)
        if np.min(df_val.groupby('id').size()) > n_var_per_id:
            df_val = df_val.groupby('id').sample(n=n_var_per_id)

    if df_train.shape[0] < batch_size:
        raise ValueError(f"Train dataset has fewer samples {df_train.shape[0]} "
                         f"than the requested batch size {batch_size}!")

    if df_val.shape[0] < batch_size:
        print(f"WARNING: number of validation samples ({df_val.shape[0]}) "
              f"is smaller than the batch size ({batch_size})! "
              f"Setting validation batch size to {df_val.shape[0]}.")
        val_batch_size = df_val.shape[0]
    else:
        val_batch_size = batch_size

    # For categorical features (in Y or Z), we need to infer
    # the categories *on the full dataset* (here: `df_comb`),
    # otherwise we risk that the train and val set are encoded
    # differently (e.g., 'M' -> 0 in train, 'M' -> in val)
    df_comb = pd.concat((df_train, df_val), axis=0)
    cat_encoders = {}  # we'll store them for later
    for col in list(Y_col) + list(Z_col):
        if col is not None:
            if col in CAT_COLS:
                # Note to self: encode entire dataset (`df`) because otherwise
                # test IDs may not be represented in the one-hot-encoding
                df_comb = df_comb.astype({col: str})
                # StringLookup converts strings to integers and then
                # (with output_mode='one_hot') to a dummy representation
                slu = StringLookup(output_mode='one_hot', num_oov_indices=0)
                slu.adapt(df_comb[col].to_numpy())
                cat_encoders[col] = slu

    # Shuffle rows of dataframe; much faster than
    # shuffling the Dataset object!
    #if shuffle:
    #    df_train = df_train.sample(frac=1)
    #    df_val = df_val.sample(frac=1)
        
    train_val_datasets = []
    for ds_name, df_ in [('training', df_train), ('validation', df_val)]:
        # `files` is a list with paths to images
        files = df_[X_col].tolist()
        
        # Note to self: Dataset.list_files automatically sorts the input,
        # undoing the df.sample randomization! So use from_tensor_slices instead
        files_dataset = tf.data.Dataset.from_tensor_slices(files)

        # Note to self: I think Keras does not automatically shuffle the data when passing it
        # a tensorflow Dataset object, so we need to explicitly shuffle it here
        # Also, we do it here (instead of at the end), because the files are ony a pointer, so
        # it's much faster
        if shuffle and ds_name == 'training':
            files_dataset = files_dataset.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)

        # Load img + resize + normalize (/255)
        # Do not overwrite files_dataset, because we need it later
        X_dataset = files_dataset.map(lambda f: _preprocess_img(f, target_size),
                                      num_parallel_calls=tf.data.AUTOTUNE)

        # Extract and preprocess other input (Z) and output (Y) vars
        ds_tmp = {'Y': (), 'Z': ()}
        for name, cols in {'Y': Y_col, 'Z': Z_col}.items():
                        
            for col in cols:

                if col is None:
                    # More often than not, there are no `Z` cols
                    continue

                # shape/tex/3d are loaded from hdf5 file
                # using a custom (non-tf) function
                if col in ['shape', 'tex', 'background']:
                    # Tensorflow cannot handle `None` values, so set to 0
                    n_coeff = 0 if n_coeff is None else n_coeff
                    ds = files_dataset.map(
                        lambda f: tf.py_function(
                            func=_load_hdf_features, inp=[f, col, n_coeff], Tout=tf.float32,
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                    ds_tmp[name] += (ds,)  # add to temporary dataset tuple
                    continue  # skip rest of block
                
                # Below is necessary because sometimes pandas
                # treats numeric colums as object/str ...
                if col in CAT_COLS:
                    df_ = df_.astype({col: str})
                else:
                    df_ = df_.astype({col: float})

                # Use df column as tensor slices, to be preprocessed
                # in a way depending on whether it's a categorical or
                # continuous variable
                v = df_[col].to_numpy()  
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
    
        # Optimization tricks (drop_remainder is necessary for multi-GPU
        # training, don't know why)
        if ds_name == 'training':
            dataset = dataset.batch(batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(val_batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Get rid of tf stdout vomit
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        
        if cache:
            dataset = dataset.cache()

        train_val_datasets.append(dataset)

    # Return as tuple instead of list   
    return train_val_datasets[0], train_val_datasets[1]


def _preprocess_img(file, target_size):
    """ Image-to-tensor (plus some preprocessing).
    To be used in dataset.map(). """
    img = tf.io.read_file(file)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [*target_size[:2]])
    img = img / 255.  # rescale
    return img


def _load_hdf_features(f, feature, n_coeff):
    f = f.numpy().decode('utf-8')
    feature = feature.numpy().decode('utf-8')
    with h5py.File(f.replace('image.png', 'features.h5'), 'r') as f_in:
        data = f_in[feature][:].astype(np.float32)

    if feature == 'tex' and n_coeff != 0:
        data = data[:, :n_coeff]
    
    if feature == 'shape' and n_coeff != 0:
        data = data[:n_coeff]

    if feature in ('tex', 'background'):
        data = data.flatten()
    
    return data


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
    
    df = pd.read_csv('/analyse/Project0257/lukas/data/gmf_manyIDfewIMG.csv')
    train = create_dataset(
        df, X_col='image_path', Y_col=['shape', 'tex'], Z_col=None, batch_size=256,
        n_id_train=None, n_id_val=None, target_size=(224, 224, 3),
        shuffle=True, cache=False
    )
    #model = create_mock_model(n_inp=1, n_out=1, in_shape=[(224, 224, 3)], n_y=1970)
    #model.compile(optimizer='adam', loss='mse', metrics='cosine_similarity')
    #model.fit(train, epochs=5)    