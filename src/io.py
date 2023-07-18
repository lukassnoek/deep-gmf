"""This module contains functions for loading and preprocessing the data for 
TensorFlow models. The images are assumed to be stored as JPG images and the 
metadata (ID parameters, shape, texture) as HDF5 files. The choice for HDF5 files is
probably not a good one, as there's no native TensorFlow function to load HDF5 files,
so we have to use a Python function, which means the entire pipeline cannot be 
jit compiled (as in in general slower than when using TF functions)."""

import h5py
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import StringLookup


# CATegorical variables & CONTinuous variables
CAT_COLS = ['id', 'ethn', 'gender', 'bg']
CONT_COLS = ['age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl']



def create_dataset(df, Y_col='id', batch_size=256, n_id_train=None, n_id_val=None,
                   n_var_per_id=None, n_shape=None, n_tex=None, query=None,
                   shuffle=True, n_cpu=10, binocular=False):
    """Creates a tf.data.Dataset object from a DataFrame. The DataFrame should contain
    the paths to the images and the metadata (ID parameters). The function returns
    a separate tf.dataDataset for the train and validation data.
    
    Parameters
    ----------
    Y_col : str, list, tuple
        The column(s) in the DataFrame that should be used as the target variable(s).
    batch_size : int
        The batch size; assumed to be the same for the train and validation dataset
    n_id_train : int
        The number of face IDs to use for training; if None, all IDs are used
    n_id_val : int
        The number of face IDs to use for validation; if None, all IDs are used
    n_var_per_id : int
        The number of variations per face ID to use for training; if None, all variations
        are used
    n_shape : int
        The number of shape components to load; if None, all shape components are loaded
    n_tex : int
        The number of texture components to load; if None, all texture components are loaded
    query : str
        A query string to filter the DataFrame (e.g., "yr > 0", to use only right-rotated
        images)
    shuffle : bool
        Whether to shuffle the dataset before batching
    n_cpu : int
        The number of CPU cores to use for loading the data
    binocular : bool
        Whether to load binocular images (i.e., left and right eye) or monocular images

    Returns
    -------
    ds_train : tf.data.Dataset
        The batched training dataset
    ds_val : tf.data.Dataset
        The batched validation dataset

    Raises
    ------
    ValueError
        If the requested `n_id_train`, `n_id_val`, or `n_var_per_id` is not possible
    """

    # Never train a model on test-set stimuli
    df = df.query("split != 'testing'")

    # Make sure Y_col and Z_col are lists by default
    if isinstance(Y_col, str):
        Y_col = [Y_col]

    if 'id' in Y_col:
        # If we're classifying face ID, the subsample of face IDs should be the same in
        # the train and val set
        df = df.query("split == 'training'")
        if n_id_train is not None:
            n_id_val = n_id_train

        if n_id_val is not None:
            n_id_train = n_id_val

    if query is not None:
        # Filter dataset with query string
        df = df.query(query)
    
    if 'id' in Y_col:
        # If 'id' is one of the targets, we cannot stratify according
        # to 'id', because that will lead to unique 'id' values in the val set
        if n_id_train is not None:
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
    for col in list(Y_col):
        if col is not None:
            if col in CAT_COLS:
                # Note to self: encode entire dataset (`df`) because otherwise
                # test IDs may not be represented in the one-hot-encoding
                df_comb = df_comb.astype({col: str})
                # StringLookup converts strings to integers and then
                # (with output_mode='one_hot') to a dummy representation
                
                slu = StringLookup(output_mode='one_hot', num_oov_indices=0)
                slu.adapt(df_comb[col].unique())
                cat_encoders[col] = slu

    if shuffle:
        # Make sure first epoch is also shuffled?
        df_train = df_train.sample(frac=1)

    train_val_datasets = []
    for ds_name, df_ in [('training', df_train), ('validation', df_val)]:
        # Note to self: Dataset.list_files automatically sorts the input,
        # undoing the df.sample randomization! So use from_tensor_slices instead
        for col in list(Y_col):
            if col in CAT_COLS:
                df_[col] = df_[col].apply(str)

        df_dataset = tf.data.Dataset.from_tensor_slices(dict(df_))
        
        if binocular:
            Xl_dataset = df_dataset.map(lambda row: _preprocess_img(row, 'image_path_left'),
                                    num_parallel_calls=n_cpu)
            Xr_dataset = df_dataset.map(lambda row: _preprocess_img(row, 'image_path_right'),
                                    num_parallel_calls=n_cpu)
            X_dataset = tf.data.Dataset.zip((Xl_dataset, Xr_dataset))
        else:
            X_dataset = df_dataset.map(lambda row: _preprocess_img(row, 'image_path'),
                                    num_parallel_calls=n_cpu)

        # Extract and preprocess output (Y) vars
        ds_tmp = []
        for col in Y_col:

            # shape/tex/3d are loaded from hdf5 file
            # using a custom (non-tf) function
            if col in ['shape', 'tex']:
                # Tensorflow cannot handle `None` values, so set to 0
                n_shape = 0 if n_shape is None else n_shape
                n_tex = 0 if n_tex is None else n_tex
                ds = df_dataset.map(
                    lambda row: tf.py_function(
                        func=_load_hdf_features, inp=[row['image_path'], col, n_shape, n_tex], Tout=tf.float32,
                    ),
                    num_parallel_calls=n_cpu
                )
                ds_tmp.append(ds)  # add to temporary dataset tuple
                continue  # skip rest of block

            if col in CAT_COLS:
                # Use previously created StringLookup layers
                # to one-hot encode string values
                ds = df_dataset.map(lambda x: cat_encoders[col](x[col]),
                                    num_parallel_calls=n_cpu)

            elif col in CONT_COLS:
                # Cast to float32 (necessary for e.g. rotation params)
                ds = df_dataset.map(lambda x: tf.cast(x[col], tf.float32))
            else:
                raise ValueError("Not sure whether {col} is categorical or continuous!")

            # Append to dict of Datasets
            ds_tmp.append(ds)
    
        if len(ds_tmp) == 1:
            # If only one output, it should not be nested!
            ds_tmp = ds_tmp[0]

        # Merge inputs (X, Z) with outputs (Y)
        dataset = tf.data.Dataset.zip((X_dataset, ds_tmp))

        # Get rid of tf stdout vomit
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        
        if ds_name == 'training':
            dataset = dataset.batch(batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(val_batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        train_val_datasets.append(dataset)

    ds_train, ds_val = train_val_datasets
    return ds_train, ds_val


def _preprocess_img(row, key='image_path'):
    """ Image-to-tensor (plus some preprocessing).
    To be used in dataset.map(). """
    img = tf.io.read_file(row[key])
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    img = img / 255.  # rescale
    return img


def _load_hdf_features(f, feature, n_shape, n_tex):
    """ Function to load in features from hdf5 file. To be used in
    a .map method call from a Tensorflow Dataset object. """
    f = f.numpy().decode('utf-8')
    feature = feature.numpy().decode('utf-8')
    to_replace = 'image.jpg'

    with h5py.File(f.replace(to_replace, 'features.h5'), 'r') as f_in:
        data = f_in[feature][:]

    if feature == 'tex' and n_tex != 0:
        data = data[:, :n_tex]
    
    if feature == 'shape' and n_shape != 0:
        data = data[:n_shape]

    if feature in ('tex',):
        data = data.T.flatten()
    
    return data


def create_test_dataset(df_test, batch_size=256, n_shape=None, n_tex=None, n_cpu=10,
                        binocular=False):
    
    # Note to self: no reason to shuffle dataset for testing!
    df_dataset = tf.data.Dataset.from_tensor_slices(dict(df_test))
    if binocular:
        l_dataset = df_dataset.map(lambda row: _preprocess_img(row, 'image_path_left'),
                                 num_parallel_calls=n_cpu)
        r_dataset = df_dataset.map(lambda row: _preprocess_img(row, 'image_path_right'),
                                 num_parallel_calls=n_cpu)
        dataset = tf.data.Dataset.zip((l_dataset, r_dataset))
    else:
        dataset = df_dataset.map(lambda row: _preprocess_img(row),
                                 num_parallel_calls=n_cpu)
    
    n_shape = 0 if n_shape is None else n_shape
    n_tex = 0 if n_tex is None else n_tex
    
    shape_tex_bg = []
    for feat in ('shape', 'tex'):
        # by default, we want all shape/tex coefficients
        key = 'image_path' if not binocular else 'image_path_left'
        ds = df_dataset.map(
            lambda row: tf.py_function(
                func=_load_hdf_features, inp=[
                    row[key], feat, n_shape, n_tex],
                Tout=tf.float32,
            ),
            num_parallel_calls=n_cpu
        )
        shape_tex_bg.append(ds)

    dataset = tf.data.Dataset.zip((dataset, *shape_tex_bg))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=n_cpu)
    
    return dataset
