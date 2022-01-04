import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


DATASETS = {
    'gmf': Path('/analyse/Project0257/lukas/data/gmf'),
    'gmf_random': Path('/analyse/Project0257/lukas/data/gmf_random'),    
}

# CATegorical variables & CONTinuous variables
CAT_COLS = ['id', 'ethn', 'age', 'gender', 'bg', 'l']
CONT_COLS = ['xr', 'yr', 'zr', 'xt', 'yt', 'zt']


def create_dataset(df, X_col='filename', Y_col='id', Z_col=None, batch_size=256,
                   target_size=(224, 224, 3), validation_split=None, shuffle=True):
    
    # Shuffle rows of dataframe; much faster than
    # shuffling the Dataset object
    if shuffle:
        df = df.sample(frac=1)
    
    # Note to self: Dataset.list_files automatically sorts the input,
    # undoing the df.sample randomization! So use from_tensor_slices instead
    files = df[X_col].tolist()
    dataset_files = tf.data.Dataset.from_tensor_slices(files)    
    dataset = dataset_files.map(lambda f: _preprocess_img(f, target_size),
                                num_parallel_calls=tf.data.AUTOTUNE)

    # Extract and preprocess other input (Z) and output (Y) vars
    ds_tmp = {'Y': (), 'Z': ()}
    for name, cols in {'Y': Y_col, 'Z': Z_col}.items():
        if cols is None:
            continue
        
        if isinstance(cols, str):
            cols = [cols]
               
        for col in cols:
            
            # shape/tex/3d are loaded from identity-specific
            # npz file using a custom (non-tf) function
            if col in ['shape', 'tex', '3d']:
                ds = dataset_files.map(
                    lambda f: tf.py_function(
                        func=_load_3d_data, inp=[f, col], Tout=tf.float32,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                ds_tmp[name] += (ds,)
                continue  # skip rest of function
            
            # Use series as tensor slices
            v = df[col].to_numpy()  
            ds = tf.data.Dataset.from_tensor_slices(v)

            # If categorical, encode as integers (table lookup)
            # and one-hot encode
            if col in CAT_COLS:
                # FIXME: keys are not sorted!
                keys, _ = tf.unique(v)
                values = tf.range(0, len(keys))
                init = tf.lookup.KeyValueTensorInitializer(keys, values)
                table = tf.lookup.StaticHashTable(init, default_value=-1)
                ds = ds.map(lambda x: _preprocess_categorical(x, table),
                            num_parallel_calls=tf.data.AUTOTUNE)
            elif col in CONT_COLS:
                # Cast to float32 (necessary for e.g. rotation params)
                ds = ds.map(lambda x: tf.cast(x, tf.float32))

            ds_tmp[name] += (ds,)
    
    if Z_col is not None:
        # Merge Z into inputs (X, images); for now,
        # do not nest Z (hence the *ds_tmp)
        dataset = tf.data.Dataset.zip((dataset, *ds_tmp['Z']))

    if len(ds_tmp['Y']) == 1:
        # If only one output, it should not be nested!
        ds_tmp['Y'] = ds_tmp['Y'][0] 

    # Merge inputs (X, Z) with outputs (Y)
    dataset = tf.data.Dataset.zip((dataset, ds_tmp['Y']))

    # Some optimization tricks
    if validation_split is not None:        
        n_val = int(df.shape[0] * validation_split)
        val = dataset.take(n_val)
        # Note to self: drop_remainder=True is necessary for
        # distributed training (not sure why ...)
        val = val.batch(batch_size, drop_remainder=True)
        val = val.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.skip(n_val)
        # Note to self: shuffling is done with the dataframe
         # (faster than using Dataset.shuffle)
    
    # Note to self: drop_remainder is necessary for multi-GPU
    # training (don't know why ...)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Get rid of tf stdout vomit
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)
    
    if validation_split is not None:
        val = val.with_options(options)
        return dataset, val
    else:
        return dataset


def _preprocess_img(file, target_size):
    """ Image-to-tensor (plus some preprocessing).
    To be used in dataset.map(). """
    img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [*target_size[:2]])
    img = img / 255.  # rescale
    return img


def _preprocess_categorical(z, table):
    """ Categorical variable-to-tensor, using
    some fance table lookup tf functionality. """
    z = table.lookup(z)
    z = tf.one_hot(z, depth=tf.cast(table.size(), tf.int32))
    return tf.cast(z, tf.int32)


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