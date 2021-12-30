import pandas as pd
import tensorflow as tf
from pandas.api.types import is_string_dtype


DATASETS = {
    'gmf': '/analyse/Project0257/lukas/data/gmf',
    'gmfmini': '/analyse/Project0257/lukas/data/gmfmini'
}


def create_dataset(df, X_col='filename', Y_col='id', Z_col=None, batch_size=256,
                 target_size=(224, 224, 3), validation_split=None, shuffle=True):
    
    # Shuffle rows of dataframe; much faster than
    # shuffling the Dataset object
    if shuffle:
        df = df.sample(frac=1)
    
    # Note to self: Dataset.list_files automatically sorts the input,
    # undoing the df.sample randomization! So use from_tensor_slices instead
    files = df[X_col].tolist()
    dataset = tf.data.Dataset.from_tensor_slices(files)    
    dataset = dataset.map(lambda f: _preprocess_img(f, target_size),
                         num_parallel_calls=tf.data.AUTOTUNE)

    # Extract and preprocess other input (Z) and output (Y) vars
    ds_tmp = {'Y': (), 'Z': ()}
    for name, cols in {'Y': Y_col, 'Z': Z_col}.items():
        if cols is None:
            continue
        
        if isinstance(cols, str):
            cols = [cols]
               
        for col in cols:
            v = df[col].to_numpy()  
            ds = tf.data.Dataset.from_tensor_slices(v)

            if is_string_dtype(df[col]):
                # FIXME: keys are not sorted!
                keys, _ = tf.unique(v)
                values = tf.range(0, len(keys))
                init = tf.lookup.KeyValueTensorInitializer(keys, values)
                table = tf.lookup.StaticHashTable(init, default_value=-1)
                ds = ds.map(lambda x: _preprocess_categorical(x, table))
                
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
    #dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)

    # Some optimization tricks
    if validation_split is not None:        
        n_val = int(df.shape[0] * validation_split)
        val = dataset.take(n_val)
        # Note to self: drop_remainder=True is necessary for
        # distributed training (not sure why ...)
        val = val.batch(batch_size, drop_remainder=True)
        val = val.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.skip(n_val)
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    #dataset = dataset.cache()

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
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [*target_size[:2]])
    img = img / 255.  # rescale
    return img

def _preprocess_categorical(z, table):
    """ Categorical variable-to-tensor, using
    some fance table lookup tf functionality. """
    z = table.lookup(z)
    z = tf.one_hot(z, depth=tf.cast(table.size(), tf.int32))
    return tf.cast(z, tf.int32)

def create_mock_model(n_inp, n_out, in_shape=None, n_y=30):
    
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
    df = pd.concat((df.iloc[:10, :], df.iloc[-10:, :]),axis=0)
    train = create_dataset(df, Y_col='bg', validation_split=None)
    for X, y in train.as_numpy_iterator():
        print(X)
        print(y)
    exit()
    #model = create_mock_model(n_inp=2, n_out=1, in_shape=[(224, 224, 3), (3,)])
    
    import sys
    sys.path.append('.')
    from src.models import ResNet6
    model = ResNet6(n_classes=30)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    model.fit(train, validation_data=val, epochs=5)