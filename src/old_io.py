import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.data.ops.dataset_ops import Dataset


def create_data_generator(df_path, y, rescale=1./255, target_size=(224, 224),
                          validation_split=0.1, batch_size=256, seed=42, shuffle=True,
                          return_df=False):
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
    """
    # Load and shuffle
    if isinstance(df_path, str):
        df = pd.read_csv(df_path)
    else:
        df = df_path

    if shuffle:
        df = df.sample(frac=1)

    classes = df[y].unique().tolist()

    # Initialize data generator
    data_gen = ImageDataGenerator(rescale=rescale, validation_split=validation_split)
    
    # Create Tensorflow Dataset objects
    output_spec = (
        # `None` represents batch size
        tf.TensorSpec(shape=(None, target_size[0], target_size[0], 3),dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(classes)), dtype=tf.int32)
    )

    train_data = tf.data.Dataset.from_generator(
        lambda: data_gen.flow_from_dataframe(
            df, x_col='filename', y_col=y, subset='training',
            target_size=target_size, batch_size=batch_size, seed=seed,
            classes=classes,
        ),
        output_signature=output_spec
    )
    val_data = Dataset.from_generator(
        lambda: data_gen.flow_from_dataframe(
            df, x_col='filename', y_col=y, subset='validation',
            target_size=target_size, batch_size=batch_size, seed=seed,
            classes=classes
        ),
        output_signature=output_spec
    )

    # Avoid Tensorflow stdout vomit
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    to_return = (train_data, val_data)
    if return_df:
        to_return += (df,)

    return to_return

class GmfDataGenerator(Sequence):
    """
    
    Adapted from example by Arjun Muraleedharan (https://medium.com/
    analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-
    keras-1252b64e41c3).
    """
    def __init__(self, df, X_col='filename', Y_col='id', Z_col=None, batch_size=256,
                 target_size=(224, 224, 3), 
                 shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.Y_col = Y_col
        self.Z_col = Z_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.n = self.df.shape[0]
        self._setup()

    def _setup(self):
        """ Some extra things to do upon init. """        
        if self.shuffle:  # Shuffle the first time
            self.df = self.df.sample(frac=1)
            
        if isinstance(self.Y_col, str):
            self.Y_col = [self.Y_col]
        
        if isinstance(self.Z_col, str):
            self.Z_col = [self.Z_col]

    def on_epoch_end(self):
        """ Stuff to do after each epoch. """
        if self.shuffle:
            self.df = self.df.sample(frac=1)

    def _get_X(self, start, stop):
        
        paths = self.df[self.X_col].iloc[start:stop]
        X = np.zeros((self.batch_size, *self.target_size), dtype=np.float32)
        for i, path in enumerate(paths):
            img = load_img(path, target_size=self.target_size)
            X[i, ...] = img_to_array(img, data_format='channels_last')     
        
        X /= 255.  # normalize
        return X
    
    def _get_YZ(self, start, stop, cols):

        df = self.df.iloc[start:stop, :]
        data = ()
        for col in cols:
            series = df[col]
            if is_string_dtype(series):
                classes = self.df[col].unique()
                mapping = {c: i for i, c in enumerate(classes)}
                series = [mapping[c] for c in series]               
                data_ = to_categorical(series, num_classes=len(classes))
                data_ = data_.astype(np.int32)
            else:
                data_ = series.to_numpy()

            data += (data_,)
            
        return data       
        
    def __getitem__(self, idx):
        #return np.random.randn(256, 224, 244, 3), np.random.randint(0, 29, size=256)        
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        
        X = self._get_X(start, stop)      

        # Z represents extra inputs, so append to X
        if self.Z_col is not None:
            Z = self._get_YZ(start, stop, self.Z_col)
            X = (X,) + Z  # Z is always a tuple
    
        Y = self._get_YZ(start, stop, self.Y_col)
        # Y can, but doesn't have to be, multiple cols
        if len(Y) == 1:
            Y = Y[0]

        return X, Y

    def __len__(self):
        # Number of batches in the Sequence.
       return ceil(self.n / self.batch_size)
    
    def get_spec(self):
        X_shape = (self.batch_size, *self.target_size)
        inp_spec = (tf.TensorSpec(shape=X_shape, dtype=tf.float32),)
        
        if self.Z_col is not None:
            
            for col in self.Z_col: 
                series = self.df[col]
                if is_string_dtype(series):
                    n_target = series.nunique()
                    dtype = tf.int32
                else:
                    n_target = 1
                    dtype = tf.float32
            
            Z_shape = (self.batch_size, n_target)
            inp_spec += (tf.TensorSpec(shape=Z_shape, dtype=dtype),)
        else:
            inp_spec = inp_spec[0]

        out_spec = ()
        for col in self.Y_col:
            series = self.df[col]
            if is_string_dtype(series):
                n_target = series.nunique()
                dtype = tf.int32
            else:
                n_target = 1
                dtype = tf.float32
        
            Y_shape = (self.batch_size, n_target)
            out_spec += (tf.TensorSpec(shape=Y_shape, dtype=dtype),)
            
        if len(out_spec) == 1:
            out_spec = out_spec[0]           
        
        return inp_spec, out_spec


def create_dataset_from_generator(validation_split, **kwargs):
    """ Creates a Tensorflow Dataset from the custom
    GmfDataGenerator defined below. """

    gen = GmfDataGenerator(**kwargs)
    spec = gen.get_spec()

    # FIXME: spec doesn't work for multidim reg dvs
    dataset = Dataset.from_generator(
        lambda: gen,
        output_signature=spec
    )

    if validation_split is not None:
        n_val = ceil(gen.df.shape[0] * validation_split)
        val = dataset.take(n_val)
        train = dataset.skip(n_val)
        return train, val

    return dataset

