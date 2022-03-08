from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback


CLASS_OR_REG = {
    'id': 'classification',
    'ethn': 'classification',
    'age': 'regression',
    'gender': 'classification',
    'bg': 'classification',
    'l': 'classification',
    '3d': 'regression',
    'shape': 'regression',
    'tex': 'regression',
    'xr': 'regression',
    'yr': 'regression',
    'zr': 'regression',
    'xt': 'regression',
    'yt': 'regression',
    'zt': 'regression'
}


def add_head(body, targets, n_out):
    """ Adds one or more classification/regression heads to 
    an existing model ('body'). 
    
    Parameters
    ----------
    body : tf.keras.Model
        Existing Model (without a head)
    targets : tuple/list
        Tuple or list of strings with target labels.
        The length of `targets` indicates the number of
        heads
    n_out : tuple/list
        Number of output units per target
    """    
    x = body.output  # intermediate output from body
    
    y = []
    for n_o, target in zip(list(n_out), targets):
        act = None
        if CLASS_OR_REG[target] == 'classification':
            act = 'softmax'

        y.append(Dense(n_o, activation=act, name=target)(x))

    if len(y) == 1:
        y = y[0]

    model = Model(inputs=body.input, outputs=y, name=body.name)    
    return model


class EarlyStoppingOnLoss(Callback):
    """ Stops training when a particular value of `val_loss` has
    been observed. """    
    def __init__(self, monitor='val_loss', value=0.01, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        
        if current < self.value:
            self.model.stop_training = True
