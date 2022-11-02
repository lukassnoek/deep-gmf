import tensorflow as tf


def RSquared(reduce='median', center=True):
    """ Computes R-squared with custom reduction (median) and
    optional centering. Mimics a Keras class, but is actually just a 
    function that returns a function in order to allow
    extra parameters.
    
    Parameters
    ----------
    reduce : str
        How to reduce the scores for all targets; 'median' (default)
        or mean (anything else)
    center : bool
        Whether to center the `y_true` and `y_pred` variables; note
        that setting this to `True` does not yield to usual/strict
        definition of R-squared!

    Returns
    -------
    r_squared : callable
        Function that computes R-squared
    """

    def r_squared(y_true, y_pred):
        y_true_mean = tf.reduce_mean(y_true, axis=0)

        if center:
            y_true = y_true - y_true_mean
            y_pred = y_pred - tf.reduce_mean(y_pred, axis=0)

        num = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
        denom = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=0)
        r_sq = 1 - (num / denom)

        if reduce == 'median':
            return _reduce_median(r_sq)
        else:
            return tf.reduce_mean(r_sq)
                    
    return r_squared


def _reduce_median(v):
    """ From Stackoverflow user BlueSun:
    https://stackoverflow.com/questions/43824665/tensorflow-median-value
    """
    v = tf.reshape(v, [-1])
    m = tf.shape(v)[0] // 2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)
