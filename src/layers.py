import math
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ArcMarginPenaltyLogits(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogits"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[input_shape[-1], self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, self.logist_scale, 'arcface_logits')
        return logits


class PCATransform(Layer):
    """ Linear transform of data into a lower-dimensional
    space using an (already fitted) PCA decomposition.
    
    If using fixed weights (mu, W), *first* initialize the
    entire model (a `tf.keras.Model` object) and call the
    layer's `set_weights` method.
    
    Parameters
    ----------
    n_comp : int
        Number of PCA components (needed for initialization of
        weights array)
    trainable : bool
        Whether the linear transform parameters (W and mu) are
        trainable; if not, the weights need to be set using
        the `set_weights(mu, W)` method
    name : str
        Name of layer
    **kwargs : dict
        Other keyword parameters passed to the Keras Layer
        init method
        
    Examples
    --------
    Assuming you have a numpy array for `mu` and `W`, you
    can set them as follows:
    
    >>> pca = PCATransform(n_comp=500)
    >>> y = pca(x)
    >>> model = tf.keras.Model(input=x, output=y)
    >>> model.get_layer('pca_transform').set_weights(mu, W)
    """
    def __init__(self, n_comp=500, trainable=False, name='pca_transform', **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.n_comp = n_comp
        
    def build(self, input_shape):
        self.mu = self.add_weight(shape=(input_shape[1],), trainable=False, name='mu')
        self.W = self.add_weight(shape=(input_shape[1], self.n_comp), trainable=False, name='W')
        
    def call(self, inputs):
        return tf.matmul(tf.subtract(inputs, self.mu), self.W)
        

class Normalization(Layer):
    """ Batch normalization, but the parameters are not learned. 
    This is helpful in "decoding" blocks in which we don't care
    about learning biases/intercepts of the linear models mapping
    a layer to target variables (such as shape components).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        inputs_centered = tf.subtract(inputs, tf.reduce_mean(inputs, axis=0))
        return tf.divide(inputs_centered, tf.math.reduce_std(inputs))


@tf.custom_gradient
def _grad_reverse(x):
    """ Pass through of input (`x`) and inverts gradient. 
    
    Adapted from: https://stackoverflow.com/questions/56841166/
    how-to-implement-gradient-reversal-layer-in-tf-2-0

    """
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy

    return y, custom_grad


class InvertGradient(Layer):
    """ Layer implementation of _grad_reverse
    function above. """

    def call(self, x):
        return _grad_reverse(x)

    
class Rdm(Layer):
    def __init__(self, add_loss_=False, batch_size=512, add_loss=False, pearson=True):
        """ Representational Dissimilarity Matrix (RDM) layer. 

        Parameters
        ----------
        add_loss_ : bool
            Whether to add the loss to the network this layer is used in
            (which will just return in input when calling the layer) or
            whether to just return the loss (when using it inside a custom
            loss function, like the `cka_loss` below)
        batch_size : int
            Ugly hack to get this layer to work as part of a network
            (instead of just a loss function), which needs know the batch
            size. Should be fixed to infer it upon runtime
        """
        super().__init__()
        self.add_loss_ = add_loss_
        self.batch_size = batch_size
        self.add_loss = add_loss
        self.pearson = pearson
        self.rdm_N = None  # "Neural" RDM 
        self.rdm_F = None  # "Feature" RDM

    def call(self, a_N, a_F):
       
        self.rdm_N = self.get_rdm(a_N)
        self.rdm_F = self.get_rdm(a_F)

        if self.pearson:
            l = self._pearson(
                self._squareform(self.rdm_N),
                self._squareform(self.rdm_F)
            )

        if self.add_loss_:
            self.add_loss(l)
            return self.rdm_N
        else:
            return l

    def get_rdm(self, Z):
        raise NotImplementedError
     
    def _squareform(self, rdm):
        """Squareform operation (get upper triangle from matrix). """        
        ones = tf.ones_like(rdm)
        mask_a = tf.linalg.band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask

        return tf.boolean_mask(rdm, mask)

    def _pearson(self, x, y):
        """ Tensorflow version of Pearson correlation between
        two vectors (should be 1 dimensional without singleton dims). """ 
        xm = x - tf.reduce_mean(x)
        ym = y - tf.reduce_mean(y)
        xnorm = tf.math.l2_normalize(xm)
        ynorm = tf.math.l2_normalize(ym)
        cosine_loss = tf.keras.losses.cosine_similarity(xnorm, ynorm)
        # Invert cosine "loss" to get proper cosine similarity (here: pearson corr)
        return -cosine_loss
    

class EuclideanRdm(Rdm):
    """ Euclidean distance RDM.

    Vectorized Euclidean distance matrix implementation adapted from Samuel Albanie:
    https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
    """

    def get_rdm(self, Z):
        eps = tf.keras.backend.epsilon()
        g_Z = tf.matmul(Z, tf.transpose(Z))
        diag_Z = tf.expand_dims(tf.linalg.diag_part(g_Z), 1)
        d_Z = diag_Z + tf.transpose(diag_Z) - 2 * g_Z
        return tf.sqrt(tf.maximum(d_Z, eps))  # avoid sqrt(<0)!


class OneMinCorrRdm(Rdm):
    
    def get_rdm(self, Z):
        eps = tf.keras.backend.epsilon()
        Zn = (Z - tf.reduce_mean(Z, axis=0)) / tf.math.reduce_std(Z, axis=0)
        Zn = tf.where(tf.math.is_nan(Zn), tf.zeros_like(Zn), Zn)
        return 1 - tf.matmul(Zn, Zn, transpose_b=True) / (Zn.shape[1] - 1)


class CKA(Rdm):
    def __init__(self, kernel='linear', **kwargs):
        """ Centered kernel alignment layer.
        Adapted from the Torch implementation from Jay Roxis:
        https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py

        Parameters
        ----------
        kernel : str
            Either 'linear' or 'rbf'
        add_loss_ : bool
            Whether to add the loss to the network this layer is used in
            (which will just return in input when calling the layer) or
            whether to just return the loss (when using it inside a custom
            loss function, like the `cka_loss` below)
        batch_size : int
            Ugly hack to get this layer to work as part of a network
            (instead of just a loss function), which needs know the batch
            size. Should be fixed to infer it upon runtime
        """
        super().__init__(**kwargs)
        self.kernel = kernel

    #def build(self, input_shape):
        # n = batch size
    #    n = tf.Variable(input_shape[0])
    #    unit = tf.ones([n, n])
    #    I = tf.eye(n.numpy())
    #    self.H = I - unit / n.numpy()

    def call(self, a_N, a_F):
        """ Computes CKA similarity [0, 1].
        
        Parameters
        ----------
        a_N : Tensor
            Tensorflow Tensor with dims N (observations) x
            K (neurons), representing the *N*eural activity
        a_F : Tensor
            Tensorflow Tensor with dims N (observations) x
            K (neurons / variables), representing the
            *F*eature activity       
        """
        l = getattr(self, f'_{self.kernel}_CKA')(a_N, a_F)
        if self.add_loss_:
            self.add_loss(l)
            return a_N  # just pass input
        else:
            return l
    
    def _centering(self, K):
        # Ugly hack to fix batch size issue
        if self.add_loss_:
            n = self.batch_size
        else:
            n = K.shape[0]
        
        unit = tf.ones([n, n])
        I = tf.eye(n)
        H = I - unit / n        
        return tf.matmul(tf.matmul(H, K), H)  

    def _rbf(self, Z, sigma=None):
        GZ = tf.matmul(Z, tf.transpose(Z))
        KZ = tf.diag(GZ) - GZ + tf.transpose(tf.diag(GZ) - GZ)
        if sigma is None:
            mdist = tf.median(KZ[KZ != 0])
            sigma = tf.sqrt(mdist)
        KZ *= - 0.5 / (sigma * sigma)
        KZ = tf.exp(KZ)
        return KZ

    def _kernel_HSIC(self, a_N, a_F, sigma):
        return tf.sum(self._centering(self._rbf(a_N, sigma)) * \
                      self._centering(self._rbf(a_F, sigma)))

    def _linear_HSIC(self, a_N, a_F):
        self.rdm_N = self._centering(tf.matmul(a_N, tf.transpose(a_N)))
        self.rdm_F = self._centering(tf.matmul(a_F, tf.transpose(a_F)))
        return tf.reduce_sum(self.rdm_N * self.rdm_F)

    def _linear_CKA(self, a_N, a_F):
        hsic = self._linear_HSIC(a_N, a_F)
        var1 = tf.sqrt(self._linear_HSIC(a_N, a_N))
        var2 = tf.sqrt(self._linear_HSIC(a_F, a_F))

        return hsic / (var1 * var2)

    def _kernel_CKA(self, a_N, a_F, sigma=None):
        hsic = self._kernel_HSIC(a_N, a_F, sigma)
        var1 = tf.sqrt(self._kernel_HSIC(a_N, a_N, sigma))
        var2 = tf.sqrt(self._kernel_HSIC(a_F, a_F, sigma))
        return hsic / (var1 * var2)
    
    def get_rdm(self, a_N):
        return self._centering(tf.matmul(a_N, tf.transpose(a_N)))


def TestModel(add_loss=False):
    from tensorflow.keras.layers import Input, Dense, Flatten
    from tensorflow.keras import Model

    s = Input(shape=(32, 32, 3))
    x = Flatten()(s)
    x = Dense(units=10)(x)
    
    if add_loss:
        a_F = Input(shape=(10,))
        x = CKA(add_loss_=True)(x, a_F)
        inputs = [s, a_F]
        outputs = Dense(1, 'sigmoid')(x)
    else:
        inputs = s
        outputs = x

    return Model(inputs=inputs, outputs=outputs)
