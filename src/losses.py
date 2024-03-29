from math import pi
import tensorflow as tf
from .layers import CKA, EuclideanRdm
from tensorflow.keras.losses import Loss


class AngleLoss(Loss):
    """ Mean squared (or absolute) error between two angles. Can be used as a loss
    function for regression problems where the target values are angles (like orientation
    of a face).
    
    Parameters
    ----------
    absolute : bool
        Whether to use MSE (if False) or MAE (if True)
    is_degree : bool
        Whether the labels are encoded as degrees (if True)
        or radians (if False). In case of the former, the
        labels (and predictions) are converted to radians first
    name : str
        Name of loss function
    kwargs : kwargs
        Keyword arguments to be passed to parent Loss object initialization
    """
    def __init__(self, absolute=False, is_degree=True, name='angle_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.absolute = absolute
        self.is_degree = is_degree
        
    def call(self, y_true, y_pred):
        if self.is_degree:  # deg -> rad
            y_true = tf.multiply(y_true, (pi / 180))
            y_pred = tf.multiply(y_pred, (pi / 180))
    
        # https://stats.stackexchange.com/questions/425234/
        # loss-function-and-encoding-for-angles
        l = 2 * (1 - tf.math.cos(y_true - y_pred))
        if self.absolute:
            # MAE instead of MSE
            l = tf.sqrt(l)
            
        l = tf.reduce_mean(l)
        return l


class CkaLoss(Loss):
    """ Centered Kernel Alignment loss. The better Z1 'correlates' with Z2, the higher
    the loss, which can be used to make representation Z1 uncorrelated with representation
    Z2. If inverse is True, it can be used to make Z1 and Z2 similar!

    Stopped using RDM-style losses/analyses as they tend to not be as sensitive as
    supervised decoding (layers -> feature)

    Parameters
    ----------
    inverse : bool
        Whether to return the actual CKA value (if False) or 1 - CKA value (if True);
        the latter is useful if you want the network to learn to "filter" out the
        information in Z1 that is present in Z2
    """

    def __init__(self, inverse=False, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse

    def call(self, Z1, Z2):
        l = CKA()(Z1, Z2)
        if self.inverse:
            l = 1 - l

        return l


class EuclideanRdmLoss(Loss):
    """ Representational dissimilarity matrix loss.
    
    RSA-like loss based on the correspondence between two observations x features
    matrices (Z1 and Z2), where the most likely usecase is Z1 being a (flattened) DNN
    layer and Z2 some stimulus feature representation.

    Same caveats as CkaLoss apply.
    """

    def __init__(self, inverse=False, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse    
    
    def loss(self, Z1, Z2):
        l = EuclideanRdm(Z1, Z2)
        if self.inverse:
            l = 1 - l

        return l
