from .layers import CKA, DistanceMatrix
from tensorflow.keras.losses import Loss


class CkaLoss(Loss):
    """ Centered Kernel Alignment loss.
    The better Z1 'correlates' with Z2, the higher
    the loss, which can be used to make representation Z1
    uncorrelated with representation Z2. If inverse is True,
    it can be used to make Z1 and Z2 similar!
    """

    def __init__(self, inverse=False):
        super().__init__()
        self.inverse = inverse

    def call(self, Z1, Z2):
        l = CKA()(Z1, Z2)
        if self.inverse:
            l = 1 - l

        return l


class RdmLoss(Loss):
    """ Representational dissimilarity matrix loss.
    
    RSA-like loss based on the correspondence between
    two observations x features matrices (Z1 and Z2),
    where the most likely usecase is Z1 being a 
    (flattened) DNN layer and Z2 some stimulus feature
    representation.

    The better Z1 'correlates' with Z2, the higher
    the loss, which can be used to make representation Z1
    uncorrelated with representation Z2. If inverse is True,
    it can be used to make Z1 and Z2 similar!
    """

    def __init__(self, inverse=False):
        super().__init__()
        self.inverse = inverse    
    
    def loss(self, Z1, Z2):
        l = DistanceMatrix(Z1, Z2)
        if self.inverse:
            l = 1 - l

        return l