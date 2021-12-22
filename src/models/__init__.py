from .cornet_z import CORnet_Z
#from .cornet_rt import CORnet_RT
from .resnet import ResNet10, ResNet6
from .pikenet import PikeNet
from .vnet import VNet, VNet_skip
from .vgg16 import VGG16


MODELS = {
    'CORnet-Z': CORnet_Z,
    'VNet': VNet,
    'VNet-skip': VNet_skip,
    'ResNet6': ResNet6,
    'ResNet10': ResNet10,
    'PikeNet': PikeNet,
    'VGG16': VGG16
}