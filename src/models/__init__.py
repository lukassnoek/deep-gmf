from .cornet_z import CORnet_Z
from .resnet import ResNet10, ResNet6
from .pikenet import PikeNet

MODELS = {
    'CORnet-Z': CORnet_Z,
    'ResNet6': ResNet6,
    'ResNet10': ResNet10,
    'PikeNet': PikeNet
}