from .cornet_z import CORnet_Z
from .cornet_rt import CORnet_RT
from .resnet import ResNet10, ResNet6
from .pikenet import PikeNet

MODELS = {
    'CORnet-Z': CORnet_Z,
    'CORnet-RT': CORnet_RT,
    'ResNet6': ResNet6,
    'ResNet10': ResNet10,
    'PikeNet': PikeNet
}