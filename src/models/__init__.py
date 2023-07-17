from .cornet_z import CORnet_Z
#from .cornet_rt import CORnet_RT
from .resnet import ResNet6, ResNet10, ResNet18, ResNet34
from .vgg16 import VGG16
from .stereonet import StereoResNet10, StereoResNet6
from .arcface import ArcFace


MODELS = {
    'CORnet-Z': CORnet_Z,
    'ResNet6': ResNet6,
    'ResNet10': ResNet10,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'VGG16': VGG16,
    'StereoResNet10': StereoResNet10,
    'StereoResNet6': StereoResNet6,
    'ArcFace': ArcFace
}