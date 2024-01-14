from .resnet import resnet12
from .resnet_v2 import ResNet18, ResNet50, ResNet101
from .resnext import ResNeXt50
from .convnet import convnet4


model_dict = {
    'resnet12': resnet12,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet18': ResNet18,
    'resnext50': ResNeXt50,
    'convnet4': convnet4,

}
