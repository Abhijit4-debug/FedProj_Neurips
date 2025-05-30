import numpy as np

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg7_bn', 'vgg7_bn_256', 'vgg9', 'vgg11', 'vgg11_bn', 'vgg12_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGLight(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGGLight, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGLight_256(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGGLight_256, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    'F': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'G': [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 512, 512, 512, 'M'],
    'H': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'I': [64, 'M', 128, 'M', 256, 'M', 256, 'M'],
}

def vgg7_bn_256(num_classes=10):
    """VGG 7-layer model (configuration "A")"""
    return VGGLight_256(make_layers(cfg['I'], batch_norm=True), num_classes)

def vgg7_bn(num_classes=10):
    """VGG 7-layer model (configuration "A")"""
    return VGGLight(make_layers(cfg['H'], batch_norm=True), num_classes)

def vgg9(num_classes=10):
    """VGG 9-layer model (configuration "A")"""
    return VGG(make_layers(cfg['F']), num_classes)

def vgg11(num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes)


def vgg11_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)

def vgg12_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['G'], batch_norm=True), num_classes)

def vgg13(num_classes=10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes)


def vgg13_bn(num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes)


def vgg16(num_classes=10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes)


def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes)


def vgg19(num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes)


def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)
