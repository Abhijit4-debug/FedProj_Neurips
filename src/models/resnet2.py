import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, Tuple, Union, List, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        output += residual
        output = self.relu(output)
        return output
    


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
#                  last_feature_dim=512, **kwargs):
        
#         #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
#         super(ResNet, self).__init__()
#         self.l2_norm = l2_norm
#         self.in_planes = 64
#         conv1_kernel_size = 3
#         if use_pretrained:
#             conv1_kernel_size = 7

#         Conv2d = self.get_conv()
#         Linear = self.get_linear()   
#         self.conv1 = Conv2d(3, 64, kernel_size=conv1_kernel_size,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer)
#         self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer)

#         self.logit_detach = False        

#         if use_pretrained:
#             resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
#             self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
#             self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
#             self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
#             self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)

#         self.num_layers = 6 # layer0 to layer5 (fc)

#         if l2_norm:
#             self.fc = Linear(last_feature_dim * block.expansion, num_classes, bias=False)
#         else:
#             self.fc = Linear(last_feature_dim * block.expansion, num_classes)


#     def get_conv(self):
#         return nn.Conv2d
    
#     def get_linear(self):
#         return nn.Linear

#     def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, Conv2d=self.get_conv()))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, return_feature=False):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)

#         if self.l2_norm:
#             self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
#             out = F.normalize(out, dim=1)
#             logit = self.fc(out)
#         else:
#             logit = self.fc(out)
            
#         if return_feature==True:
#             return out, logit
#         else:
#             return logit
        
    
#     def forward_classifier(self,x):
#         logit = self.fc(x)
#         return logit        
    
    
#     def sync_online_and_global(self):
#         state_dict=self.state_dict()
#         for key in state_dict:
#             if 'global' in key:
#                 x=(key.split("_global"))
#                 online=(x[0]+x[1])
#                 state_dict[key]=state_dict[online]
#         self.load_state_dict(state_dict)

    





# class ResNet_base(ResNet):
#     def __init__(self, block, layers, num_classes=1000, scaling=1.0, logit_detach: bool = False):
#         super().__init__(block, layers, num_classes=num_classes, scaling=scaling)
#         self.logit_detach = logit_detach

#     def forward_layer(self, layer, x, no_relu=True):
#         if isinstance(layer, nn.Linear):
#             x = F.adaptive_avg_pool2d(x, 1)
#             x = x.view(x.size(0), -1)
#             out = layer(x)
#         else:
#             if no_relu:
#                 out = x
#                 for sublayer in layer[:-1]:
#                     out = sublayer(out)
#                 out = layer[-1](out, no_relu=no_relu)
#             else:
#                 out = layer(x)
#         return out

#     def forward_layer_by_name(self, layer_name: str, x: torch.Tensor, no_relu=True) -> torch.Tensor:
#         layer = getattr(self, layer_name)
#         return self.forward_layer(layer, x, no_relu)

#     def forward_layer0(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
#         out = self.bn1(self.conv1(x))
#         if not no_relu:
#             out = F.relu(out)
#         return out

#     def freeze_backbone(self):
#         for name, param in self.named_parameters():
#             if 'fc' not in name:
#                 param.requires_grad = False
#         logger.warning('Freeze backbone parameters (except fc)')
#         return

#     def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
#         results = {}

#         out0 = self.bn1(self.conv1(x))
#         if not no_relu:
#             out0 = F.relu(out0)
#         results['layer0'] = out0
#         out = F.relu(out0)

#         if no_relu:
#             for idx, layer in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
#                 current_layer = getattr(self, layer)
#                 for i, sublayer in enumerate(current_layer):
#                     sub_norelu = (i == len(current_layer) - 1)
#                     out = sublayer(out, no_relu=sub_norelu)
#                 results[layer] = out
#                 out = F.relu(out)
#         else:
#             out = self.layer1(out)
#             results['layer1'] = out
#             out = self.layer2(out)
#             results['layer2'] = out
#             out = self.layer3(out)
#             results['layer3'] = out
#             out = self.layer4(out)
#             results['layer4'] = out

#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)
#         results['feature'] = out

#         if self.logit_detach:
#             logit = self.fc(out.detach())
#         else:
#             logit = self.fc(out)

#         results['logit'] = logit
#         results['layer5'] = logit

#         return results

# class ResNet8(nn.Module):
#     """Resnet model"""

#     def __init__(self, block, layers, scaling=1.0, num_classes=1000):
#         super(ResNet8, self).__init__()
#         assert int(64 * scaling) > 0
        
#         self.inplanes = int(64 * scaling)
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(block, int(64 * scaling), layers[0])
#         self.layer2 = self._make_layer(block, int(128 * scaling), layers[1], stride=2)
#         self.layer3 = self._make_layer(block, int(256 * scaling), layers[2], stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(int(256 * scaling) * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d): 
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#     def _make_layer(self, block, planes, layers, stride=1):
#         downsample = None
#         if stride !=1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers_list = []
#         layers_list.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, layers):
#             layers_list.append(block(self.inplanes, planes))
        
#         return nn.Sequential(*layers_list)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         out = self.fc(x)

#         return out





#####ORGINAL#####
class ResNet8(nn.Module):
    """General ResNet Model"""

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet8, self).__init__()
        assert int(64 * scaling) > 0

        self.inplanes = int(64 * scaling)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 * scaling), layers[0])
        self.layer2 = self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * scaling), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * scaling), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out


    
# class ResNet8(nn.Module):
#     """Resnet model

#     Args:
#         block (class): block type, BasicBlock or BottleneckBlock
#         layers (int list): layer num in each block
#         num_classes (int): class num
#     """

#     def __init__(self, block, layers, scaling=1.0, num_classes=1000):
#         super(ResNet8, self).__init__()
#         assert int(64 * scaling) > 0
        
#         self.inplanes = int(64 * scaling)
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
#         #kernel_size=7, stride=2, padding=3
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layers = nn.ModuleList()
#         self._make_layer(block, int(64 * scaling), layers[0])
#         self._make_layer(block, int(128 * scaling), layers[1], stride=2)
#         self._make_layer(block, int(256 * scaling), layers[2], stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(int(256 * scaling) * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d): 
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#     def _make_layer(self, block, planes, layers, stride=1):
#         """A block with 'layers' layers

#         Args:
#             block (class): block type
#             planes (int): output channels = planes * expansion
#             layers (int): layer num in the block
#             stride (int): the first layer stride in the block
#         """
#         downsample = None
#         if stride !=1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, layers):
#             self.layers.append(block(self.inplanes, planes))
        
#         return
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         for i in range(len(self.layers)):
#             x = self.layers[i](x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         out = self.fc(x)

#         return out
    
class ResNet8_feat(nn.Module):
    """Resnet model

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet8_feat, self).__init__()
        assert int(64 * scaling) > 0
        
        self.inplanes = int(64 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
        #kernel_size=7, stride=2, padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        self._make_layer(block, int(64 * scaling), layers[0])
        self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        self._make_layer(block, int(256 * scaling), layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(256 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            self.layers.append(block(self.inplanes, planes))
        
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        feat = self.avgpool(x)  # Features before the classification layer
        feat = torch.flatten(feat, 1)  # Flattened features
        out = self.fc(feat)  # Classification output

        return out, feat

if __name__ == "__main__":
    import torch
    net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)

    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
    
    total = 0
    for name, param in net.named_parameters():
        #print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')
    
