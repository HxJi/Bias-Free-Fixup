import torch
import torch.nn as nn
import numpy as np
import scipy.io as io

__all__ = ['FixupResNet', 'fixup_resnet18', 'fixup_resnet34', 'fixup_resnet50', 'fixup_resnet101', 'fixup_resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        #out = self.conv1(x + self.bias1a)
        out = self.conv1(x)
        out = self.relu(out + self.bias1b)

        #record activation after relu 
        print('Out 1 type,size:', type(out))
        np.savetxt(out.numpy())

        #out = self.conv2(out + self.bias2a)
        #out = self.conv2(out)
        #out = out * self.scale + self.bias2b

        if self.downsample is not None:
            #identity = self.downsample(x)
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        #record activation after relu 2
        #print (type(out))

        return out


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,  numlayer, epoch, stride=1, downsample=None):
        super(FixupBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.numlayer = numlayer
        self.epoch = epoch

    def forward(self, x):
        identity = x
        # relu->bias#a->conv remove them

        out = self.conv1(x)
        out = self.relu(out + self.bias1b)
        #sparsity_file = open('sparsity-{0}.csv'.format(self.epoch), 'a')
        activation_file_1 = open('activation-{0}-{1}-1'.format(self.epoch, self.numlayer), 'ab')
        print(1-torch.nonzero(out).size(0)/torch.numel(out))
        #sparsity_file.write('{}'.format(asp))

        array = out.cpu().detach().numpy()
        array[array!=0] = 1
        array2 = array.astype(np.uint8)
        array2 = array2[0:16]    
        activation_file_1.write(array2)
        activation_file_1.close()

        out = self.conv2(out)
        out = self.relu(out + self.bias2b)

        print(1-torch.nonzero(out).size(0)/torch.numel(out))
        #sparsity_file.write('{}'.format(asp))

        activation_file_2 = open('activation-{0}-{1}-2'.format(self.epoch, self.numlayer), 'ab')
        array3 = out.cpu().detach().numpy()
        array3[array3!=0] = 1
        array4 = array3.astype(np.uint8)
        array4 = array4[0:16]         
        activation_file_2.write(array4)
        activation_file_2.close()

        out = self.conv3(out)
        out = out * self.scale + self.bias3b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        print(1-torch.nonzero(out).size(0)/torch.numel(out))
        #sparsity_file.write('{}'.format(asp))

        activation_file_3 = open('activation-{0}-{1}-3'.format(self.epoch, self.numlayer), 'ab')
        array5 = out.cpu().detach().numpy()
        array5[array5!=0] = 1
        array6 = array5.astype(np.uint8)
        array6 = array6[0:16]      
        activation_file_3.write(array6)
        activation_file_3.close()
        #sparsity_file.close()

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, epoch, num_classes=1000):
        super(FixupResNet, self).__init__()
        
        self.numlayer = 1
        self.epoch = epoch
        print('key step', epoch)
        print('self key', self.epoch)
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], self.numlayer, self.epoch)
        self.numlayer = self.numlayer + layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], self.numlayer, self.epoch, stride=2)
        self.numlayer = self.numlayer + layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], self.numlayer, self.epoch, stride=2 )
        self.numlayer = self.numlayer + layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], self.numlayer, self.epoch, stride=2 )
        self.numlayer = self.numlayer + layers[3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2 / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, numlayer, epoch, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, numlayer, epoch, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, numlayer+i, epoch))

        return nn.Sequential(*layers)

    def forward(self, x, epoch):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        print(1-torch.nonzero(x).size(0)/torch.numel(x))
        activation_file_pool = open('activation-{0}-{1}-relu'.format(epoch,0), 'ab')
        array = x.cpu().detach().numpy()
        array[array!=0] = 1
        brray = array.astype(np.uint8)
        brray = brray[0:16] 
        activation_file_pool.write(brray)
        activation_file_pool.close()

        x = self.maxpool(x)
        print(1-torch.nonzero(x).size(0)/torch.numel(x))
        activation_file_pool = open('activation-{0}-{1}-maxpool'.format(epoch,0), 'ab')
        array = x.cpu().detach().numpy()
        array[array!=0] = 1
        brray = array.astype(np.uint8)
        brray = brray[0:16] 
        activation_file_pool.write(brray)
        activation_file_pool.close()
        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        print(1-torch.nonzero(x).size(0)/torch.numel(x))
        activation_file_pool = open('activation-{0}-{1}-avgpool'.format(epoch,0), 'ab')
        array = x.cpu().detach().numpy()
        array[array!=0] = 1
        brray = array.astype(np.uint8)
        brray = brray[0:16] 
        activation_file_pool.write(brray)
        activation_file_pool.close()

        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x

def fixup_resnet18(**kwargs):
    """Constructs a Fixup-ResNet-18 model.

    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def fixup_resnet34(**kwargs):
    """Constructs a Fixup-ResNet-34 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet50(**kwargs):
    """Constructs a Fixup-ResNet-50 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet101(**kwargs):
    """Constructs a Fixup-ResNet-101 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def fixup_resnet152(**kwargs):
    """Constructs a Fixup-ResNet-152 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 8, 36, 3], **kwargs)
    return model
