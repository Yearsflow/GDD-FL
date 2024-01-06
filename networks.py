import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
class ResNetProto(ResNet):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNetProto, self).__init__(block, num_blocks, channel, num_classes, norm)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        x1 = self.layer4(out)
        out = F.avg_pool2d(x1, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, x1

def ResNet18(args, channel, num_classes, norm='instancenorm'):
    if args.approach == 'fedproto':
        return ResNetProto(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm=norm)
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm=norm)

def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat
    
class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        if args.dataset == 'mnist':
            self.in_channels = 1
            self.conv1 = nn.Conv2d(1, 196, kernel_size=3, stride=1, padding=1)
            size = 28
        else:
            self.in_channels = 3
            self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
            if args.dataset == 'cifar10':
                size = 32
            else:
                size = 224
        self.ln1 = nn.LayerNorm(normalized_shape=[196, size, size])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(normalized_shape=[196, size // 2, size // 2])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm(normalized_shape=[196, size // 4, size // 4])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm(normalized_shape=[196, 4, 4])
        self.lrelu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.lrelu2(x)

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.lrelu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.ln4(x)
        x = self.lrelu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.ln5(x)
        x = self.lrelu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.ln6(x)
        x = self.lrelu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.ln7(x)
        x = self.lrelu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        x = self.ln8(x)
        x = self.lrelu8(x)

        if print_size:
            print(x.size())

        x = self.pool(x)

        if print_size:
            print(x.size())

        x = x.view(x.size(0), -1)

        if print_size:
            print(x.size())

        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)

        if print_size:
            print("fc1_out size: {}".format(fc1_out.size()))
            print("fc10_out size: {}".format(fc10_out.size()))

        return fc1_out, fc10_out


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.bn0 = nn.BatchNorm1d(196*4*4)
        self.relu0 = nn.ReLU()

        if args.dataset == 'mnist':
            self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(196)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(196)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(196)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(196)
        self.relu4 = nn.ReLU()

        #if args.data == 'mnist':
        #    self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=2, padding=1)
        #else:
        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(196)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(196)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(196)
        self.relu7 = nn.ReLU()

        if args.dataset == 'mnist':
            self.conv8 = nn.Conv2d(196, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.conv8 = nn.Conv2d(196, 3, kernel_size=3, stride=1, padding=1)
        # bn and relu are not applied after conv8

        self.tanh = nn.Tanh()

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.fc1(x)
        x = self.bn0(x)
        x = self.relu0(x)

        if print_size:
            print(x.size())

        x = x.view(-1, 196, 4, 4)

        if print_size:
            print(x.size())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        # bn and relu are not applied after conv8

        if print_size:
            print(x.size())

        x = self.tanh(x)

        if print_size:
            print("output (tanh) size: {}".format(x.size()))

        return x