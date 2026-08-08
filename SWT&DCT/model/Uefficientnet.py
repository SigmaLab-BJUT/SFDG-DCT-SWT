import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class upblock(nn.Module):
    def __init__(self,
                 input_c: int,
                 out_c: int):
        super(upblock, self).__init__()

        uplayers = OrderedDict()
        uplayers.update({"transpose": nn.ConvTranspose2d(input_c, input_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))})
        uplayers.update({"conv1": nn.Conv2d(input_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))})
        uplayers.update({"bn1": nn.BatchNorm2d(out_c)})
        uplayers.update({"silu1": nn.SiLU()})
        uplayers.update({"conv2": nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))})
        uplayers.update({"bn2": nn.BatchNorm2d(out_c)})
        uplayers.update({"silu2": nn.SiLU()})


        self.block = nn.Sequential(uplayers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)

        return result


class UEfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(UEfficientNet, self).__init__()


        # Encoder : efficientnet
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        list_inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            inverted_residual_setting = []
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = 'down-'+str(stage + 2) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1
            list_inverted_residual_setting.append(inverted_residual_setting)



        self.down_s1 =  ConvBNActivation(in_planes=3,
                                         out_planes=adjust_channels(32),
                                         kernel_size=3,
                                         stride=2,
                                         norm_layer=norm_layer)
        down_s1_output_c = adjust_channels(32)

        layers_s2 = OrderedDict()
        for cnf in list_inverted_residual_setting[0]:
            layers_s2.update({cnf.index: block(cnf, norm_layer)})
        self.down_s2 = nn.Sequential(layers_s2)
        down_s2_output_c = list_inverted_residual_setting[0][-1].out_c

        layers_s3 = OrderedDict()
        for cnf in list_inverted_residual_setting[1]:
            layers_s3.update({cnf.index: block(cnf, norm_layer)})
        self.down_s3 = nn.Sequential(layers_s3)
        down_s3_output_c = list_inverted_residual_setting[1][-1].out_c

        layers_s4 = OrderedDict()
        for cnf in list_inverted_residual_setting[2]:
            layers_s4.update({cnf.index: block(cnf, norm_layer)})
        self.down_s4 = nn.Sequential(layers_s4)
        down_s4_output_c = list_inverted_residual_setting[2][-1].out_c

        self.down_s4_5 = ConvBNActivation(in_planes=down_s4_output_c*2,
                                         out_planes=down_s4_output_c,
                                         kernel_size=3,
                                         stride=1,
                                         norm_layer=norm_layer)

        layers_s5 = OrderedDict()
        for cnf in list_inverted_residual_setting[3]:
            layers_s5.update({cnf.index: block(cnf, norm_layer)})
        self.down_s5 = nn.Sequential(layers_s5)
        down_s5_output_c = list_inverted_residual_setting[3][-1].out_c

        layers_s6 = OrderedDict()
        for cnf in list_inverted_residual_setting[4]:
            layers_s6.update({cnf.index: block(cnf, norm_layer)})
        self.down_s6 = nn.Sequential(layers_s6)
        down_s6_output_c = list_inverted_residual_setting[4][-1].out_c

        layers_s7 = OrderedDict()
        for cnf in list_inverted_residual_setting[5]:
            layers_s7.update({cnf.index: block(cnf, norm_layer)})
        self.down_s7 = nn.Sequential(layers_s7)
        down_s7_output_c = list_inverted_residual_setting[5][-1].out_c

        layers_s8 = OrderedDict()
        for cnf in list_inverted_residual_setting[6]:
            layers_s8.update({cnf.index: block(cnf, norm_layer)})
        self.down_s8 = nn.Sequential(layers_s8)

        down_s8_output_c = list_inverted_residual_setting[6][-1].out_c

        down_s9_output_c = adjust_channels(1280)
        self.down_s9 = ConvBNActivation(in_planes=down_s8_output_c,
                                               out_planes=down_s9_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)
        #dct输入的维度为1
        self.dct_s1 = ConvBNActivation(in_planes=1,
                                        out_planes=adjust_channels(32),
                                        kernel_size=3,
                                        stride=2,
                                        norm_layer=norm_layer)


        # unet的维度拉伸
        # decoder
        up_out_ch = [512, 512, 256, 128, 64, 32]
        self.down_conv0 = nn.Conv2d(down_s9_output_c, up_out_ch[0], (1, 1), stride=1, padding=0)
        self.down_bnc0 = nn.BatchNorm2d(up_out_ch[0])
        self.down_silu = nn.SiLU()

        self.up_s1 = upblock(up_out_ch[0]+down_s8_output_c, up_out_ch[1])
        self.up_s2 = upblock(up_out_ch[1]+down_s6_output_c, up_out_ch[2])
        self.up_s3 = upblock(up_out_ch[2]+down_s4_output_c, up_out_ch[3])
        self.up_s4 = upblock(up_out_ch[3]+down_s3_output_c, up_out_ch[4])
        self.up_s5 = upblock(up_out_ch[4]+down_s2_output_c, up_out_ch[5])
        # self.up_s1 = upblock(up_out_ch[0]+down_s8_output_c*2, up_out_ch[1])
        # self.up_s2 = upblock(up_out_ch[1]+down_s6_output_c*2, up_out_ch[2])
        # self.up_s3 = upblock(up_out_ch[2]+down_s4_output_c*2, up_out_ch[3])
        # self.up_s4 = upblock(up_out_ch[3]+down_s3_output_c*2, up_out_ch[4])
        # self.up_s5 = upblock(up_out_ch[4]+down_s2_output_c*2, up_out_ch[5])
        self.up_outconv = nn.Conv2d(up_out_ch[5], 2, 1)

        self.dropout_rate = dropout_rate
        #分类处理
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(down_s9_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def x_concat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        cat = torch.cat([x1, x2], dim=1)
        return cat
    # def x_concat(self, x1, x2, x3):
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]
    #
    #     x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
    #                     diffY // 2, diffY - diffY // 2))
    #
    #     cat = torch.cat([x1, x2], dim=1)
    #     cat = torch.cat([cat, x3], dim=1)
    #     return cat

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.down_s1(x)
        x2 = self.down_s2(x1)
        x3 = self.down_s3(x2)
        x4 = self.down_s4(x3)
        x5 = self.down_s5(x4)
        x6 = self.down_s6(x5)
        x7 = self.down_s7(x6)
        x8 = self.down_s8(x7)
        x9 = self.down_s9(x8)


        maskx = self.down_silu(self.down_bnc0(self.down_conv0(x9)))
        maskx = self.x_concat(maskx, x8.detach())
        maskx = self.up_s1(maskx)
        maskx = self.x_concat(maskx, x6.detach())
        maskx = self.up_s2(maskx)
        maskx = self.x_concat(maskx, x4.detach())
        maskx = self.up_s3(maskx)
        maskx = self.x_concat(maskx, x3.detach())
        maskx = self.up_s4(maskx)
        maskx = self.x_concat(maskx, x2.detach())
        maskx = self.up_s5(maskx)
        maskout = self.up_outconv(maskx)


        class_x = self.avgpool(x9)
        class_x = torch.flatten(class_x, 1)
        class_x = self.classifier(class_x)
        return class_x, maskout


def Uefficientnet_b0(num_classes=2):
    # input image size 224x224
    return UEfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def Uefficientnet_b1(num_classes=2):
    # input image size 240x240
    return UEfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def Uefficientnet_b2(num_classes=2):
    # input image size 260x260
    return UEfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def Uefficientnet_b3(num_classes=2):
    # input image size 300x300
    return UEfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def Uefficientnet_b4(num_classes=2):
    # input image size 380x380
    return UEfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def Uefficientnet_b5(num_classes=2):
    # input image size 456x456
    return UEfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def Uefficientnet_b6(num_classes=2):
    # input image size 528x528
    return UEfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def Uefficientnet_b7(num_classes=2):
    # input image size 600x600
    return UEfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)



if __name__ == '__main__':
    model = Uefficientnet_b4(num_classes=2)
    # print(model.down_s2[1].block[1].fc1.out_channels)  #down-2b->block->se->fc1
    # print(model.down_s2[-1].block[-1][0].out_channels)  #down-2b->block->project_conv->0

    # 打印GPU信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 用单个/多个GPU。直接代替model = model.cuda()
    imgs = torch.rand(6, 3, 320, 320)

    label = np.ones(6)
    label = torch.tensor(label).long()
    print(label.shape)
    dctimgs = torch.rand(6, 1, 320, 320)
    classout, maskout = model(imgs.to(device), dctimgs.to(device))
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    loss = criterion1(classout.to(device), label.to(device))
    loss1 = criterion2(maskout.to(device), dctimgs.to(device).squeeze(1).long())
    loss2=loss+loss1
    loss2.backward()
    print(classout)
    print(maskout.shape)
