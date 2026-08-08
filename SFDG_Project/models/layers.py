from turtle import forward
from warnings import simplefilter
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.xception import SeparableConv2d


def conv3x3(in_planes, out_planes, stride, padding):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def MLP(dim_in=256 * 3):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, 1024)
            )

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def AdaIN(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return x * self.sigmoid(avgout + maxout) + x


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base
        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, C, H, W]
        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, C, H, W]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, C, H, W]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 4*C, H, W]
        return out


class Adaptive_Head(nn.Module):
    def __init__(self, size=380, window_size=10, M=4):
        super(Adaptive_Head, self).__init__()

        self.window_size = window_size
        self._M = M
        self.FAD_Head = FAD_Head(size)
        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        # self._DCT_patch_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        # self._DCT_patch_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=window_size, padding=0)
        # self.conv = conv3x3(3, 3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(3)
        self.resblock = ResBlock(3,3)
        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
        self.transform = torchvision.transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

    def forward(self, x):
        # turn RGB into Gray
        # x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        # x = x_gray.unsqueeze(1)
        N, C, W, H = x.size()
        M = torch.sigmoid(self.resblock(x))
        Global_feature = self.FAD_Head(x)
        # rescale to 0 - 255
        # x_YCrCb = torch.zeros_like(x)
        # x = (x + 1.) * 122.5
        # x_YCrCb[:,0,:,:] = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        # x_YCrCb[:,1,:,:] = 0.511*x[:,0,:,:] - 0.428*x[:,1,:,:] + 0.083*x[:,2,:,:] + 128
        # x_YCrCb[:,2,:,:] = -0.172*x[:,0,:,:] - 0.339*x[:,1,:,:] + 0.511*x[:,2,:,:] + 128
        # x_YCrCb = self.transform(x_YCrCb)
        # calculate size
        S = self.window_size
        size_after = int((W - S)/S) + 1
        assert size_after == 38

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        x_dct = torch.abs(x_dct)
        x_dct = torch.log10(x_dct + 1e-15)
        x_dct = x_dct.reshape_as(x)

        # Local_feature_R = x_dct[:,0,:,:].unsqueeze(1).repeat(1,4,1,1)   
        Local_feature = x_dct.repeat(1, 4, 1, 1)
        Local_feature = torch.cat([Local_feature[:,0::3,:,:], Local_feature[:,1::3,:,:],Local_feature[:,2::3,:,:]],1)
        M = M.repeat(1, 4, 1, 1)
        M = torch.cat([M[:,0::3,:,:], M[:,1::3,:,:],M[:,2::3,:,:]],1)
        out = Global_feature * M + Local_feature * (1 - M)

        return out


class SepBlock(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(SepBlock, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

    
class AttentionRefinement(nn.Module):
    def __init__(self, in_chan=4, out_chan=4, *args, **kwargs):
        super(AttentionRefinement, self).__init__()
        self.conv = conv3x3(in_chan, out_chan, 1, 1)
        self.bn = nn.BatchNorm2d(out_chan)
        # self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = F.relu(self.bn(self.conv(x)))
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

                
class Multi_Scale_Context_Extractor(nn.Module):
    def __init__(self, in_channels):
        super(Multi_Scale_Context_Extractor, self).__init__()
        self.arm = AttentionRefinement(in_channels, 4)
        self.arm2 = AttentionRefinement(in_channels, 4)
        self.conv_head = ConvBNReLU(4, 4, ks=3, stride=1, padding=1)
        self.conv_head2 = ConvBNReLU(4, 4, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(in_channels, 4, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        # feat8, feat16, feat32 = self.resnet(x)
        fea2 = F.interpolate(x, (H0//2, W0//2), mode='nearest')
        # H8, W8 = feat8.size()[2:]
        # H16, W16 = feat16.size()[2:]
        # H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(fea2, fea2.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H0//2, W0//2), mode='nearest')

        fea2_arm = self.arm2(fea2)
        fea2_sum = fea2_arm + avg_up
        fea2_up = F.interpolate(fea2_sum, (H0, W0), mode='nearest')
        fea2_up = self.conv_head2(fea2_up)

        fea_arm = self.arm(x)
        fea_sum = fea_arm + fea2_up
        out = self.conv_head(fea_sum)

        return out, fea_sum#feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.bn2 = nn.BatchNorm2d(out_channels)


        # self.channel_attention = ChannelAttention(out_channels * 2)
        # self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=kernel_size, padding=padding)

        self.theta = theta

        # self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        # self.conv1_weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))

    def forward(self, x):
        # # shape = self.conv_weight.shape
        # cdc_weight_c = self.conv_weight.sum(dim=[2, 3], keepdim=True)
        # # conv_weight_reshape = self.conv_weight.view(shape[0], shape[1], -1)
        # # conv_weight_reshape[:, :, 4] = conv_weight_reshape[:, :, 4] - cdc_weight_c
        # # conv_out = conv_weight_reshape.view(shape)
        # # cdc_fea = F.tanh(F.conv2d(x, conv_out, stride=1, padding=1, dilation=1, groups=1))
        #
        # yc = F.conv2d(x, cdc_weight_c, stride=1, padding=0, groups=1)
        # y = F.conv2d(x, self.conv_weight, stride=1, padding=1, dilation=1, groups=1)
        #
        # cdc_fea = F.tanh(y - yc)
        #
        # shape = self.conv1_weight.shape
        # conv1_weight_reshape = nn.Parameter(self.conv1_weight.view(shape[0], shape[1], -1))
        # conv1_out = (conv1_weight_reshape - conv1_weight_reshape[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        # adc_fea = F.tanh(F.conv2d(x, conv1_out, stride=1, padding=1, dilation=1, groups=1))
        #
        # out = self.channel_attention(torch.cat([cdc_fea, adc_fea], dim=1))
        # return F.relu(self.conv2(out))

        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            out = out_normal - self.theta * out_diff
            return F.tanh(self.bn1(out))


def convert_pdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    elif op == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv


if __name__ == '__main__':
    image = torch.randn(4,4,23,23)
    model = Conv2d_cd(4, 64)
    print(model(image).size())