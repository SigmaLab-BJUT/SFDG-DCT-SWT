import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.xception import xception
from models.efficientnet import EfficientNet
import kornia
import os
import torchvision.models as torchm
from utils import cont_grad
from models.layers import ResBlock, conv1x1, FAD_Head, Adaptive_Head, conv3x3, SepBlock
from models.layers import Conv2d_cd, ChannelAttention, Multi_Scale_Context_Extractor
from models.Graph_layers import get_graph_feature, KNN
# from models.xception import SeparableConv2d
# from models.HRNet_layers import HRAttentionNet
# from models.loss import *
# from models.QCO_layer import TEM
# from models.loss import Poincare_loss#, SC_loss


class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):    # in_channels = 160, out_channels = 4
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels

        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1) #extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.Multi_Scale_Context_Extractor = Multi_Scale_Context_Extractor(in_channels)

        # self.conv_extract_1 = ResBlock(in_channels, in_channels)
        # self.conv_extract_2 = ResBlock(in_channels, in_channels)
        # self.conv3 = nn.Sequential(
        #    conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        fea2_sum = 0

        x0 = self.conv_extract(x)
        x0 = self.bn1(x0)
        x0 = F.relu(x0,inplace=True)
        x0 = self.conv2(x0)
        x0 = self.bn2(x0)

        x_MSC, fea2_sum = self.Multi_Scale_Context_Extractor(x)

        # x0 = self.conv_extract_1(x)
        # x0 = self.conv_extract_2(x0)
        # x0 = self.conv3(x0)

        x = x0 + x_MSC
        x = F.elu(x)+1

        mask = F.interpolate(
            self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        return x*mask, fea2_sum


class DynamicGraph(nn.Module):
    def __init__(self, emb_dims, k):
        super(DynamicGraph, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(2*emb_dims, 64, kernel_size=1, bias=False),    # 8
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)
        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv6(x)

        return x


class Frequency_Extract(nn.Module):
    def __init__(self, size, in_channels, out_channels):
        super(Frequency_Extract, self).__init__()
        self.output_features = out_channels
        self.mid_features = out_channels
        #self.FAD_Head = FAD_Head(size[0])
        self.Adaptive_Head = Adaptive_Head(size=size[0])

        # [12, 380, 380] ==> [32, 190, 190]
        self.conv_1 = conv3x3(
            in_planes=in_channels, out_planes=64, stride=2, padding=1)  # [32, 190, 190]
        self.bn1 = nn.BatchNorm2d(64)
        self.conv_2 = conv3x3(in_planes=64, out_planes=128,
                              stride=1, padding=1)  # [32, 190, 190]
        self.bn2 = nn.BatchNorm2d(128)

        # [32, 190, 190] ==> [64, 95, 95]
        self.Sepblock_1 = SepBlock(
            128, 256, 2, 2, start_with_relu=False, grow_first=True)   # [64, 95, 95]

#         # [64, 95, 95] ==> [128, 47, 47]
#         self.Sepblock_2 = SepBlock(256, 256, 2, 2, start_with_relu=True, grow_first=True)   # [128, 47, 47]

#         # [128, 47, 47] ==> [256, 23, 23]
#         self.Sepblock_3 = SepBlock(256, 256, 2, 2, start_with_relu=True, grow_first=True)   # [256, 23, 23]

        self.conv_3 = conv1x1(in_planes=256, out_planes=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #x_d = self.FAD_Head(x)
        x_d = self.Adaptive_Head(x)

        x = self.conv_1(x_d)
        x = F.relu(self.bn1(x))

        x = self.conv_2(x)
        x_0 = F.relu(self.bn2(x))

        x_1 = self.Sepblock_1(x_0)
#         x = self.Sepblock_2(x_1)
#         x = self.Sepblock_3(x)

#         x = self.conv_3(x)
#         x_2 = F.relu(self.bn3(x))

        x = self.conv_3(x_1)
        x_2 = F.relu(self.bn3(x))
        return x_2


class Auxiliary_Loss_v2(nn.Module):
    def __init__(self, M, N, C, alpha=0.05, margin=1, inner_margin=[0.1, 5]):
        super().__init__()
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(
                attentions, (H, W), mode='bilinear', align_corners=True)
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix-feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = self.alpha * \
            torch.mean(center_momentum*real_mask, dim=0)+feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        fctsd, torch.distributed.ReduceOp.SUM)
                    fctsd /= torch.distributed.get_world_size()
                    # fctsd /= card_num
                self.feature_centers = fctsd
        inner_margin = self.inner_margin[y]
        intra_class_loss = F.relu(torch.norm(
            feature_matrix-fcts, dim=[1, 2])*torch.sign(inner_margin)-inner_margin)
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0

        # KNN weight
        # Ones = torch.ones(M, M).cuda()
        # Index = KNN(self.feature_centers, 2)
        # F_matrix = torch.zeros(M, M).cuda()
        # F_matrix.scatter_(1,Index,Ones)

        for j in range(M):
            for k in range(j+1, M):
                # * F_matrix[j,k]
                inter_class_loss += F.relu(self.margin -
                                           torch.dist(fcts[j], fcts[k]), inplace=False) #* F_matrix[j,k]

        # attentions_ = attentions.view(attentions.shape[0], attentions.shape[1], -1)
        # center_norm = torch.norm(attentions_, 2, dim=1, keepdim=True)
        # center_norm = torch.div(attentions_, center_norm+1e-6)
        # ED_loss = 0.0
        # for b in range(B):
        #     # print(torch.det(torch.mm(center_norm[b,:,:].T, center_norm[b,:,:])))
        #     ED_loss += torch.log(torch.det(torch.mm(center_norm[b,:,:].T, center_norm[b,:,:]))+1e-6)
        # ED_loss /= B

        # center_norm = torch.norm(fcts, 2, dim=0, keepdim=True)
        # center_norm = torch.div(fcts, center_norm+1e-6)
        # ED_loss = torch.log(torch.det(torch.mm(center_norm.T, center_norm))+1e-6)

        inter_class_loss = inter_class_loss/M/self.alpha  #- 0.1 * ED_loss

        # fmd=attentions.flatten(2)
        # diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss+inter_class_loss, feature_matrix


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(
                H, W), mode='bilinear', align_corners=True)
        if norm == 1:
            attentions = attentions+1e-8
        if len(features.shape) == 4:
            feature_matrix = torch.einsum(
                'imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum(
                'imjk,imnjk->imn', attentions, features)
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)+1e-8
            feature_matrix /= w
        return feature_matrix


class Texture_Enhance_v2(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features
        self.output_features_d = num_features
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)
        # self.cdc_extract = Conv2d_cd(num_features, num_features)

        self.conv0 = nn.Conv2d(num_features*num_attentions, num_features *
                               num_attentions, 5, padding=2, groups=num_attentions)
        self.conv1 = nn.Conv2d(num_features*num_attentions, num_features *
                               num_attentions, 3, padding=1, groups=num_attentions)
        self.bn1 = nn.BatchNorm2d(num_features*num_attentions)
        self.conv2 = nn.Conv2d(num_features*2*num_attentions, num_features *
                               num_attentions, 3, padding=1, groups=num_attentions)
        self.bn2 = nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3 = nn.Conv2d(num_features*3*num_attentions, num_features *
                               num_attentions, 3, padding=1, groups=num_attentions)
        self.bn3 = nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last = nn.Conv2d(
            num_features*4*num_attentions, num_features*num_attentions, 1, groups=num_attentions)
        self.bn4 = nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features*num_attentions)

        self.M = num_attentions

    def cat(self, a, b):
        B, C, H, W = a.shape
        c = torch.cat([a.reshape(B, self.M, -1, H, W), b.reshape(B,
                      self.M, -1, H, W)], dim=2).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (
                int(H*attention_maps[0]), int(W*attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        # output size [attention_size, attention_size]
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - \
                F.interpolate(
                    feature_maps_d, (feature_maps.shape[2], feature_maps.shape[3]), mode='nearest')
            # LBP = LBP.repeat(1, feature_maps.size(1), 1, 1)
            # feature_maps = AdaIN(feature_maps, LBP)

            # feature_maps=feature_maps+F.interpolate(LBP,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')

        attention_maps = (torch.tanh(F.interpolate(attention_maps.detach(), (H, W), mode='bilinear', align_corners=True))).unsqueeze(2) if type(attention_maps) != tuple else 1
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps*attention_maps).reshape(B, -1, H, W)
        # feature_maps=self.TEM(feature_maps)
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(
            F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(
            F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(
            F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(self.bn_last(self.conv_last(
            F.relu(self.bn4(feature_maps3_), inplace=True))), inplace=True)
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


class MAT(nn.Module):
    def __init__(self, net='xception', feature_layer='b3', attention_layer='final', num_classes=2, M=8, mid_dims=256,
                 dropout_rate=0.5, drop_final_rate=0.5, pretrained=False, alpha=0.05, size=(380, 380), margin=1, inner_margin=[0.01, 0.02]):
        super(MAT, self).__init__()
        self.num_classes = num_classes
        self.M = M
        if 'xception' in net:
            self.net = xception(num_classes)
        elif net.split('-')[0] == 'efficientnet':
            self.net = EfficientNet.from_pretrained(
                net, advprop=True, num_classes=num_classes)
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.mid_dims = mid_dims
        if pretrained:
            a = torch.load(pretrained, map_location='cpu')
            keys = {i: a['state_dict'][i]
                    for i in a.keys() if i.startswith('net')}
            if not keys:
                keys = a['state_dict']
            self.net.load_state_dict(keys, strict=False)
        self.Frequency_Extract = Frequency_Extract(size, 12, mid_dims)
        self.attentions = AttentionMap(
            layers[self.attention_layer].shape[1], self.M)
        self.attentions_freq = AttentionMap(
            layers[self.attention_layer].shape[1], self.M)
        self.atp = AttentionPooling()
        # Graph feature extract
        # self.DG = DynamicGraph(8, 10)
        self.DG = DynamicGraph(4, 10)
        self.texture_enhance = Texture_Enhance_v2(num_features, M)
        self.num_features = self.texture_enhance.output_features
        self.num_features_d = self.texture_enhance.output_features_d
        self.num_features_freq = self.Frequency_Extract.mid_features
        self.projection_local = nn.Sequential(nn.Linear(
            M*self.num_features, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, mid_dims))
        self.projection_local_freq = nn.Sequential(nn.Linear(
            M*self.num_features_freq, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, mid_dims))
        self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)

        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(
            mid_dims*3, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, num_classes))
        self.auxiliary_loss = Auxiliary_Loss_v2(
            M, self.num_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_freq = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)

    def train_batch(self, x, y, jump_aux=False, drop_final=False):
        layers = self.net(x)

        if self.feature_layer == 'logits':
            logits = layers['logits']
            loss = F.cross_entropy(logits, y)
            return dict(loss=loss, logits=logits)

        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_, fea2_sum_s = self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones(
            [attention_maps_.shape[0], self.M, 1], device=x.device))
        attention_maps = attention_maps_*torch.unsqueeze(dropout_mask, -1)

        # Texture Enhancement
        feature_maps, feature_maps_d = self.texture_enhance(
            feature_maps, attention_maps_)
        feature_maps_d = feature_maps_d - \
            feature_maps_d.mean(dim=[2, 3], keepdim=True)
        feature_maps_d = feature_maps_d / \
            (torch.std(feature_maps_d, dim=[2, 3], keepdim=True)+1e-8)
        feature_matrix_ = self.atp(feature_maps, attention_maps_)

        # Graph feature extract
        feature_matrix_ = self.DG(feature_matrix_)
        feature_matrix = feature_matrix_*dropout_mask

        # Freq feature extraction
        Freq_feature_maps = self.Frequency_Extract(x)
        Freq_attention_maps, fea2_sum_f = self.attentions_freq(raw_attentions)
        dropout_mask = self.dropout_freq(torch.ones(
            [Freq_attention_maps.shape[0], self.M, 1], device=x.device))
        Freq_feature_matrix_ = self.atp(Freq_feature_maps, Freq_attention_maps)

        # Freq_feature_matrix_down = F.interpolate(
        #    Freq_feature_matrix_, feature_matrix_.shape[2])
        # feature_matrix_Fusion_ = self.DG(
        #    torch.cat((feature_matrix_, Freq_feature_matrix_down), dim=1))
        # feature_matrix_Fusion_ = torch.chunk(feature_matrix_Fusion_, 2, 1)
        # feature_matrix_down, Freq_feature_matrix_down = feature_matrix_Fusion_[
        #    0], feature_matrix_Fusion_[1]
        # Freq_feature_matrix_ = Freq_feature_matrix_ + \
        #    F.interpolate(Freq_feature_matrix_down,
        #                  Freq_feature_matrix_.shape[2])
        # feature_matrix_ = feature_matrix_ + feature_matrix_down

        # Freq Graph
        # feature_matrix = feature_matrix_*dropout_mask
        Freq_feature_matrix = Freq_feature_matrix_*dropout_mask

        B, M, N = feature_matrix.size()
        if not jump_aux:
            aux_loss, feature_matrix_d = self.auxiliary_loss(
                feature_maps_d, attention_maps_, y)
        else:
            feature_matrix_d = self.atp(feature_maps_d, attention_maps_)
            aux_loss = 0
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']  # [B, 1792, 11, 11]

        # Freq feature projection
        Freq_feature_matrix = Freq_feature_matrix.view(B, -1)
        Freq_feature_matrix = F.hardswish(
            self.projection_local_freq(Freq_feature_matrix))

        # Freq_attention_maps=Freq_attention_maps.sum(dim=1,keepdim=True)
        # Freq_final=self.atp(Freq_final,Freq_attention_maps,norm=1).squeeze(1)
        # Freq_final=self.dropout_final(Freq_final)
        # Freq_final=F.hardswish(self.project_freq_final(Freq_final))

        attention_maps = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps, norm=1).squeeze(1)
        final = self.dropout_final(final)
        projected_final = F.hardswish(self.project_final(final))
        # projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        if drop_final:
            projected_final *= 0
        feature_matrix = torch.cat(
            (feature_matrix, projected_final, Freq_feature_matrix), 1)
        ensemble_logit = self.ensemble_classifier_fc(
            feature_matrix.squeeze(-1).squeeze(-1))
        ensemble_loss = F.cross_entropy(ensemble_logit, y)
        return dict(ensemble_loss=ensemble_loss, aux_loss=aux_loss, attention_maps=attention_maps_, ensemble_logit=ensemble_logit, feature_matrix=feature_matrix_, feature_matrix_d=feature_matrix_d)

    def forward(self, x, y=0, train_batch=False, AG=None):
        if train_batch:
            if AG is None:
                return self.train_batch(x, y)
            else:
                loss_pack = self.train_batch(x, y)
                with torch.no_grad():
                    Xaug, index = AG.agda(x, loss_pack['attention_maps'])
                # self.eval()
                loss_pack2 = self.train_batch(Xaug, y, jump_aux=False)
                # self.train()
                loss_pack['AGDA_ensemble_loss'] = loss_pack2['ensemble_loss']
                loss_pack['AGDA_aux_loss'] = loss_pack2['aux_loss']
                one_hot = F.one_hot(index, self.M)
                loss_pack['match_loss'] = torch.mean(torch.norm(
                    loss_pack2['feature_matrix_d']-loss_pack['feature_matrix_d'], dim=-1)*(torch.ones_like(one_hot)-one_hot))
                return loss_pack

        layers = self.net(x)
        if self.feature_layer == 'logits':
            logits = layers['logits']
            B = logits.shape[0]
            return logits, layers['b7'].view(B,-1)
        raw_attentions = layers[self.attention_layer]
        attention_maps, fea2_sum_s = self.attentions(raw_attentions)
        feature_maps = layers[self.feature_layer]
        feature_maps, feature_maps_d = self.texture_enhance(feature_maps, attention_maps)
        # torch.save(attention_maps, '/home/wy/Face-forgery-detection/multiple-attention-v1/attention_maps.pt')
        feature_matrix = self.atp(feature_maps, attention_maps)
        feature_matrix = self.DG(feature_matrix)

        # Freq feature extraction
        Freq_feature_maps = self.Frequency_Extract(x)
        Freq_attention_maps, fea2_sum_f = self.attentions_freq(raw_attentions)
        # torch.save(Freq_attention_maps, '/home/wy/Face-forgery-detection/multiple-attention-v1/Freq_attention_maps.pt')
        dropout_mask = self.dropout_freq(torch.ones(
            [Freq_attention_maps.shape[0], self.M, 1], device=x.device))
        Freq_feature_matrix_ = self.atp(Freq_feature_maps, Freq_attention_maps)

        # Freq_feature_matrix_down = F.interpolate(
        #    Freq_feature_matrix_, feature_matrix.shape[2])
        # feature_matrix_Fusion_ = self.DG(
        #    torch.cat((feature_matrix, Freq_feature_matrix_down), dim=1))
        # feature_matrix_Fusion_ = torch.chunk(feature_matrix_Fusion_, 2, 1)
        # feature_matrix_down, Freq_feature_matrix_down = feature_matrix_Fusion_[
        #    0], feature_matrix_Fusion_[1]
        # Freq_feature_matrix_ = Freq_feature_matrix_ + \
        #    F.interpolate(Freq_feature_matrix_down,
        #                  Freq_feature_matrix_.shape[2])
        # feature_matrix = feature_matrix + feature_matrix_down

        # Freq Graph
        # Freq_feature_matrix_ = self.DG2(Freq_feature_matrix_)

        Freq_feature_matrix_ = Freq_feature_matrix_*dropout_mask

        B, M, N = feature_matrix.size()
        feature_matrix = self.dropout(feature_matrix)
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']

        # Freq feature projection
        Freq_feature_matrix = Freq_feature_matrix_.view(B, -1)
        Freq_feature_matrix = F.hardswish(
            self.projection_local_freq(Freq_feature_matrix))

        # Freq_attention_maps2=Freq_attention_maps.sum(dim=1,keepdim=True)
        # Freq_final=self.atp(Freq_final,Freq_attention_maps2,norm=1).squeeze(1)
        # Freq_final=self.dropout_final(Freq_final)
        # Freq_final=F.hardswish(self.project_freq_final(Freq_final))

        attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps2, norm=1).squeeze(1)
        projected_final = F.hardswish(self.project_final(final))
        feature_matrix_all = torch.cat(
            (feature_matrix, projected_final, Freq_feature_matrix), 1)
        # feature_matrix = self.CA(feature_matrix.unsqueeze(-1).unsqueeze(-1))
        ensemble_logit = self.ensemble_classifier_fc(
            feature_matrix_all.squeeze(-1).squeeze(-1))
        return ensemble_logit#, layers['b7'].view(B,-1)#, attention_maps, fea2_sum_s layers['b7'].view(B,-1)


def load_state(net, ckpt):
    sd = net.state_dict()
    nd = {}
    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
    net.load_state_dict(nd, strict=False)
