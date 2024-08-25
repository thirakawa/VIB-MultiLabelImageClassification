#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

from .self_attention import self_attention


MODEL_NAMES = [
    'resnet50',
    'resnext50_32x4d',
    'densenet169',
    'regnet_y_32gf'
]

def load_resnet_backbone(model_name, pretrained=False):
    if pretrained:
        _resnet_model = get_model(name=model_name, weights="DEFAULT")
    else:
        _resnet_model = get_model(name=model_name, weights=None)

    backbone = nn.Sequential(
        _resnet_model.conv1,
        _resnet_model.bn1,
        _resnet_model.relu,
        _resnet_model.maxpool,
        _resnet_model.layer1,
        _resnet_model.layer2,
        _resnet_model.layer3,
        _resnet_model.layer4
    )

    n_features = _resnet_model.fc.in_features

    return backbone, n_features


def load_densenet_backbone(model_name, pretrained=False):
    if pretrained:
        _densenet_model = get_model(model=model_name, weights="DEFAULT")
    else:
        _densenet_model = get_model(model=model_name, weights=None)

    backbone = nn.Sequential(
        _densenet_model.features,
        nn.ReLU(inplace=True)
    )

    n_features = _densenet_model.classifier.in_features

    return backbone, n_features


def load_regnet_backbone(model_name, pretrained=False):
    if pretrained:
        _regnet_model = get_model(model=model_name, weights="DEFAULT")
    else:
        _regnet_model = get_model(model=model_name, weights=None)

    backbone = nn.Sequential(
        _regnet_model.stem,
        _regnet_model.trunk_output
    )

    n_features = _regnet_model.fc.in_features

    return backbone, n_features


def load_backbone(model_name, pretrained=False):
    if model_name not in MODEL_NAMES:
        print("ERROR: invalid network model name: %s" % model_name)
        exit(-1)

    if model_name.startswith('res'):
        return load_resnet_backbone(model_name=model_name, pretrained=pretrained)
    elif model_name.startswith('dense'):
        return load_densenet_backbone(model_name=model_name, pretrained=pretrained)
    elif model_name.startswith('reg'):
        return load_regnet_backbone(model_name=model_name, pretrained=pretrained)


class VIBClassificationModel2(nn.Module):

    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()

        self.backbone, _n_features = load_backbone(model_name, pretrained)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv1d(in_channels=49, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_clf = nn.Linear(_n_features, num_classes)

        ### initialize FC weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    # TODO: use torch.compile
    def reparameterization_trick(self, mu, sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(sigma / 2.0) * eps

    def forward(self, x, deterministic=False):
        h = self.backbone(x)

        # mu
        h_mu = self.avgpool(h)
        h_mu = torch.flatten(h_mu, 1)

        # sigma
        h_sigma = self_attention(torch.flatten(h, 2), torch.flatten(h, 2), torch.flatten(h, 2), 49)
        h_sigma = self.conv1x1(out.transpose(1, 2))
        h_sigma = F.softplus(torch.flatten(h_sigma, 1), beta=1.0)

        if deterministic:
            z = h_mu
        else:
            z = self.reparameterization_trick(h_mu, h_sigma)
        out = self.fc_clf(z)
        return out, h_mu, h_sigma, z


if __name__ == '__main__':
    
    model1_scratch = VIBClassificationModel2('resnet50', num_classes=40, pretrained=False)
    model1_ftune   = VIBClassificationModel2('resnet50', num_classes=40, pretrained=True)

    model2_scratch = VIBClassificationModel2('resnext50_32x4d', num_classes=40, pretrained=False)
    model2_ftune   = VIBClassificationModel2('resnext50_32x4d', num_classes=40, pretrained=True)

    model3_scratch = VIBClassificationModel2('densenet169', num_classes=40, pretrained=False)
    model3_ftune   = VIBClassificationModel2('densenet169', num_classes=40, pretrained=True)

    model4_scratch = VIBClassificationModel2('regnet_y_32gf', num_classes=40, pretrained=False)
    model4_ftune   = VIBClassificationModel2('regnet_y_32gf', num_classes=40, pretrained=True)
