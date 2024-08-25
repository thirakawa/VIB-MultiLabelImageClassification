#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model


MODEL_NAMES = [
    'resnet50',
    'resnext50_32x4d',
    'densenet169',
    'regnet_y_32gf'
]


def load_backbone(model_name, pretrained=False):
    if model_name not in MODEL_NAMES:
        print("ERROR: invalid network model name: %s" % model_name)
        exit(-1)

    if pretrained:
        backbone = get_model(name=model_name, weights="DEFAULT")
    else:
        backbone = get_model(name=model_name, weights=None)

    for _l_name, _layer in backbone.named_modules():
        if type(_layer) is nn.Linear:
            break
    n_features = _layer.in_features
    setattr(backbone, _l_name, nn.Identity())

    return backbone, n_features


class VIBClassificationModel1(nn.Module):

    def __init__(self, model_name, num_classes, embed_dim, pretrained=False):
        super().__init__()

        self.backbone, _n_features = load_backbone(model_name, pretrained)

        self.fc_mu    = nn.Linear(_n_features, embed_dim)  # for mean
        self.fc_sigma = nn.Linear(_n_features, embed_dim)  # for co-variance

        self.fc_clf = nn.Linear(embed_dim, num_classes)

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
        h_mu = self.fc_mu(h)                              # mu
        h_sigma = F.softplus(self.fc_sigma(h), beta=1.0)  # sigma
        if deterministic:
            z = h_mu
        else:
            z = self.reparameterization_trick(h_mu, h_sigma)
        out = self.fc_clf(z)
        return out, h_mu, h_sigma, z


if __name__ == '__main__':
    
    model1_scratch = VIBClassificationModel1('resnet50', num_classes=40, embed_dim=512, pretrained=False)
    model1_ftune   = VIBClassificationModel1('resnet50', num_classes=40, embed_dim=512, pretrained=True)

    model2_scratch = VIBClassificationModel1('resnext50_32x4d', num_classes=40, embed_dim=512, pretrained=False)
    model2_ftune   = VIBClassificationModel1('resnext50_32x4d', num_classes=40, embed_dim=512, pretrained=True)

    model3_scratch = VIBClassificationModel1('densenet169', num_classes=40, embed_dim=512, pretrained=False)
    model3_ftune   = VIBClassificationModel1('densenet169', num_classes=40, embed_dim=512, pretrained=True)

    model4_scratch = VIBClassificationModel1('regnet_y_32gf', num_classes=40, embed_dim=512, pretrained=False)
    model4_ftune   = VIBClassificationModel1('regnet_y_32gf', num_classes=40, embed_dim=512, pretrained=True)
