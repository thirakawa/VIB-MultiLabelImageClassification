#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F


def self_attention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output


if __name__ == '__main__':

    import torch.nn as nn
    conv = nn.Conv1d(in_channels=49, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

    data = torch.randn([128, 2048, 49], dtype=torch.float32)
    print("input shape:", data.shape)
    out = self_attention(data, data, data, 49)
    print("output shape:", out.shape)

    out_conv = conv(out.transpose(1, 2))
    print("conv shape:", out_conv.shape)

    out_conv_trans = out_conv.view(out_conv.size(0), -1)
    print("view shape:", out_conv_trans.shape)
