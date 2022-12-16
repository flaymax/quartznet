from typing import List

import torch
from torch import nn


class QuartzNetBlock(nn.Module):
    def __init__(self, feat_in, filters, repeat=3, kernel_size=11, stride=1,
                 dilation=1, residual=True, separable=False, dropout=0.2):
        super(QuartzNetBlock, self).__init__()
        self.dropout = dropout
        self.res = nn.Sequential(nn.Conv1d(feat_in,
                                           filters,
                                           kernel_size=1),
                                 nn.BatchNorm1d(filters)) if residual else None
        self.conv = nn.ModuleList()
        for idx in range(repeat):
            if dilation > 1:
                same_padding = (dilation * kernel_size) // 2 - 1
            else:
                same_padding = kernel_size // 2
            if separable:
                layers = [
                    nn.Conv1d(feat_in,
                              feat_in,
                              kernel_size,
                              groups=feat_in,
                              stride=stride,
                              dilation=dilation,
                              padding=same_padding),
                    nn.Conv1d(feat_in,
                              filters,
                              kernel_size=1)
                ]
            else:
                layers = [
                    nn.Conv1d(feat_in,
                              filters,
                              kernel_size,
                              stride=stride,
                              dilation=dilation,
                              padding=same_padding)
                ]
            layers.append(nn.BatchNorm1d(filters))

            self.conv.extend(layers)
            if idx != repeat - 1 and residual:
                self.conv.extend([nn.ReLU(),
                                  nn.Dropout(p=dropout)])
            feat_in = filters
        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Dropout(p=dropout))

    def forward(self, inputs):
        inputs_for_res = inputs
        for layer in self.conv:
            inputs = layer(inputs)
        if self.res is not None:
            inputs = inputs + self.res(inputs_for_res)
        inputs = self.out(inputs)
        return inputs


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
