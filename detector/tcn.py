"""用于二分类扰动检测的轻量因果 TCN。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    """只在时间轴左侧补零，保证输出不使用未来信息。"""

    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, inputs):
        residual = self.residual(inputs)
        outputs = self.conv(F.pad(inputs, (self.left_padding, 0)))
        outputs = self.dropout(self.activation(outputs))
        return self.activation(outputs + residual)


class DisturbanceTCN(nn.Module):
    """输入 `[B,14,100]`，输出每个窗口末端时刻的一个 logit。"""

    def __init__(
        self,
        input_channels=14,
        hidden_channels=32,
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16),
        dropout=0.1,
    ):
        super().__init__()
        blocks = []
        block_input_channels = input_channels
        for dilation in dilations:
            blocks.append(
                CausalResidualBlock(
                    block_input_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
            block_input_channels = hidden_channels
        self.temporal = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, inputs):
        if inputs.ndim != 3:
            raise ValueError(f"TCN 输入必须是 [B,C,T]，实际为 {tuple(inputs.shape)}")
        temporal_features = self.temporal(inputs)
        return self.classifier(temporal_features[:, :, -1]).squeeze(-1)


def receptive_field(kernel_size, dilations):
    """计算每个 block 含一层卷积时的时间感受野。"""
    return 1 + (kernel_size - 1) * sum(dilations)
