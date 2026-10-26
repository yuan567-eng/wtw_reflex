"""Detector V2 使用的轻量因果 TCN。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class CausalResidualBlock(nn.Module):
    """仅在时间轴左侧补零的残差卷积块，当前输出不读取未来帧。"""

    def __init__(self, input_channels, output_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=kernel_size, dilation=dilation
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
        outputs = self.conv(functional.pad(inputs, (self.left_padding, 0)))
        outputs = self.dropout(self.activation(outputs))
        return self.activation(outputs + residual)


class DisturbanceTCN(nn.Module):
    """输入 ``[B,C,100]``，输出每个窗口末帧的扰动 logit。

    C 由消融配置决定：B0=14、B1=26、B2=38、B3=26、B3-delta=30。模型本身不关心具体配置，
    训练 checkpoint 会保存 input_channels 与 variant 以避免推理时用错输入。
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        dilations=(1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0 or kernel_size <= 0:
            raise ValueError("input_channels、hidden_channels、kernel_size 必须大于 0")
        blocks = []
        channels = input_channels
        for dilation in dilations:
            blocks.append(
                CausalResidualBlock(
                    channels, hidden_channels, kernel_size, dilation, dropout
                )
            )
            channels = hidden_channels
        self.input_channels = input_channels
        self.temporal = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, inputs):
        if inputs.ndim != 3:
            raise ValueError(f"TCN 输入必须为 [B,C,T]，实际为 {tuple(inputs.shape)}")
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"TCN 输入通道为 {inputs.shape[1]}，模型期望 {self.input_channels}"
            )
        temporal_features = self.temporal(inputs)
        return self.classifier(temporal_features[:, :, -1]).squeeze(-1)


def receptive_field(kernel_size: int, dilations) -> int:
    """计算每个 block 含一层卷积时的时间感受野。"""
    return 1 + (kernel_size - 1) * sum(dilations)
