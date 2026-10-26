"""Detector V2 的单向 GRU 扰动分类网络。"""

from __future__ import annotations

import torch
from torch import nn


class DisturbanceGRU(nn.Module):
    """输入 ``[B,C,T]``，沿时间轴因果读取并输出窗口末端的扰动 logit。"""

    def __init__(self, input_size: int, hidden_size: int = 48, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("input_size、hidden_size、num_layers 必须大于 0")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0,1)")
        self.input_size = input_size
        # 非双向 GRU 只读取窗口中从过去到当前的帧，满足在线因果检测。
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.input_size:
            raise ValueError(f"GRU 输入应为 [B,{self.input_size},T]，实际为 {tuple(inputs.shape)}")
        # 数据集按 [B,C,T] 输出以统一窗口存储；GRU 接口要求 [B,T,C]。
        sequence = inputs.transpose(1, 2)
        output, _ = self.gru(sequence)
        return self.classifier(output[:, -1]).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters() if item.requires_grad)
