"""用于二分类扰动检测的轻量单向 GRU。"""

from __future__ import annotations

import torch
from torch import nn


class DisturbanceGRU(nn.Module):
    """将一个因果 IMU 窗口映射为窗口末端时刻的扰动 logit。

    数据集为了与 TCN 的 ``Conv1d`` 接口保持一致，输出 shape 为
    ``[batch, channels, time]``。GRU 使用 ``[batch, time, features]``，
    因此在网络入口处只转置时间轴和特征轴。
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size 必须大于 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须大于 0")
        if num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")

        # 单向 GRU 只按过去到当前的方向更新隐藏状态，不会使用未来帧。
        # PyTorch 仅在层数大于 1 时才会在层间应用 dropout。
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 ``[B, 14, 100]``，输出每个窗口末端的一个 logit ``[B]``。"""
        if inputs.ndim != 3:
            raise ValueError(
                f"GRU 输入必须是 [B,C,T]，实际为 {tuple(inputs.shape)}"
            )
        sequence = inputs.transpose(1, 2)
        outputs, _ = self.gru(sequence)

        # 最后一帧隐藏状态只汇总窗口内从过去到当前的信息，对应当前 Pd(t)。
        return self.classifier(outputs[:, -1, :]).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    """返回可训练参数总数，用于与 TCN 保持容量接近的对比。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
