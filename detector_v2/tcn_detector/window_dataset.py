"""TCN Detector V2 的 B0/B1/B2 特征构造与因果滑动窗口数据集。

本文件只服务 TCN。GRU 后续会维护自己的数据处理实现，以便两种网络独立调参。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    PROJECT_ROOT / "detector_v2/dataset/raw/train_normal_1min.npz",
    PROJECT_ROOT / "detector_v2/dataset/raw/train_abnormal_4min.npz",
)
GAIT_COUNT = 4
FEATURE_DIMS = {"b0": 14, "b1": 26, "b2": 38, "b3": 26, "b3_delta": 30}
REQUIRED_FIELDS = (
    "timestamp_us", "imu_raw", "commands", "gait_id", "joint_pos",
    "joint_vel", "joint_default_pos", "switching", "settling",
    "detector_target", "loss_mask", "episode_id",
)
# 这些字段不参与网络输入，但预测/评估需要将其传递到窗口末帧。
OPTIONAL_METADATA_FIELDS = (
    "collection_episode_index", "disturbance_force_y", "scenario_id", "scenario_names",
)


@dataclass(frozen=True)
class WindowIndex:
    """一个因果窗口的来源与闭区间索引。"""

    source_index: int
    start: int
    end: int


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_episode_file(path: Path) -> dict[str, np.ndarray]:
    """读取并校验 V2 原始文件；不在这里做归一化，保证本轮只消融关节特征。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        arrays = {name: np.asarray(data[name]) for name in REQUIRED_FIELDS}
        # B3 专用字段：旧 B0/B1/B2 数据不含该字段，不能作为全局必填项。
        if "joint_torque" in data.files:
            arrays["joint_torque"] = np.asarray(data["joint_torque"])
        for name in OPTIONAL_METADATA_FIELDS:
            if name in data.files:
                arrays[name] = np.asarray(data[name])

    sample_count = len(arrays["timestamp_us"])
    if sample_count == 0:
        raise ValueError(f"{path} 没有采样帧")
    expected_shapes = {
        "imu_raw": (sample_count, 6),
        "commands": (sample_count, 4),
        "joint_pos": (sample_count, 12),
        "joint_vel": (sample_count, 12),
        "gait_id": (sample_count,),
        "switching": (sample_count,),
        "settling": (sample_count,),
        "detector_target": (sample_count,),
        "loss_mask": (sample_count,),
        "episode_id": (sample_count,),
        "joint_default_pos": (12,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{path} 的 {name}={arrays[name].shape}，期望 {expected}")
    for name in ("imu_raw", "commands", "joint_pos", "joint_vel", "joint_default_pos"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path} 的 {name} 含 NaN 或 Inf")
    if "joint_torque" in arrays:
        if arrays["joint_torque"].shape != (sample_count, 12):
            raise ValueError(
                f"{path} 的 joint_torque={arrays['joint_torque'].shape}，期望 {(sample_count, 12)}"
            )
        if not np.isfinite(arrays["joint_torque"]).all():
            raise ValueError(f"{path} 的 joint_torque 含 NaN 或 Inf")
    if not np.all(np.isin(arrays["gait_id"], np.arange(GAIT_COUNT))):
        raise ValueError(f"{path} 的 gait_id 必须在 0~{GAIT_COUNT - 1}")
    if not np.all(np.isin(arrays["detector_target"], (0, 1))):
        raise ValueError(f"{path} 的 detector_target 只能是 0 或 1")
    return arrays


def command_delta(commands: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    """计算因果命令差分，并在每个 episode 第一帧置零。

    命令值本来已经位于窗口历史中；显式差分只用于让网络更容易感知命令切换。
    必须清除 episode 边界的差分，避免 reset 前后的命令跳变泄漏为伪特征。
    """
    delta = np.zeros_like(commands, dtype=np.float32)
    same_episode = episode_id[1:] == episode_id[:-1]
    delta[1:][same_episode] = commands[1:][same_episode] - commands[:-1][same_episode]
    return delta


def build_features(arrays: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """按 B0/B1/B2/B3/B3-delta 定义构造每帧特征，输出 [T, feature_dim]。

    B1 使用 q_rel=q-q_default；B2 在 B1 后附加真实关节速度。IMU 与命令保持原始
    物理尺度，避免本轮实验同时混入归一化方法这一变量。
    """
    if variant not in FEATURE_DIMS:
        raise ValueError(f"未知 variant：{variant}")
    gait_id = arrays["gait_id"].astype(np.int64, copy=False)
    gait_one_hot = np.eye(GAIT_COUNT, dtype=np.float32)[gait_id]
    blocks = [
        arrays["imu_raw"].astype(np.float32, copy=False),
        arrays["commands"].astype(np.float32, copy=False),
        gait_one_hot,
    ]
    if variant in ("b1", "b2"):
        q_rel = arrays["joint_pos"].astype(np.float32, copy=False) - arrays[
            "joint_default_pos"
        ].astype(np.float32, copy=False)[None, :]
        blocks.append(q_rel)
    if variant == "b2":
        blocks.append(arrays["joint_vel"].astype(np.float32, copy=False))
    if variant in ("b3", "b3_delta"):
        if "joint_torque" not in arrays:
            raise ValueError("B3/B3-delta 需要 joint_torque[T,12]；请使用字段完整的 B3 数据集")
        blocks.append(arrays["joint_torque"].astype(np.float32, copy=False))
    if variant == "b3_delta":
        blocks.append(command_delta(arrays["commands"].astype(np.float32, copy=False), arrays["episode_id"]))
    features = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    expected_dim = FEATURE_DIMS[variant]
    if features.shape != (len(gait_id), expected_dim):
        raise RuntimeError(f"{variant} features={features.shape}，期望 [T,{expected_dim}]")
    return features


def find_episode_ranges(episode_id: np.ndarray, path: Path) -> list[tuple[int, int]]:
    """返回连续 episode 的 [start,end) 区间，并拒绝同一 id 的不连续重复。"""
    boundaries = np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_id)]))
    seen_ids: set[int] = set()
    result = []
    for start, end in zip(starts, ends):
        current_id = int(episode_id[start])
        if current_id in seen_ids:
            raise ValueError(f"{path} 的 episode_id={current_id} 出现在不连续片段")
        seen_ids.add(current_id)
        result.append((int(start), int(end)))
    return result


class TCNWindowDataset(Dataset):
    """以窗口末帧标签监督 TCN，特征布局为 Conv1d 所需的 [C,T]。"""

    def __init__(self, paths, variant: str, window_size: int = 100, stride: int = 4):
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size 与 stride 必须大于 0")
        if variant not in FEATURE_DIMS:
            raise ValueError(f"variant 必须是 {tuple(FEATURE_DIMS)}")
        self.paths = [resolve_path(Path(path)) for path in paths]
        self.variant = variant
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sources = []
        self.windows: list[WindowIndex] = []

        for source_index, path in enumerate(self.paths):
            arrays = load_episode_file(path)
            self.sources.append(
                {
                    "features": build_features(arrays, variant),
                    "detector_target": arrays["detector_target"].astype(np.float32),
                    "loss_mask": arrays["loss_mask"].astype(np.bool_),
                    "episode_id": arrays["episode_id"].astype(np.int64),
                }
            )
            for episode_start, episode_end in find_episode_ranges(arrays["episode_id"], path):
                last_start = episode_end - self.window_size
                for start in range(episode_start, last_start + 1, self.stride):
                    self.windows.append(WindowIndex(source_index, start, start + self.window_size - 1))
        if not self.windows:
            raise ValueError("未生成任何窗口，请检查 episode 长度与 window_size")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        source = self.sources[window.source_index]
        # [100,C] 转置为 TCN/Conv1d 使用的 [C,100]。
        features = source["features"][window.start : window.end + 1].T.copy()
        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(source["detector_target"][window.end]),
            "loss_mask": torch.tensor(bool(source["loss_mask"][window.end]), dtype=torch.bool),
            "episode_id": torch.tensor(source["episode_id"][window.end]),
            "end_index": torch.tensor(window.end, dtype=torch.int64),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="验证 TCN V2 B0/B1/B2 窗口数据集")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--variant", choices=tuple(FEATURE_DIMS), required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须大于 0")
    return args


def main():
    args = parse_args()
    dataset = TCNWindowDataset(args.inputs, args.variant, args.window_size, args.stride)
    targets = np.array(
        [dataset.sources[item.source_index]["detector_target"][item.end] for item in dataset.windows]
    )
    masks = np.array(
        [dataset.sources[item.source_index]["loss_mask"][item.end] for item in dataset.windows]
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    print(f"variant：{dataset.variant}（每帧 {FEATURE_DIMS[dataset.variant]} 维）")
    print(f"输入文件数：{len(dataset.paths)}")
    print(f"窗口总数：{len(dataset)}")
    print(f"有效窗口：{int(masks.sum())}；忽略窗口：{int((~masks).sum())}")
    print(f"正样本窗口：{int((targets == 1).sum())}；有效正样本：{int(((targets == 1) & masks).sum())}")
    print(f"单窗口 shape：{tuple(dataset[0]['features'].shape)}")
    print(f"batch shape：{tuple(batch['features'].shape)}")


if __name__ == "__main__":
    main()
