"""GRU Detector V2 的 B0/B1/B2 特征与因果滑动窗口。

本文件刻意独立于 TCN 的同名实现，便于之后单独修改 GRU 的特征或窗口策略。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (PROJECT_ROOT / "detector_v2/dataset/raw/train_normal_1min.npz",
                  PROJECT_ROOT / "detector_v2/dataset/raw/train_abnormal_4min.npz")
FEATURE_DIMS = {"b0": 14, "b1": 26, "b2": 38}
REQUIRED = ("timestamp_us", "imu_raw", "commands", "gait_id", "joint_pos", "joint_vel",
            "joint_default_pos", "switching", "settling", "detector_target", "loss_mask", "episode_id")


@dataclass(frozen=True)
class WindowIndex:
    source_index: int
    start: int
    end: int


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_episode_file(path: Path) -> dict[str, np.ndarray]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    with np.load(path, allow_pickle=False) as raw:
        missing = [name for name in REQUIRED if name not in raw.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        result = {name: np.asarray(raw[name]) for name in REQUIRED}
        # 仅供预测结果的事件级评估使用，不参与 GRU 特征构造。
        for name in ("collection_episode_index", "disturbance_force_y"):
            if name in raw.files:
                result[name] = np.asarray(raw[name])
    count = len(result["timestamp_us"])
    expected = {"imu_raw": (count, 6), "commands": (count, 4), "gait_id": (count,),
                "joint_pos": (count, 12), "joint_vel": (count, 12),
                "joint_default_pos": (12,), "detector_target": (count,),
                "loss_mask": (count,), "episode_id": (count,)}
    for name, shape in expected.items():
        if result[name].shape != shape:
            raise ValueError(f"{path} 的 {name}={result[name].shape}，期望 {shape}")
    if not np.all(np.isin(result["gait_id"], np.arange(4))):
        raise ValueError(f"{path} 的 gait_id 必须是 0~3")
    if not np.all(np.isin(result["detector_target"], (0, 1))):
        raise ValueError(f"{path} 的 detector_target 必须是 0/1")
    for name in ("imu_raw", "commands", "joint_pos", "joint_vel", "joint_default_pos"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"{path} 的 {name} 含 NaN/Inf")
    return result


def build_features(arrays: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """B0=14维；B1 增加相对默认姿态；B2 再增加关节速度。"""
    if variant not in FEATURE_DIMS:
        raise ValueError(f"未知 variant：{variant}")
    gait = np.eye(4, dtype=np.float32)[arrays["gait_id"].astype(np.int64)]
    blocks = [arrays["imu_raw"].astype(np.float32), arrays["commands"].astype(np.float32), gait]
    if variant in ("b1", "b2"):
        blocks.append(arrays["joint_pos"].astype(np.float32) - arrays["joint_default_pos"].astype(np.float32))
    if variant == "b2":
        blocks.append(arrays["joint_vel"].astype(np.float32))
    output = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if output.shape[1] != FEATURE_DIMS[variant]:
        raise RuntimeError(f"{variant} 特征维度错误：{output.shape}")
    return output


def find_episode_ranges(ids: np.ndarray, path: Path) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(ids)]
    seen, result = set(), []
    for start, end in zip(starts, ends):
        current = int(ids[start])
        if current in seen:
            raise ValueError(f"{path} 中 episode_id={current} 不是连续片段")
        seen.add(current)
        result.append((int(start), int(end)))
    return result


class GRUWindowDataset(Dataset):
    """返回 [C,T]；GRU 网络入口负责转成 [B,T,C]。"""
    def __init__(self, paths, variant: str, window_size: int = 100, stride: int = 4):
        if variant not in FEATURE_DIMS or window_size <= 0 or stride <= 0:
            raise ValueError("variant、window_size 或 stride 无效")
        self.paths, self.variant = [resolve_path(Path(p)) for p in paths], variant
        self.window_size, self.stride = window_size, stride
        self.sources, self.windows = [], []
        for source_index, path in enumerate(self.paths):
            arrays = load_episode_file(path)
            self.sources.append({"features": build_features(arrays, variant),
                                 "detector_target": arrays["detector_target"].astype(np.float32),
                                 "loss_mask": arrays["loss_mask"].astype(bool),
                                 "episode_id": arrays["episode_id"].astype(np.int64)})
            for begin, finish in find_episode_ranges(arrays["episode_id"], path):
                for start in range(begin, finish - window_size + 1, stride):
                    self.windows.append(WindowIndex(source_index, start, start + window_size - 1))
        if not self.windows:
            raise ValueError("未生成窗口，请检查窗口长度")

    def __len__(self): return len(self.windows)

    def __getitem__(self, index):
        item = self.windows[index]; source = self.sources[item.source_index]
        return {"features": torch.from_numpy(source["features"][item.start:item.end + 1].T.copy()),
                "target": torch.tensor(source["detector_target"][item.end]),
                "loss_mask": torch.tensor(bool(source["loss_mask"][item.end]), dtype=torch.bool),
                "episode_id": torch.tensor(source["episode_id"][item.end]),
                "end_index": torch.tensor(item.end)}


def main():
    parser = argparse.ArgumentParser(description="验证 GRU V2 的 B0/B1/B2 窗口")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--variant", choices=tuple(FEATURE_DIMS), required=True)
    parser.add_argument("--window-size", type=int, default=100); parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args(); data = GRUWindowDataset(args.inputs, args.variant, args.window_size, args.stride)
    batch = next(iter(DataLoader(data, batch_size=4)))
    print(f"variant={args.variant}, windows={len(data)}, single={tuple(data[0]['features'].shape)}, batch={tuple(batch['features'].shape)}")


if __name__ == "__main__": main()
