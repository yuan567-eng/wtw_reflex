"""把逐帧数据转换为 Detector 的因果滑动窗口，支持raw/normalized消融。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    PROJECT_ROOT / "dataset/raw/normal_gait/normalization_data/normal_normalized.npz",
    PROJECT_ROOT / "dataset/raw/abnormal_gait/normalization_data/abnormal_normalized.npz",
)
BASE_REQUIRED_FIELDS = (
    "timestamp_us",
    "imu_raw",
    "commands",
    "gait_id",
    "switching",
    "settling",
    "detector_target",
    "loss_mask",
    "episode_id",
)
NORMALIZED_REQUIRED_FIELDS = ("imu_normalized", "commands_normalized")
GAIT_COUNT = 4
FEATURE_COUNT = 14


@dataclass(frozen=True)
class WindowIndex:
    """一个窗口在某个输入文件中的闭区间索引。"""

    source_index: int
    start: int
    end: int


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_feature_file(path: Path, feature_mode: str) -> dict[str, np.ndarray]:
    """读取逐帧文件，并按特征模式校验所需字段。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据：{path}")

    required_fields = BASE_REQUIRED_FIELDS + (
        NORMALIZED_REQUIRED_FIELDS if feature_mode == "normalized" else ()
    )

    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in required_fields if field not in data.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        arrays = {field: np.asarray(data[field]) for field in required_fields}

    sample_count = len(arrays["timestamp_us"])
    expected_shapes = {
        "imu_raw": (sample_count, 6),
        "commands": (sample_count, 4),
    }
    if feature_mode == "normalized":
        expected_shapes.update(
            imu_normalized=(sample_count, 6),
            commands_normalized=(sample_count, 4),
        )
    for field, expected_shape in expected_shapes.items():
        if arrays[field].shape != expected_shape:
            raise ValueError(
                f"{path} 的 {field} shape={arrays[field].shape}，期望 {expected_shape}"
            )
    for field in required_fields:
        if field not in expected_shapes and arrays[field].shape != (sample_count,):
            raise ValueError(
                f"{path} 的 {field} shape={arrays[field].shape}，期望 ({sample_count},)"
            )
    if sample_count == 0:
        raise ValueError(f"{path} 没有样本")
    checked_fields = (
        ("imu_normalized", "commands_normalized")
        if feature_mode == "normalized"
        else ("imu_raw", "commands")
    )
    for field in checked_fields:
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f"{path} 的 {field} 包含 NaN 或 Inf")

    gait_id = arrays["gait_id"]
    if not np.all(np.isin(gait_id, np.arange(GAIT_COUNT))):
        raise ValueError(f"{path} 的 gait_id 必须位于 0~{GAIT_COUNT - 1}")
    if not np.all(np.isin(arrays["detector_target"], (0, 1))):
        raise ValueError(f"{path} 的 detector_target 只能为 0 或 1")
    return arrays


def build_features(arrays: dict[str, np.ndarray], feature_mode: str) -> np.ndarray:
    """拼接 6维 IMU、4维命令和4维 gait one-hot。"""
    gait_id = arrays["gait_id"].astype(np.int64, copy=False)
    gait_one_hot = np.eye(GAIT_COUNT, dtype=np.float32)[gait_id]
    imu_field = "imu_normalized" if feature_mode == "normalized" else "imu_raw"
    command_field = (
        "commands_normalized" if feature_mode == "normalized" else "commands"
    )
    features = np.concatenate(
        (
            arrays[imu_field].astype(np.float32, copy=False),
            arrays[command_field].astype(np.float32, copy=False),
            gait_one_hot,
        ),
        axis=1,
    )
    if features.shape != (len(gait_id), FEATURE_COUNT):
        raise RuntimeError(f"features shape 错误：{features.shape}")
    return features


def find_episode_ranges(episode_id: np.ndarray, path: Path) -> list[tuple[int, int]]:
    """返回每个连续 episode 的 [start, end) 范围，并拒绝重复分段。"""
    boundaries = np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_id)]))

    seen_ids: set[int] = set()
    ranges = []
    for start, end in zip(starts, ends):
        current_id = int(episode_id[start])
        if current_id in seen_ids:
            raise ValueError(f"{path} 的 episode_id={current_id} 出现在多个不连续分段")
        seen_ids.add(current_id)
        ranges.append((int(start), int(end)))
    return ranges


class DetectorWindowDataset(Dataset):
    """按 episode 构造窗口，返回 TCN 使用的 `[14, 100]` 特征。"""

    def __init__(self, paths, window_size=100, stride=4, feature_mode="normalized"):
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size 和 stride 必须大于 0")
        if feature_mode not in ("normalized", "raw"):
            raise ValueError("feature_mode 必须是 normalized 或 raw")

        self.paths = [resolve_path(Path(path)) for path in paths]
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.feature_mode = feature_mode
        self.sources = []
        self.windows: list[WindowIndex] = []

        for source_index, path in enumerate(self.paths):
            arrays = load_feature_file(path, self.feature_mode)
            source = {
                "features": build_features(arrays, self.feature_mode),
                "detector_target": arrays["detector_target"].astype(np.float32),
                "loss_mask": arrays["loss_mask"].astype(np.bool_),
                "episode_id": arrays["episode_id"].astype(np.int64),
            }
            self.sources.append(source)

            for episode_start, episode_end in find_episode_ranges(arrays["episode_id"], path):
                episode_length = episode_end - episode_start
                if episode_length < self.window_size:
                    continue
                last_start = episode_end - self.window_size
                for start in range(episode_start, last_start + 1, self.stride):
                    end = start + self.window_size - 1
                    self.windows.append(WindowIndex(source_index, start, end))

        if not self.windows:
            raise ValueError("没有生成任何窗口；请检查 episode 长度和 window_size")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        source = self.sources[window.source_index]
        # Conv1d/TCN 使用 [channels, time]，因此由 [100,14] 转置为 [14,100]。
        features = source["features"][window.start : window.end + 1].T.copy()
        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(source["detector_target"][window.end]),
            "loss_mask": torch.tensor(
                bool(source["loss_mask"][window.end]), dtype=torch.bool
            ),
            "episode_id": torch.tensor(source["episode_id"][window.end]),
            "end_index": torch.tensor(window.end, dtype=torch.int64),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="验证 Detector 滑动窗口数据加载器")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        default=list(DEFAULT_INPUTS),
        help="一个或多个逐帧 .npz 文件",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("normalized", "raw"),
        default="normalized",
        help="使用归一化连续特征或原始连续特征",
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    return args


def main():
    args = parse_args()
    dataset = DetectorWindowDataset(
        args.inputs, args.window_size, args.stride, args.feature_mode
    )
    targets = np.array(
        [dataset.sources[item.source_index]["detector_target"][item.end] for item in dataset.windows]
    )
    masks = np.array(
        [dataset.sources[item.source_index]["loss_mask"][item.end] for item in dataset.windows]
    )
    episode_ids = np.array(
        [dataset.sources[item.source_index]["episode_id"][item.end] for item in dataset.windows]
    )

    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    print(f"输入文件数：{len(dataset.paths)}")
    print(f"特征模式：{dataset.feature_mode}")
    print(f"episode 数：{len(np.unique(episode_ids))}")
    print(f"窗口总数：{len(dataset)}")
    print(f"有效窗口：{int(masks.sum())}")
    print(f"忽略窗口：{int((~masks).sum())}")
    print(f"正样本窗口：{int((targets == 1).sum())}")
    print(f"有效正样本窗口：{int(((targets == 1) & masks).sum())}")
    print(f"单窗口 features shape：{tuple(dataset[0]['features'].shape)}")
    print(f"DataLoader batch shape：{tuple(batch['features'].shape)}")


if __name__ == "__main__":
    main()
