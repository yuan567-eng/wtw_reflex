"""合并多个 gait episode，生成统一的总数据集。

本脚本只负责校验和拼接，不修改 IMU 数值，也不执行归一化。
后续构造 TCN 滑动窗口时，必须依据 episode_id 分段，禁止跨 episode 取窗口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")
REQUIRED_FIELDS = (
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


def parse_args():
    parser = argparse.ArgumentParser(description="合并多步态 episode 数据")
    parser.add_argument(
        "--category",
        choices=("normal_gait", "abnormal_gait"),
        default="normal_gait",
        help="需要合并的数据类别",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选的输出 .npz 路径；默认写入对应 total_gait_data 目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的总数据集",
    )
    return parser.parse_args()


def load_gait_mapping() -> dict[str, int]:
    mapping_path = DATASET_ROOT / "metadata/gait_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    missing = [name for name in GAIT_NAMES if name not in mapping]
    if missing:
        raise ValueError(f"gait_mapping.json 缺少步态：{missing}")
    return {name: int(mapping[name]) for name in GAIT_NAMES}


def discover_episode_files(category_root: Path) -> list[tuple[str, Path]]:
    """按照固定 gait 顺序发现输入文件，保证每次合并结果可复现。"""
    episode_files: list[tuple[str, Path]] = []
    for gait_name in (*GAIT_NAMES, "gait_switching"):
        gait_dir = category_root / gait_name
        if not gait_dir.is_dir():
            continue
        for path in sorted(gait_dir.glob("*.npz")):
            episode_files.append((gait_name, path))

    if not episode_files:
        raise FileNotFoundError(f"没有在 {category_root} 下找到 episode .npz 文件")
    return episode_files


def validate_episode(
    path: Path,
    source_name: str,
    category: str,
    gait_mapping: dict[str, int],
) -> dict[str, np.ndarray]:
    """读取并校验一个采集文件；文件内部可以包含多个连续 episode。"""
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        episode = {field: np.asarray(data[field]) for field in REQUIRED_FIELDS}

    sample_count = len(episode["timestamp_us"])
    if sample_count == 0:
        raise ValueError(f"{path} 是空 episode")
    if episode["imu_raw"].shape != (sample_count, 6):
        raise ValueError(
            f"{path} 的 imu_raw shape={episode['imu_raw'].shape}，"
            f"期望 ({sample_count}, 6)"
        )
    if episode["commands"].shape != (sample_count, 4):
        raise ValueError(
            f"{path} 的 commands shape={episode['commands'].shape}，"
            f"期望 ({sample_count}, 4)"
        )
    for field in REQUIRED_FIELDS:
        if field in ("imu_raw", "commands"):
            continue
        if episode[field].shape != (sample_count,):
            raise ValueError(
                f"{path} 的 {field} shape={episode[field].shape}，"
                f"期望 ({sample_count},)"
            )

    if not np.isfinite(episode["imu_raw"]).all():
        raise ValueError(f"{path} 的 imu_raw 包含 NaN 或 Inf")
    if not np.isfinite(episode["commands"]).all():
        raise ValueError(f"{path} 的 commands 包含 NaN 或 Inf")
    timestamp_us = episode["timestamp_us"]
    if not np.issubdtype(timestamp_us.dtype, np.integer):
        raise ValueError(f"{path} 的 timestamp_us 必须是整数类型")
    episode_id = episode["episode_id"]
    boundaries = np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sample_count]))
    segment_ids = episode_id[starts]
    if len(np.unique(segment_ids)) != len(segment_ids):
        raise ValueError(f"{path} 的同一个 episode_id 出现在多个不连续区段")
    for start, end in zip(starts, ends):
        if end - start > 1 and not np.all(np.diff(timestamp_us[start:end]) > 0):
            raise ValueError(
                f"{path} 的 episode_id={int(episode_id[start])} 内时间戳不是严格递增"
            )

    gait_ids = np.unique(episode["gait_id"])
    if source_name in gait_mapping:
        expected_gait_id = gait_mapping[source_name]
        if not np.array_equal(gait_ids, np.array([expected_gait_id])):
            raise ValueError(
                f"{path} 位于 {source_name} 目录，但 gait_id={gait_ids}，"
                f"期望 {expected_gait_id}"
            )

    targets = np.unique(episode["detector_target"])
    if category == "normal_gait" and not np.array_equal(targets, np.array([0])):
        raise ValueError(f"正常数据 {path} 的 detector_target 必须全部为 0")
    if not np.all(np.isin(targets, (0, 1))):
        raise ValueError(f"{path} 的 detector_target 只能是 0 或 1，实际为 {targets}")

    return episode


def resolve_output_path(args, category_root: Path) -> Path:
    if args.output is None:
        prefix = "normal" if args.category == "normal_gait" else "abnormal"
        return category_root / "total_gait_data" / f"{prefix}_total.npz"
    if args.output.is_absolute():
        return args.output
    return PROJECT_ROOT / args.output


def merge_episodes(args) -> Path:
    category_root = DATASET_ROOT / "raw" / args.category
    gait_mapping = load_gait_mapping()
    episode_files = discover_episode_files(category_root)

    loaded_files: list[dict[str, np.ndarray]] = []
    seen_episode_ids: dict[int, Path] = {}
    for source_name, path in episode_files:
        collection = validate_episode(path, source_name, args.category, gait_mapping)
        boundaries = np.flatnonzero(
            collection["episode_id"][1:] != collection["episode_id"][:-1]
        ) + 1
        starts = np.concatenate(([0], boundaries))
        file_episode_ids = [int(value) for value in collection["episode_id"][starts]]
        for episode_id in file_episode_ids:
            if episode_id in seen_episode_ids:
                raise ValueError(
                    f"episode_id={episode_id} 重复：{seen_episode_ids[episode_id]} 与 {path}"
                )
            seen_episode_ids[episode_id] = path
        loaded_files.append(collection)
        print(
            f"已检查：{source_name}/{path.name}，"
            f"samples={len(collection['timestamp_us'])}，episodes={len(file_episode_ids)}"
        )

    # 只沿时间轴拼接；episode_id 保留原值，供后续窗口构造时识别边界。
    merged = {
        field: np.concatenate([collection[field] for collection in loaded_files], axis=0)
        for field in REQUIRED_FIELDS
    }

    output_path = resolve_output_path(args, category_root)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在：{output_path}；如需重建请添加 --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    gait_ids, gait_counts = np.unique(merged["gait_id"], return_counts=True)
    distribution = ", ".join(
        f"gait_id={int(gait_id)}: {int(count)}"
        for gait_id, count in zip(gait_ids, gait_counts)
    )
    print(f"已保存：{output_path}")
    print(
        f"files={len(loaded_files)}, episodes={len(seen_episode_ids)}, "
        f"samples={len(merged['timestamp_us'])}, "
        f"imu_shape={merged['imu_raw'].shape}, commands_shape={merged['commands'].shape}"
    )
    print(f"样本分布：{distribution}")
    return output_path


if __name__ == "__main__":
    merge_episodes(parse_args())
