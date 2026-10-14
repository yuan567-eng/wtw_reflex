"""拟合正常数据归一化参数，或使用正常参数转换其他数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"

DEFAULT_INPUT = DATASET_ROOT / "raw/abnormal_gait/trotting/episode_1791376876691668.npz"
DEFAULT_OUTPUT = DATASET_ROOT / "raw/abnormal_gait/normalization_data/episode_1791376876691668_normalized.npz"
GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")
# 命令按 checkpoint 的固定训练范围映射，顺序为 vx、vy、yaw_rate、gait_frequency。
COMMAND_CENTER = np.array([0.0, 0.0, 0.0, 3.0], dtype=np.float32)
COMMAND_SCALE = np.array([1.0, 0.6, 1.0, 1.0], dtype=np.float32)
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
    parser = argparse.ArgumentParser(description="按 gait 归一化六维 IMU 数据")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 normal_total.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 normal_normalized.npz")
    parser.add_argument(
        "--reference-normalized",
        type=Path,
        default=None,
        help="可选的正常归一化文件；提供后只复用其参数，不从当前输入重新拟合",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.0,
        help="归一化下分位数，取值范围 [0, 1]，默认 0.0（最小值）",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=1.0,
        help="归一化上分位数，取值范围 [0, 1]，默认 1.0（最大值）",
    )
    parser.add_argument(
        "--accel-min-scale",
        type=float,
        default=0.1,
        help="加速度通道的最小尺度，单位 m/s²",
    )
    parser.add_argument(
        "--gyro-min-scale",
        type=float,
        default=0.01,
        help="角速度通道的最小尺度，单位 rad/s",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6, help="分母数值稳定项")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    args = parser.parse_args()
    if args.accel_min_scale <= 0 or args.gyro_min_scale <= 0:
        parser.error("最小尺度必须大于 0")
    if args.epsilon <= 0:
        parser.error("epsilon 必须大于 0")
    if not 0.0 <= args.low_quantile < args.high_quantile <= 1.0:
        parser.error("分位数必须满足 0 <= low_quantile < high_quantile <= 1")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_gait_mapping() -> dict[str, int]:
    mapping_path = DATASET_ROOT / "metadata/gait_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    missing = [name for name in GAIT_NAMES if name not in raw_mapping]
    if missing:
        raise ValueError(f"gait_mapping.json 缺少步态：{missing}")
    mapping = {name: int(raw_mapping[name]) for name in GAIT_NAMES}

    # center 和 scale 的第一维直接由 gait_id 索引，因此 ID 必须是 0~3 的一个排列。
    expected_ids = set(range(len(GAIT_NAMES)))
    if set(mapping.values()) != expected_ids:
        raise ValueError(
            f"四种 gait_id 必须是 {sorted(expected_ids)} 的一个排列，实际为 {mapping}"
        )
    return mapping


def load_total_data(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入数据：{path}")

    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"输入数据缺少字段：{missing}")
        arrays = {field: np.asarray(data[field]) for field in REQUIRED_FIELDS}

    sample_count = len(arrays["timestamp_us"])
    if arrays["imu_raw"].shape != (sample_count, 6):
        raise ValueError(
            f"imu_raw shape={arrays['imu_raw'].shape}，期望 ({sample_count}, 6)"
        )
    if arrays["commands"].shape != (sample_count, 4):
        raise ValueError(
            f"commands shape={arrays['commands'].shape}，期望 ({sample_count}, 4)"
        )
    for field in REQUIRED_FIELDS:
        if field not in ("imu_raw", "commands") and arrays[field].shape != (sample_count,):
            raise ValueError(f"{field} shape={arrays[field].shape}，期望 ({sample_count},)")
    if sample_count == 0:
        raise ValueError("输入数据没有样本")
    if not np.isfinite(arrays["imu_raw"]).all():
        raise ValueError("imu_raw 包含 NaN 或 Inf")
    if not np.isfinite(arrays["commands"]).all():
        raise ValueError("commands 包含 NaN 或 Inf")
    return arrays


def fit_and_transform(
    arrays: dict[str, np.ndarray],
    gait_mapping: dict[str, int],
    minimum_scale: np.ndarray,
    epsilon: float,
    low_quantile: float,
    high_quantile: float,
):
    imu_raw = arrays["imu_raw"].astype(np.float32, copy=False)
    gait_id = arrays["gait_id"]
    observed_ids = set(int(value) for value in np.unique(gait_id))
    known_ids = set(gait_mapping.values())
    unknown_ids = observed_ids.difference(known_ids)
    if unknown_ids:
        raise ValueError(f"数据中存在 gait_mapping 未定义的 gait_id：{sorted(unknown_ids)}")

    # 未出现的 gait 使用恒等占位参数，并由 fitted_gait_mask 明确标记为不可用。
    center = np.zeros((len(GAIT_NAMES), 6), dtype=np.float32)
    scale = np.ones((len(GAIT_NAMES), 6), dtype=np.float32)
    fitted_gait_mask = np.zeros(len(GAIT_NAMES), dtype=np.bool_)
    imu_normalized = np.empty_like(imu_raw, dtype=np.float32)

    # 只用正常、稳定且参与训练的样本拟合统计量，切换与 settling 样本不污染 baseline。
    stable_normal = (
        arrays["loss_mask"].astype(bool)
        & (arrays["detector_target"] == 0)
        & ~arrays["switching"].astype(bool)
        & ~arrays["settling"].astype(bool)
    )

    summaries = []
    for gait_name in GAIT_NAMES:
        gait_id_value = gait_mapping[gait_name]
        gait_samples = gait_id == gait_id_value
        if not np.any(gait_samples):
            continue
        fit_samples = gait_samples & stable_normal
        fit_count = int(np.count_nonzero(fit_samples))
        if fit_count == 0:
            raise ValueError(f"{gait_name} 没有可用于拟合归一化参数的正常稳定样本")

        q_low, q_high = np.quantile(
            imu_raw[fit_samples], (low_quantile, high_quantile), axis=0
        )
        gait_center = (q_high + q_low) / 2.0
        raw_scale = (q_high - q_low) / 2.0
        gait_scale = np.maximum(raw_scale, minimum_scale)

        center[gait_id_value] = gait_center.astype(np.float32)
        scale[gait_id_value] = gait_scale.astype(np.float32)
        fitted_gait_mask[gait_id_value] = True
        imu_normalized[gait_samples] = (
            (imu_raw[gait_samples] - center[gait_id_value])
            / (scale[gait_id_value] + epsilon)
        )

        normalized_fit = imu_normalized[fit_samples]
        normalized_q_low, normalized_q_high = np.quantile(
            normalized_fit, (low_quantile, high_quantile), axis=0
        )
        summaries.append(
            (
                gait_name, gait_id_value, fit_count,
                normalized_q_low, normalized_q_high,
            )
        )

    if not np.isfinite(imu_normalized).all():
        raise RuntimeError("imu_normalized 包含 NaN 或 Inf")
    return imu_normalized, center, scale, fitted_gait_mask, summaries


def transform_with_stats(
    arrays: dict[str, np.ndarray],
    gait_mapping: dict[str, int],
    center: np.ndarray,
    scale: np.ndarray,
    fitted_gait_mask: np.ndarray,
    epsilon: float,
):
    """使用已经由正常数据拟合的参数转换 IMU。"""
    imu_raw = arrays["imu_raw"].astype(np.float32, copy=False)
    gait_id = arrays["gait_id"]
    known_ids = set(gait_mapping.values())
    observed_ids = set(int(value) for value in np.unique(gait_id))
    unknown_ids = observed_ids.difference(known_ids)
    if unknown_ids:
        raise ValueError(f"数据中存在 gait_mapping 未定义的 gait_id：{sorted(unknown_ids)}")
    unfitted_ids = sorted(value for value in observed_ids if not fitted_gait_mask[value])
    if unfitted_ids:
        raise ValueError(f"正常参考文件没有拟合这些 gait_id：{unfitted_ids}")

    imu_normalized = np.empty_like(imu_raw, dtype=np.float32)
    for gait_id_value in observed_ids:
        gait_samples = gait_id == gait_id_value
        imu_normalized[gait_samples] = (
            (imu_raw[gait_samples] - center[gait_id_value])
            / (scale[gait_id_value] + epsilon)
        )
    if not np.isfinite(imu_normalized).all():
        raise RuntimeError("imu_normalized 包含 NaN 或 Inf")
    return imu_normalized


def load_reference_stats(path: Path):
    """读取并严格校验正常归一化文件中的参数。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到正常参考文件：{path}")
    required = (
        "center", "scale", "fitted_gait_mask", "command_center", "command_scale",
        "epsilon", "quantile_low", "quantile_high",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ValueError(f"正常参考文件缺少字段：{missing}")
        stats = {field: np.asarray(data[field]) for field in required}

    expected_shapes = {
        "center": (len(GAIT_NAMES), 6),
        "scale": (len(GAIT_NAMES), 6),
        "fitted_gait_mask": (len(GAIT_NAMES),),
        "command_center": (4,),
        "command_scale": (4,),
    }
    for field, expected_shape in expected_shapes.items():
        if stats[field].shape != expected_shape:
            raise ValueError(
                f"正常参考文件的 {field} shape={stats[field].shape}，期望 {expected_shape}"
            )
    if np.any(stats["scale"] <= 0) or np.any(stats["command_scale"] <= 0):
        raise ValueError("正常参考文件中的 scale 必须全部大于 0")
    epsilon = float(stats["epsilon"])
    if epsilon <= 0:
        raise ValueError("正常参考文件中的 epsilon 必须大于 0")
    return stats


def normalize_commands(
    commands: np.ndarray,
    epsilon: float,
    command_center: np.ndarray = COMMAND_CENTER,
    command_scale: np.ndarray = COMMAND_SCALE,
):
    commands = commands.astype(np.float32, copy=False)
    command_low = command_center - command_scale
    command_high = command_center + command_scale
    tolerance = 1e-6
    if np.any(commands < command_low - tolerance) or np.any(commands > command_high + tolerance):
        actual_low = commands.min(axis=0)
        actual_high = commands.max(axis=0)
        raise ValueError(
            "commands 超出 checkpoint 训练范围；"
            f"实际 min={actual_low}, max={actual_high}"
        )
    normalized = (commands - command_center) / (command_scale + epsilon)
    if not np.isfinite(normalized).all():
        raise RuntimeError("commands_normalized 包含 NaN 或 Inf")
    return normalized.astype(np.float32, copy=False)


def normalize_dataset(args) -> Path:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在：{output_path}；如需重建请添加 --overwrite")

    arrays = load_total_data(input_path)
    gait_mapping = load_gait_mapping()
    if args.reference_normalized is None:
        mode = "fit_normal"
        epsilon = args.epsilon
        quantile_low = args.low_quantile
        quantile_high = args.high_quantile
        command_center = COMMAND_CENTER.copy()
        command_scale = COMMAND_SCALE.copy()
        minimum_scale = np.array(
            [args.accel_min_scale] * 3 + [args.gyro_min_scale] * 3,
            dtype=np.float32,
        )
        imu_normalized, center, scale, fitted_gait_mask, summaries = fit_and_transform(
            arrays,
            gait_mapping,
            minimum_scale,
            epsilon,
            quantile_low,
            quantile_high,
        )
    else:
        mode = "transform_with_normal_reference"
        reference_path = resolve_path(args.reference_normalized)
        stats = load_reference_stats(reference_path)
        center = stats["center"].astype(np.float32, copy=False)
        scale = stats["scale"].astype(np.float32, copy=False)
        fitted_gait_mask = stats["fitted_gait_mask"].astype(np.bool_, copy=False)
        command_center = stats["command_center"].astype(np.float32, copy=False)
        command_scale = stats["command_scale"].astype(np.float32, copy=False)
        epsilon = float(stats["epsilon"])
        quantile_low = float(stats["quantile_low"])
        quantile_high = float(stats["quantile_high"])
        imu_normalized = transform_with_stats(
            arrays, gait_mapping, center, scale, fitted_gait_mask, epsilon
        )
        summaries = []

    commands_normalized = normalize_commands(
        arrays["commands"], epsilon, command_center, command_scale
    )

    output = dict(arrays)
    output.update(
        imu_normalized=imu_normalized,
        commands_normalized=commands_normalized,
        center=center,
        scale=scale,
        fitted_gait_mask=fitted_gait_mask,
        command_center=command_center,
        command_scale=command_scale,
        epsilon=np.asarray(epsilon, dtype=np.float32),
        quantile_low=np.asarray(quantile_low, dtype=np.float32),
        quantile_high=np.asarray(quantile_high, dtype=np.float32),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)

    print(f"已保存：{output_path}")
    print(f"mode={mode}")
    print(f"imu_raw={arrays['imu_raw'].shape}, imu_normalized={imu_normalized.shape}")
    print(
        f"commands={arrays['commands'].shape}, "
        f"commands_normalized={commands_normalized.shape}"
    )
    print(f"fitted_gait_ids={np.flatnonzero(fitted_gait_mask).tolist()}")
    print(f"commands_normalized min={commands_normalized.min(axis=0)}")
    print(f"commands_normalized max={commands_normalized.max(axis=0)}")
    print(f"quantiles=({quantile_low}, {quantile_high})")
    for gait_name, gait_id_value, fit_count, q_low, q_high in summaries:
        print(f"{gait_name} (gait_id={gait_id_value}, fit_samples={fit_count})")
        print(
            f"  normalized quantile {quantile_low:g}: "
            f"{np.array2string(q_low, precision=4)}"
        )
        print(
            f"  normalized quantile {quantile_high:g}: "
            f"{np.array2string(q_high, precision=4)}"
        )
    return output_path


if __name__ == "__main__":
    normalize_dataset(parse_args())
