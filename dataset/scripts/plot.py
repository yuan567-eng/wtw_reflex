"""可视化 Detector 数据集中的六维 IMU 时序。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset/raw/abnormal_gait/normalization_data/episode_1791376876691668_normalized.npz"

CHANNEL_NAMES = ("a_x", "a_y", "a_z", "omega_x", "omega_y", "omega_z")
COMMAND_NAMES = ("vx_command", "vy_command", "yaw_rate_command", "gait_frequency")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制六维 IMU episode 时序图")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="输入 .npz 文件或包含 episode 的目录；默认选择 dataset/raw 下最新文件",
    )
    parser.add_argument(
        "--field",
        choices=("imu_raw", "imu_normalized"),
        default="imu_raw",
        help="绘制原始或归一化后的 IMU",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径")
    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help="可选的 predict.py 输出 .npz；提供后增加 Detector 检测面板",
    )
    parser.add_argument("--show", action="store_true", help="保存后打开交互窗口")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--parts", type=int, default=1, help="沿时间轴平均拆成几张图")
    args = parser.parse_args()
    if args.parts <= 0:
        parser.error("--parts 必须大于0")
    return args


def resolve_input(path: Path) -> Path:
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        path = candidate if candidate.exists() else path.resolve()
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"输入文件必须是 .npz：{path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到输入路径：{path}")

    episodes = [item for item in path.rglob("*.npz") if item.is_file()]
    if not episodes:
        raise FileNotFoundError(f"目录中没有 .npz 数据：{path}")
    return max(episodes, key=lambda item: item.stat().st_mtime)


def load_episode(path: Path, field: str):
    with np.load(path) as data:
        required = {"timestamp_us", field}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"数据缺少字段：{sorted(missing)}")

        timestamp_us = np.asarray(data["timestamp_us"], dtype=np.int64)
        imu = np.asarray(data[field], dtype=np.float32)
        optional = {
            name: np.asarray(data[name])
            for name in (
                "gait_id", "detector_target", "switching", "settling", "loss_mask",
                "episode_id", "commands", "commands_normalized",
            )
            if name in data.files
        }

    if timestamp_us.ndim != 1:
        raise ValueError(f"timestamp_us 应为 [T]，实际为 {timestamp_us.shape}")
    if imu.ndim != 2 or imu.shape[1] != 6:
        raise ValueError(f"{field} 应为 [T, 6]，实际为 {imu.shape}")
    if len(timestamp_us) != len(imu):
        raise ValueError("timestamp_us 与 IMU 的帧数不一致")
    if len(timestamp_us) == 0:
        raise ValueError("数据文件没有 IMU 样本")
    if not np.isfinite(imu).all():
        raise ValueError("IMU 中存在 NaN 或 Inf")
    for name, values in optional.items():
        expected_shape = (
            (len(timestamp_us), 4)
            if name in ("commands", "commands_normalized")
            else (len(timestamp_us),)
        )
        if values.shape != expected_shape:
            raise ValueError(f"{name} 应为 {expected_shape}，实际为 {values.shape}")
    for name in ("commands", "commands_normalized"):
        if name in optional and not np.isfinite(optional[name]).all():
            raise ValueError(f"{name} 中存在 NaN 或 Inf")
    return timestamp_us, imu, optional


def find_episode_slices(sample_count: int, optional):
    """根据 episode_id 找到连续区段；单 episode 文件视为一个区段。"""
    if "episode_id" not in optional:
        return [(0, sample_count)]

    episode_id = optional["episode_id"]
    starts = np.concatenate(([0], np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1))
    ends = np.concatenate((starts[1:], [sample_count]))

    # 同一个 ID 若分散在多个区段中，后续窗口边界语义会变得不明确，因此直接报错。
    segment_ids = episode_id[starts]
    if len(np.unique(segment_ids)) != len(segment_ids):
        raise ValueError("同一个 episode_id 出现在多个不连续区段中")
    return list(zip(starts.tolist(), ends.tolist()))


def build_time_axis(timestamp_us, episode_slices):
    """逐 episode 校验原始时间戳，并构造用于总数据集绘图的连续时间轴。"""
    time_s = np.empty(len(timestamp_us), dtype=np.float64)
    valid_intervals = []
    next_start_s = 0.0

    for start, end in episode_slices:
        segment_timestamp = timestamp_us[start:end]
        intervals = np.diff(segment_timestamp)
        if len(intervals) and np.any(intervals <= 0):
            raise ValueError(
                f"episode 区段 [{start}, {end}) 内的 timestamp_us 必须严格递增"
            )
        if len(intervals):
            valid_intervals.append(intervals)
            boundary_interval_s = float(np.median(intervals)) / 1_000_000.0
        elif valid_intervals:
            boundary_interval_s = float(np.median(np.concatenate(valid_intervals))) / 1_000_000.0
        else:
            boundary_interval_s = 0.0

        local_time_s = (segment_timestamp - segment_timestamp[0]) / 1_000_000.0
        time_s[start:end] = next_start_s + local_time_s
        next_start_s = time_s[end - 1] + boundary_interval_s

    if valid_intervals:
        sample_rate_hz = 1_000_000.0 / float(np.median(np.concatenate(valid_intervals)))
    else:
        sample_rate_hz = float("nan")
    return time_s, sample_rate_hz


def print_summary(path: Path, field: str, time_s, sample_rate_hz, imu, optional, episode_slices):
    duration_s = time_s[-1] - time_s[0]

    print(f"数据文件：{path}")
    print(f"字段：{field}")
    print(f"样本数：{len(imu)}")
    print(f"episode 数：{len(episode_slices)}")
    print(f"时长：{duration_s:.3f} s")
    print(f"采样率：{sample_rate_hz:.1f} Hz")
    if "gait_id" in optional:
        print(f"gait_id：{np.unique(optional['gait_id']).tolist()}")
    command_field = (
        "commands_normalized"
        if field == "imu_normalized" and "commands_normalized" in optional
        else "commands"
    )
    if command_field in optional:
        print(f"命令范围（{command_field}）：")
        for index, name in enumerate(COMMAND_NAMES):
            values = optional[command_field][:, index]
            print(f"  {name:<18} min={values.min():>8.4f}, max={values.max():>8.4f}")
    print("通道统计：")
    print("channel          mean          std          min          max")
    for index, name in enumerate(CHANNEL_NAMES):
        values = imu[:, index]
        print(
            f"{name:<10} {values.mean():>12.5f} {values.std():>12.5f} "
            f"{values.min():>12.5f} {values.max():>12.5f}"
        )


def add_state_background(ax, time_s, optional):
    states = (
        ("detector_target", "#ef4444", "disturbance"),
        ("switching", "#f59e0b", "switching"),
        ("settling", "#3b82f6", "settling"),
    )
    for key, color, label in states:
        if key in optional and np.any(optional[key].astype(bool)):
            ax.fill_between(
                time_s, 0, 1, where=optional[key].astype(bool),
                transform=ax.get_xaxis_transform(), color=color, alpha=0.12, label=label,
            )
    if "loss_mask" in optional:
        ignored = ~optional["loss_mask"].astype(bool)
        if np.any(ignored):
            ax.fill_between(
                time_s, 0, 1, where=ignored,
                transform=ax.get_xaxis_transform(), color="#6b7280", alpha=0.10, label="ignored",
            )


def load_prediction(path: Path, sample_count: int):
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "window_end_index",
            "disturbance_probability",
            "detector_prediction",
            "threshold",
        )
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ValueError(f"预测文件缺少字段：{missing}")
        result = {field: np.asarray(data[field]) for field in required}

    end_indices = result["window_end_index"].astype(np.int64, copy=False)
    probabilities = result["disturbance_probability"].astype(np.float32, copy=False)
    predictions = result["detector_prediction"].astype(np.int8, copy=False)
    if end_indices.ndim != 1:
        raise ValueError("window_end_index 必须为一维")
    if probabilities.shape != end_indices.shape or predictions.shape != end_indices.shape:
        raise ValueError("预测概率、检测结果和窗口索引 shape 必须一致")
    if len(end_indices) == 0 or np.any(end_indices < 0) or np.any(end_indices >= sample_count):
        raise ValueError("预测文件中的窗口索引为空或超出原始数据范围")
    if not np.isfinite(probabilities).all():
        raise ValueError("预测概率包含 NaN 或 Inf")
    return {
        "path": path,
        "end_indices": end_indices,
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold": float(result["threshold"]),
    }


def plot_episode(
    path: Path,
    field: str,
    output: Path,
    show: bool,
    dpi: int,
    prediction_path: Path | None = None,
    parts: int = 1,
):
    if parts <= 0:
        raise ValueError("parts 必须大于0")
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    timestamp_us, imu, optional = load_episode(path, field)
    episode_slices = find_episode_slices(len(timestamp_us), optional)
    time_s, sample_rate_hz = build_time_axis(timestamp_us, episode_slices)
    print_summary(path, field, time_s, sample_rate_hz, imu, optional, episode_slices)
    prediction = (
        load_prediction(prediction_path, len(timestamp_us))
        if prediction_path is not None
        else None
    )

    command_field = (
        "commands_normalized"
        if field == "imu_normalized" and "commands_normalized" in optional
        else "commands"
    )
    has_commands = command_field in optional
    row_count = 2 + int(has_commands) + int(prediction is not None)
    figure_height = 8 + 2 * int(has_commands) + 2.5 * int(prediction is not None)
    fig, axes = plt.subplots(row_count, 1, figsize=(13, figure_height), sharex=True)
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for index, color in zip(range(3), colors):
        for segment_index, (start, end) in enumerate(episode_slices):
            axes[0].plot(
                time_s[start:end], imu[start:end, index], color=color, linewidth=1.0,
                label=CHANNEL_NAMES[index] if segment_index == 0 else None,
            )
    for index, color in zip(range(3, 6), colors):
        for segment_index, (start, end) in enumerate(episode_slices):
            axes[1].plot(
                time_s[start:end], imu[start:end, index], color=color, linewidth=1.0,
                label=CHANNEL_NAMES[index] if segment_index == 0 else None,
            )

    if has_commands:
        command_colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
        for index, color in enumerate(command_colors):
            for segment_index, (start, end) in enumerate(episode_slices):
                axes[2].plot(
                    time_s[start:end], optional[command_field][start:end, index],
                    color=color, linewidth=1.1,
                    label=COMMAND_NAMES[index] if segment_index == 0 else None,
                )

    if prediction is not None:
        detector_axis = axes[3 if has_commands else 2]
        detector_time = time_s[prediction["end_indices"]]
        detector_axis.plot(
            detector_time,
            prediction["probabilities"],
            color="#2563eb",
            linewidth=1.4,
            label="Pd(t)",
        )
        detector_axis.step(
            detector_time,
            prediction["predictions"].astype(np.float32),
            where="post",
            color="#f97316",
            linewidth=1.0,
            alpha=0.85,
            label="detected",
        )
        detector_axis.axhline(
            prediction["threshold"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold={prediction['threshold']:.2f}",
        )
        detector_axis.set_ylim(-0.03, 1.03)
        detector_axis.set_ylabel("Detector")
        print(
            f"预测叠加：{prediction['path']}，windows={len(prediction['end_indices'])}，"
            f"detected={int(prediction['predictions'].sum())}，"
            f"max_Pd={float(prediction['probabilities'].max()):.6f}"
        )

    for ax in axes:
        add_state_background(ax, time_s, optional)
        for start, _ in episode_slices[1:]:
            ax.axvline(time_s[start], color="#111827", linewidth=0.8, alpha=0.35)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3)
    title_suffix = " + Detector" if prediction is not None else ""
    axes[0].set_ylabel("Acceleration (m/s²)" if field == "imu_raw" else "Normalized acceleration")
    axes[1].set_ylabel("Angular velocity (rad/s)" if field == "imu_raw" else "Normalized angular velocity")
    if has_commands:
        axes[2].set_ylabel(
            "Normalized command" if command_field == "commands_normalized" else "Motion command"
        )
    axes[-1].set_xlabel("Time (s)")
    output.parent.mkdir(parents=True, exist_ok=True)
    time_edges = np.linspace(float(time_s[0]), float(time_s[-1]), parts + 1)
    for part_index in range(parts):
        part_start = time_edges[part_index]
        part_end = time_edges[part_index + 1]
        axes[-1].set_xlim(part_start, part_end)
        if parts == 1:
            part_output = output
            part_title = ""
        else:
            part_output = output.with_name(
                f"{output.stem}_part{part_index + 1:02d}of{parts:02d}{output.suffix}"
            )
            part_title = f" | part {part_index + 1}/{parts}"
        axes[0].set_title(f"IMU episode: {path.name}{title_suffix}{part_title}")
        fig.tight_layout()
        fig.savefig(part_output, dpi=dpi, bbox_inches="tight")
        print(
            f"图像已保存：{part_output} "
            f"[{part_start:.1f}s, {part_end:.1f}s]"
        )
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    input_path = resolve_input(args.input)
    output_path = args.output
    if output_path is None:
        suffix = "_detector" if args.prediction is not None else ""
        output_path = input_path.with_name(f"{input_path.stem}_{args.field}{suffix}.png")
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    plot_episode(
        input_path,
        args.field,
        output_path,
        args.show,
        args.dpi,
        prediction_path=args.prediction,
        parts=args.parts,
    )


if __name__ == "__main__":
    main()
