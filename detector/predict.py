"""使用训练好的 Detector 对一个统一格式数据集执行离线预测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPTS = PROJECT_ROOT / "dataset/scripts"
if str(DATASET_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPTS))

from plot import find_episode_slices, plot_episode, resolve_input
from tcn import DisturbanceTCN


DEFAULT_CHECKPOINT = PROJECT_ROOT / "detector/checkpoints/trotting_tcn.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "detector/predictions"


def parse_args():
    parser = argparse.ArgumentParser(description="对一个 .npz 数据集执行 Detector 预测")
    parser.add_argument("input", type=Path, help="统一格式 .npz 文件或包含它的目录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None, help="预测结果 .npz 路径")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选检测阈值；默认使用 checkpoint 中保存的阈值",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--plot", action="store_true", help="同时生成 IMU 与检测叠加图")
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--plot-parts", type=int, default=1, help="沿时间轴平均拆成几张叠加图")
    parser.add_argument(
        "--plot-field",
        choices=("imu_raw", "imu_normalized"),
        default="imu_raw",
    )
    parser.add_argument("--show", action="store_true", help="显示叠加图窗口")
    parser.add_argument(
        "--phase-episodes",
        type=int,
        nargs=3,
        default=None,
        metavar=("BASE", "EXPANDED", "REDUCED"),
        help="按episode顺序分别统计原范围、扩大50%%、缩小50%%三阶段指标",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.plot_parts <= 0:
        parser.error("--plot-parts 必须大于0")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold 必须位于 (0,1)")
    if args.phase_episodes is not None and any(count <= 0 for count in args.phase_episodes):
        parser.error("--phase-episodes 的三个数量必须都大于0")
    return args


def resolve_output(path, default_path):
    if path is None:
        return default_path
    return path if path.is_absolute() else PROJECT_ROOT / path


def count_alarm_events(predictions, episode_ids):
    event_count = 0
    for episode_id in np.unique(episode_ids):
        values = predictions[episode_ids == episode_id].astype(bool)
        if len(values):
            event_count += int(values[0])
            event_count += int(np.count_nonzero(values[1:] & ~values[:-1]))
    return event_count


def print_labeled_metrics(predictions, targets, masks, label="整体"):
    valid = masks.astype(bool)
    predictions = predictions.astype(bool)
    targets = targets.astype(bool)
    tp = int(np.sum(predictions & targets & valid))
    fp = int(np.sum(predictions & ~targets & valid))
    tn = int(np.sum(~predictions & ~targets & valid))
    fn = int(np.sum(~predictions & targets & valid))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (recall + specificity) / 2
    print(
        f"{label}评估：Accuracy={accuracy:.4f}, "
        f"BalancedAccuracy={balanced_accuracy:.4f}, "
        f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
        f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
    )


def print_phase_metrics(predictions, targets, masks, prediction_episode_id, phase_counts):
    """按episode首次出现顺序统计三种扰动力范围的独立指标。"""
    _, first_indices = np.unique(prediction_episode_id, return_index=True)
    ordered_episode_ids = prediction_episode_id[np.sort(first_indices)]
    if sum(phase_counts) != len(ordered_episode_ids):
        raise ValueError(
            "--phase-episodes 数量之和必须等于数据中的episode数量："
            f"得到 {sum(phase_counts)}，数据为 {len(ordered_episode_ids)}"
        )

    phase_names = ("原范围", "扩大50%", "缩小50%")
    start = 0
    for phase_name, count in zip(phase_names, phase_counts):
        phase_ids = ordered_episode_ids[start : start + count]
        phase_mask = np.isin(prediction_episode_id, phase_ids)
        print_labeled_metrics(
            predictions[phase_mask],
            targets[phase_mask],
            masks[phase_mask],
            label=phase_name,
        )
        start += count


def run_detector(input_path, episode_slices, checkpoint_path, requested_device, batch_size):
    """使用 checkpoint 中的正常参数构造窗口并执行50 Hz推理。"""
    checkpoint_path = (
        checkpoint_path
        if checkpoint_path.is_absolute()
        else PROJECT_ROOT / checkpoint_path
    )
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    with np.load(input_path, allow_pickle=False) as data:
        required = ("imu_raw", "commands", "gait_id")
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ValueError(f"输入数据缺少 Detector 字段：{missing}")
        imu_raw = np.asarray(data["imu_raw"], dtype=np.float32)
        commands = np.asarray(data["commands"], dtype=np.float32)
        gait_id = np.asarray(data["gait_id"], dtype=np.int64)

    feature_mode = checkpoint.get("feature_mode", "normalized")
    if feature_mode == "normalized":
        stats = checkpoint.get("normalization")
        if stats is None:
            raise ValueError("normalized checkpoint 缺少归一化参数")
        center = np.asarray(stats["center"].cpu(), dtype=np.float32)
        scale = np.asarray(stats["scale"].cpu(), dtype=np.float32)
        fitted_gait_mask = np.asarray(stats["fitted_gait_mask"].cpu(), dtype=bool)
        command_center = np.asarray(stats["command_center"].cpu(), dtype=np.float32)
        command_scale = np.asarray(stats["command_scale"].cpu(), dtype=np.float32)
        epsilon = float(stats["epsilon"].cpu())
        observed_gait_ids = np.unique(gait_id)
        if np.any(observed_gait_ids < 0) or np.any(
            observed_gait_ids >= len(fitted_gait_mask)
        ):
            raise ValueError(f"gait_id 超出 checkpoint 范围：{observed_gait_ids.tolist()}")
        if not np.all(fitted_gait_mask[observed_gait_ids]):
            raise ValueError(f"checkpoint 未拟合 gait_id：{observed_gait_ids.tolist()}")
        imu_features = (imu_raw - center[gait_id]) / (scale[gait_id] + epsilon)
        command_features = (commands - command_center) / (command_scale + epsilon)
    elif feature_mode == "raw":
        imu_features = imu_raw
        command_features = commands
    else:
        raise ValueError(f"checkpoint 的 feature_mode 无效：{feature_mode}")

    gait_one_hot = np.eye(4, dtype=np.float32)[gait_id]
    features = np.concatenate(
        (imu_features, command_features, gait_one_hot), axis=1
    ).astype(np.float32, copy=False)

    model = DisturbanceTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    window_size = int(checkpoint["window_size"])
    stride = int(checkpoint["stride"])
    end_indices = []
    probabilities = []
    pending_windows = []
    pending_ends = []

    def infer_pending():
        if not pending_windows:
            return
        batch = torch.from_numpy(np.stack(pending_windows)).to(device)
        with torch.no_grad():
            batch_probabilities = torch.sigmoid(model(batch)).cpu().numpy()
        end_indices.extend(pending_ends)
        probabilities.extend(batch_probabilities.tolist())
        pending_windows.clear()
        pending_ends.clear()

    for episode_start, episode_end in episode_slices:
        last_start = episode_end - window_size
        if last_start < episode_start:
            continue
        for start in range(episode_start, last_start + 1, stride):
            end = start + window_size - 1
            pending_windows.append(features[start : end + 1].T.copy())
            pending_ends.append(end)
            if len(pending_windows) >= batch_size:
                infer_pending()
    infer_pending()
    if not end_indices:
        raise ValueError("数据长度不足，无法生成 Detector 窗口")
    return {
        "end_indices": np.asarray(end_indices, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "threshold": float(checkpoint["threshold"]),
        "window_size": window_size,
        "checkpoint": checkpoint_path,
        "device": device,
        "feature_mode": feature_mode,
    }


def predict(args):
    input_path = resolve_input(args.input)
    with np.load(input_path, allow_pickle=False) as data:
        if "timestamp_us" not in data.files:
            raise ValueError("输入数据缺少 timestamp_us")
        timestamp_us = np.asarray(data["timestamp_us"], dtype=np.int64)
        sample_count = len(timestamp_us)
        episode_id = (
            np.asarray(data["episode_id"], dtype=np.int64)
            if "episode_id" in data.files
            else np.zeros(sample_count, dtype=np.int64)
        )
        detector_target = (
            np.asarray(data["detector_target"], dtype=np.int8)
            if "detector_target" in data.files
            else None
        )
        loss_mask = (
            np.asarray(data["loss_mask"], dtype=np.bool_)
            if "loss_mask" in data.files
            else None
        )

    if episode_id.shape != (sample_count,):
        raise ValueError("episode_id 必须为 [T]")
    episode_slices = find_episode_slices(sample_count, {"episode_id": episode_id})
    detector_result = run_detector(
        input_path,
        episode_slices,
        args.checkpoint,
        args.device,
        args.batch_size,
    )
    end_indices = detector_result["end_indices"]
    probabilities = detector_result["probabilities"]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(detector_result["threshold"])
    )
    predictions = (probabilities >= threshold).astype(np.int8)
    prediction_episode_id = episode_id[end_indices]

    output = {
        "timestamp_us": timestamp_us[end_indices],
        "window_start_index": end_indices - int(detector_result["window_size"]) + 1,
        "window_end_index": end_indices,
        "episode_id": prediction_episode_id,
        "disturbance_probability": probabilities,
        "detector_prediction": predictions,
        "threshold": np.asarray(threshold, dtype=np.float32),
        "feature_mode": np.asarray(detector_result["feature_mode"]),
    }
    if detector_target is not None:
        output["detector_target"] = detector_target[end_indices]
    if loss_mask is not None:
        output["loss_mask"] = loss_mask[end_indices]

    output_path = resolve_output(
        args.output,
        DEFAULT_OUTPUT_DIR / f"{input_path.stem}_predictions.npz",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)

    intervals = np.diff(output["timestamp_us"])
    positive_windows = int(predictions.sum())
    alarm_events = count_alarm_events(predictions, prediction_episode_id)
    print(f"输入数据：{input_path}")
    print(f"checkpoint：{detector_result['checkpoint']}")
    print(f"device：{detector_result['device']}")
    print(f"feature_mode：{detector_result['feature_mode']}")
    print(f"预测窗口：{len(probabilities)}，shape={probabilities.shape}")
    if len(intervals):
        positive_intervals = intervals[intervals > 0]
        if len(positive_intervals):
            rate = 1_000_000.0 / float(np.median(positive_intervals))
            print(f"预测频率：{rate:.1f} Hz")
    print(
        f"threshold={threshold:.3f}, detected_windows={positive_windows}, "
        f"alarm_events={alarm_events}, max_Pd={float(probabilities.max()):.6f}"
    )
    if detector_target is not None and loss_mask is not None:
        prediction_targets = detector_target[end_indices]
        prediction_masks = loss_mask[end_indices]
        print_labeled_metrics(
            predictions,
            prediction_targets,
            prediction_masks,
        )
        if args.phase_episodes is not None:
            print_phase_metrics(
                predictions,
                prediction_targets,
                prediction_masks,
                prediction_episode_id,
                args.phase_episodes,
            )
    elif args.phase_episodes is not None:
        raise ValueError("--phase-episodes 需要输入数据包含 detector_target 和 loss_mask")
    print(f"预测结果：{output_path}")

    if args.plot:
        plot_output = resolve_output(
            args.plot_output,
            DEFAULT_OUTPUT_DIR / f"{input_path.stem}_detector_overlay.png",
        )
        plot_episode(
            input_path,
            args.plot_field,
            plot_output,
            args.show,
            160,
            prediction_path=output_path,
            parts=args.plot_parts,
        )
        if args.plot_parts == 1:
            print(f"检测叠加图：{plot_output}")
        else:
            print(f"检测叠加图：已按时间轴拆分为 {args.plot_parts} 张")
    return output_path


if __name__ == "__main__":
    predict(parse_args())
