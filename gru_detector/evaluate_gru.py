"""在完整验证 episode 上评估 GRU 的连续扰动概率。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPTS = PROJECT_ROOT / "dataset/scripts"
if str(DATASET_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPTS))

from window_dataset import DEFAULT_INPUTS, DetectorWindowDataset
from gru import DisturbanceGRU


DEFAULT_CHECKPOINT = PROJECT_ROOT / "gru_detector/checkpoints/trotting_gru.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "gru_detector/evaluation"
IMU_SAMPLE_RATE_HZ = 200.0
DETECTOR_PERIOD_SECONDS = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description="评估完整 episode 上的 GRU Pd(t)")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-plots", type=int, default=6)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.max_plots < 0:
        parser.error("--max-plots 不能小于 0")
    return args


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return torch.device(requested)


def count_rising_edges(binary_values):
    """统计连续 True 区间数量，而不是把连续误报帧重复计数。"""
    binary_values = np.asarray(binary_values, dtype=bool)
    if len(binary_values) == 0:
        return 0
    return int(binary_values[0]) + int(np.count_nonzero(binary_values[1:] & ~binary_values[:-1]))


def calculate_window_metrics(probabilities, targets, masks, threshold):
    valid = masks.astype(bool)
    predictions = probabilities >= threshold
    targets = targets.astype(bool)
    true_positive = int(np.sum(predictions & targets & valid))
    false_positive = int(np.sum(predictions & ~targets & valid))
    true_negative = int(np.sum(~predictions & ~targets & valid))
    false_negative = int(np.sum(~predictions & targets & valid))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def infer_validation_windows(dataset, model, validation_episode_ids, batch_size, device):
    validation_episode_ids = set(int(value) for value in validation_episode_ids)
    indices = []
    for index, window in enumerate(dataset.windows):
        source = dataset.sources[window.source_index]
        episode_id = int(source["episode_id"][window.end])
        if episode_id in validation_episode_ids:
            indices.append(index)
    if not indices:
        raise ValueError("没有找到 checkpoint 中记录的验证 episode")

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    records = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            probabilities = torch.sigmoid(model(features)).cpu().numpy()
            episode_ids = batch["episode_id"].numpy()
            end_indices = batch["end_index"].numpy()
            targets = batch["target"].numpy()
            masks = batch["loss_mask"].numpy()
            for episode_id, end_index, probability, target, mask in zip(
                episode_ids, end_indices, probabilities, targets, masks
            ):
                records[int(episode_id)].append(
                    (int(end_index), float(probability), int(target), bool(mask))
                )
    for episode_id in records:
        records[episode_id].sort(key=lambda item: item[0])
    missing = validation_episode_ids.difference(records)
    if missing:
        raise ValueError(f"这些验证 episode 没有生成窗口：{sorted(missing)}")
    return records


def locate_episode_source(dataset, episode_id):
    for source in dataset.sources:
        positions = np.flatnonzero(source["episode_id"] == episode_id)
        if len(positions):
            return source, int(positions[0]), int(positions[-1]) + 1
    raise KeyError(f"找不到 episode_id={episode_id}")


def analyze_episode(dataset, episode_id, rows, threshold):
    source, episode_start, episode_end = locate_episode_source(dataset, episode_id)
    end_indices = np.array([row[0] for row in rows], dtype=np.int64)
    probabilities = np.array([row[1] for row in rows], dtype=np.float32)
    targets = np.array([row[2] for row in rows], dtype=np.int8)
    masks = np.array([row[3] for row in rows], dtype=np.bool_)
    predictions = probabilities >= threshold

    false_alarm_flags = predictions & (targets == 0) & masks
    # 被 mask 的恢复期切断连续报警，避免把恢复期两侧错误合并成同一次误报。
    false_alarm_events = 0
    segment_start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or not masks[index]:
            false_alarm_events += count_rising_edges(false_alarm_flags[segment_start:index])
            segment_start = index + 1

    full_targets = source["detector_target"][episode_start:episode_end].astype(bool)
    full_masks = source["loss_mask"][episode_start:episode_end].astype(bool)
    active_positions = np.flatnonzero(full_targets)
    event_present = len(active_positions) > 0
    event_detected_during_force = False
    event_detected_by_recovery_end = False
    detection_delay_ms = None
    if event_present:
        onset_index = episode_start + int(active_positions[0])
        force_end_index = episode_start + int(active_positions[-1])
        ignored_after_force = np.flatnonzero(
            ~full_masks[int(active_positions[-1]) + 1 :]
        )
        if len(ignored_after_force):
            recovery_end_index = (
                force_end_index + 1 + int(ignored_after_force[-1]) + 1
            )
        else:
            recovery_end_index = force_end_index
        after_onset = end_indices >= onset_index
        during_force = after_onset & (end_indices <= force_end_index) & predictions
        by_recovery_end = after_onset & (end_indices <= recovery_end_index) & predictions
        event_detected_during_force = bool(np.any(during_force))
        event_detected_by_recovery_end = bool(np.any(by_recovery_end))
        if event_detected_by_recovery_end:
            first_detection_index = int(end_indices[np.flatnonzero(by_recovery_end)[0]])
            detection_delay_ms = (
                (first_detection_index - onset_index) / IMU_SAMPLE_RATE_HZ * 1000.0
            )

    return {
        "episode_id": episode_id,
        "episode_start": episode_start,
        "end_indices": end_indices,
        "time_seconds": (end_indices - episode_start) / IMU_SAMPLE_RATE_HZ,
        "probabilities": probabilities,
        "targets": targets,
        "masks": masks,
        "false_alarm_events": false_alarm_events,
        "negative_valid_windows": int(np.sum((targets == 0) & masks)),
        "event_present": event_present,
        "event_detected_during_force": event_detected_during_force,
        "event_detected_by_recovery_end": event_detected_by_recovery_end,
        "detection_delay_ms": detection_delay_ms,
    }


def plot_episode(result, threshold, output_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    time_seconds = result["time_seconds"]
    probabilities = result["probabilities"]
    targets = result["targets"].astype(bool)
    ignored = ~result["masks"].astype(bool)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(time_seconds, probabilities, color="#2563eb", linewidth=1.4, label="Pd(t)")
    ax.axhline(threshold, color="#111827", linestyle="--", linewidth=1.0, label="threshold")
    if np.any(targets):
        ax.fill_between(
            time_seconds, 0, 1, where=targets, color="#ef4444", alpha=0.18,
            transform=ax.get_xaxis_transform(), label="disturbance",
        )
    if np.any(ignored):
        ax.fill_between(
            time_seconds, 0, 1, where=ignored, color="#6b7280", alpha=0.14,
            transform=ax.get_xaxis_transform(), label="ignored recovery",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Disturbance probability")
    ax.set_title(f"Validation episode {result['episode_id']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate(args):
    device = select_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "gru":
        raise ValueError("该 checkpoint 不是 gru_detector 训练得到的 GRU 模型")
    dataset = DetectorWindowDataset(
        args.inputs,
        checkpoint["window_size"],
        checkpoint["stride"],
        feature_mode=checkpoint.get("feature_mode", "normalized"),
    )
    model = DisturbanceGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint["threshold"])
    records = infer_validation_windows(
        dataset,
        model,
        checkpoint["val_episode_ids"],
        args.batch_size,
        device,
    )
    episode_results = [
        analyze_episode(dataset, episode_id, rows, threshold)
        for episode_id, rows in sorted(records.items())
    ]

    probabilities = np.concatenate([result["probabilities"] for result in episode_results])
    targets = np.concatenate([result["targets"] for result in episode_results])
    masks = np.concatenate([result["masks"] for result in episode_results])
    window_metrics = calculate_window_metrics(probabilities, targets, masks, threshold)

    event_results = [result for result in episode_results if result["event_present"]]
    detected_during_force = sum(result["event_detected_during_force"] for result in event_results)
    detected_by_recovery = sum(
        result["event_detected_by_recovery_end"] for result in event_results
    )
    delays = [
        result["detection_delay_ms"]
        for result in event_results
        if result["detection_delay_ms"] is not None
    ]
    false_alarm_events = sum(result["false_alarm_events"] for result in episode_results)
    negative_minutes = (
        sum(result["negative_valid_windows"] for result in episode_results)
        * DETECTOR_PERIOD_SECONDS
        / 60.0
    )
    event_metrics = {
        "disturbance_episodes": len(event_results),
        "detected_during_force": int(detected_during_force),
        "detected_by_recovery_end": int(detected_by_recovery),
        "event_recall_during_force": detected_during_force / max(len(event_results), 1),
        "event_recall_by_recovery_end": detected_by_recovery / max(len(event_results), 1),
        "detection_delay_ms_mean": float(np.mean(delays)) if delays else None,
        "detection_delay_ms_median": float(np.median(delays)) if delays else None,
        "detection_delay_ms_max": float(np.max(delays)) if delays else None,
        "false_alarm_events": int(false_alarm_events),
        "negative_duration_minutes": negative_minutes,
        "false_alarm_events_per_minute": false_alarm_events / max(negative_minutes, 1e-12),
    }

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    abnormal_results = [result for result in episode_results if result["event_present"]]
    for result in abnormal_results[: args.max_plots]:
        plot_episode(
            result,
            threshold,
            output_dir / f"episode_{result['episode_id']}_pd.png",
        )

    summary = {
        "model_type": "gru",
        "feature_mode": checkpoint.get("feature_mode", "normalized"),
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "validation_episodes": len(episode_results),
        "window_metrics": window_metrics,
        "event_metrics": event_metrics,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"device={device}, threshold={threshold:.3f}")
    print(f"validation_episodes={len(episode_results)}")
    print(
        "window metrics: "
        f"P={window_metrics['precision']:.4f}, R={window_metrics['recall']:.4f}, "
        f"F1={window_metrics['f1']:.4f}, FPR={window_metrics['false_positive_rate']:.6f}, "
        f"TP={window_metrics['tp']}, FP={window_metrics['fp']}, "
        f"TN={window_metrics['tn']}, FN={window_metrics['fn']}"
    )
    print(
        "event metrics: "
        f"during_force={detected_during_force}/{len(event_results)}, "
        f"by_recovery_end={detected_by_recovery}/{len(event_results)}, "
        f"delay_mean={event_metrics['detection_delay_ms_mean']}ms, "
        f"delay_median={event_metrics['detection_delay_ms_median']}ms, "
        f"delay_max={event_metrics['detection_delay_ms_max']}ms"
    )
    print(
        f"false_alarm_events={false_alarm_events}, "
        f"negative_minutes={negative_minutes:.3f}, "
        f"false_alarm_events_per_minute={event_metrics['false_alarm_events_per_minute']:.3f}"
    )
    print(f"评估摘要：{summary_path}")
    print(f"概率曲线目录：{output_dir}")
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
