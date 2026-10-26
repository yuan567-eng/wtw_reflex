"""统一评估 TCN B0/B1/B2 在独立 5 分钟测试集上的窗口级与事件级性能。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b0_test_5min.npz",
    PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b1_test_5min.npz",
    PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b2_test_5min.npz",
)
STAGES = (
    ("original_40_80N", range(0, 6)),
    ("expanded_60_120N", range(6, 18)),
    ("weak_20_40N", range(18, 30)),
)
REQUIRED_FIELDS = (
    "probability", "detected", "threshold", "variant", "window_size", "stride",
    "window_start_index", "window_end_index", "timestamp_us", "detector_target",
    "loss_mask", "episode_id", "collection_episode_index", "disturbance_force_y",
)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 TCN V2 的 B0/B1/B2 预测文件")
    parser.add_argument("predictions", type=Path, nargs="*", default=list(DEFAULT_PREDICTIONS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/tcn_b0_b1_b2_test_5min.json",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_prediction(path: Path):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        arrays = {name: np.asarray(data[name]) for name in REQUIRED_FIELDS}
    length = len(arrays["probability"])
    for name in (
        "detected", "window_start_index", "window_end_index", "timestamp_us",
        "detector_target", "loss_mask", "episode_id", "collection_episode_index",
        "disturbance_force_y",
    ):
        if arrays[name].shape != (length,):
            raise ValueError(f"{path} 的 {name} shape 错误：{arrays[name].shape}")
    if not np.isfinite(arrays["probability"]).all() or not np.all((arrays["probability"] >= 0) & (arrays["probability"] <= 1)):
        raise ValueError(f"{path} 的 probability 必须是 [0,1] 内有限值")
    if not np.all(np.isin(arrays["detector_target"], (0, 1))):
        raise ValueError(f"{path} 的 detector_target 只能为 0 或 1")
    return path, arrays


def assert_same_test_set(reference, candidate, reference_path: Path, candidate_path: Path):
    """确保比较的确实是同一批窗口、标签和事件元数据。"""
    for name in (
        "window_start_index", "window_end_index", "timestamp_us", "detector_target",
        "loss_mask", "episode_id", "collection_episode_index", "disturbance_force_y",
    ):
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"{candidate_path} 与 {reference_path} 的 {name} 不一致，不能公平比较")


def window_metrics(arrays):
    valid = arrays["loss_mask"].astype(bool)
    target = arrays["detector_target"].astype(bool)
    predicted = arrays["detected"].astype(bool)
    target, predicted = target[valid], predicted[valid]
    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    tn = int(np.sum(~predicted & ~target))
    fn = int(np.sum(~predicted & target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "valid_windows": int(valid.sum()),
        "positive_windows": int(target.sum()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fp / max(fp + tn, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def contiguous_regions(mask: np.ndarray):
    """返回一个一维布尔序列中所有连续 True 的闭区间 [start,end]。"""
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(positions)]))
    return [(int(positions[start]), int(positions[end - 1])) for start, end in zip(starts, ends)]


def event_metrics(arrays):
    """一次施力区间内至少一个报警窗口，即计为该真实扰动事件检出。"""
    detected = arrays["detected"].astype(bool)
    valid = arrays["loss_mask"].astype(bool)
    target = arrays["detector_target"].astype(bool)
    episode_index = arrays["collection_episode_index"].astype(np.int64)
    episode_id = arrays["episode_id"].astype(np.int64)
    force_y = arrays["disturbance_force_y"].astype(np.float64)

    events = []
    all_alarm_events = []
    for current_episode in np.unique(episode_id):
        locations = np.flatnonzero(episode_id == current_episode)
        local_target = target[locations]
        local_alarm = detected[locations] & valid[locations]
        for start, end in contiguous_regions(local_target):
            global_region = locations[start : end + 1]
            event_episode_index = int(episode_index[global_region[0]])
            events.append(
                {
                    "episode_index": event_episode_index,
                    "force_y": float(force_y[global_region[0]]),
                    "detected": bool(np.any(detected[global_region] & valid[global_region])),
                }
            )
        for start, end in contiguous_regions(local_alarm):
            global_region = locations[start : end + 1]
            all_alarm_events.append(
                {
                    "episode_index": int(episode_index[global_region[0]]),
                    "overlaps_target": bool(np.any(target[global_region])),
                }
            )

    if len(events) != 30:
        raise ValueError(f"测试集应有 30 个真实扰动事件，实际为 {len(events)}")
    stage_metrics = {}
    for name, indices in STAGES:
        stage_events = [event for event in events if event["episode_index"] in indices]
        stage_metrics[name] = {
            "events": len(stage_events),
            "detected": sum(event["detected"] for event in stage_events),
            "event_recall": sum(event["detected"] for event in stage_events) / max(len(stage_events), 1),
        }
    detected_events = sum(event["detected"] for event in events)
    false_alarm_events = sum(not event["overlaps_target"] for event in all_alarm_events)
    return {
        "events": len(events),
        "detected": detected_events,
        "event_recall": detected_events / len(events),
        "alarm_events": len(all_alarm_events),
        "false_alarm_events": false_alarm_events,
        "stages": stage_metrics,
    }


def evaluate(args):
    loaded = [load_prediction(path) for path in args.predictions]
    if not loaded:
        raise ValueError("至少需要一个预测文件")
    reference_path, reference = loaded[0]
    results = []
    for path, arrays in loaded:
        assert_same_test_set(reference, arrays, reference_path, path)
        variant = str(arrays["variant"].item())
        results.append(
            {
                "variant": variant,
                "prediction_file": str(path),
                "threshold": float(arrays["threshold"]),
                "window_metrics": window_metrics(arrays),
                "event_metrics": event_metrics(arrays),
            }
        )
    result = {
        "description": "TCN V2 B0/B1/B2：相同训练/测试数据、相同窗口与阈值的独立 5 分钟测试比较。",
        "event_rule": "每次真实施力区间内，至少一个有效窗口报警，即算该事件检出。",
        "stage_rule": "按 collection_episode_index 分段：0-5 原范围、6-17 扩大范围、18-29 弱扰动。",
        "results": results,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    for item in results:
        window = item["window_metrics"]
        event = item["event_metrics"]
        print(
            f"{item['variant']}: F1={window['f1']:.4f}, P={window['precision']:.4f}, "
            f"R={window['recall']:.4f}, FP={window['confusion']['fp']}, "
            f"alarm_events={event['alarm_events']}, event_detect={event['detected']}/{event['events']}"
        )
    print(f"已保存：{output}")
    return output


if __name__ == "__main__":
    evaluate(parse_args())
