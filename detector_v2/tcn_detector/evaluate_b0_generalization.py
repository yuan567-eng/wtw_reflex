"""评估 TCN-B0 独立泛化测试：复杂正常运动误报与未见力度扰动检出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = (
    (0, "complex_normal"),
    (1, "weak_20_35N"),
    (2, "boundary_35_45N"),
    (3, "strong_80_110N"),
)
REQUIRED = (
    "probability", "detected", "threshold", "variant", "stride", "timestamp_us",
    "detector_target", "loss_mask", "episode_id", "collection_episode_index",
    "disturbance_force_y", "scenario_id", "scenario_names",
)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 TCN-B0 的独立五分钟泛化测试")
    parser.add_argument(
        "--prediction", type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b0_generalization_5min.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/tcn_b0_generalization_5min.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def contiguous_regions(mask: np.ndarray):
    """给出同一 episode 内连续 True 的闭区间，用于合并报警事件。"""
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    return [
        (int(positions[start]), int(positions[end - 1]))
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(positions)])
    ]


def alarm_events(detected, valid, episode_id, selector):
    """在 selector 选中的窗口中，以 episode 为界合并连续报警。"""
    result = 0
    for episode in np.unique(episode_id[selector]):
        locations = np.flatnonzero(selector & (episode_id == episode))
        result += len(contiguous_regions(detected[locations] & valid[locations]))
    return result


def load(path: Path):
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{path}")
    with np.load(path, allow_pickle=False) as raw:
        missing = [name for name in REQUIRED if name not in raw.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        arrays = {name: np.asarray(raw[name]) for name in REQUIRED}
    count = len(arrays["probability"])
    for name in REQUIRED:
        if name not in ("threshold", "variant", "stride", "scenario_names") and arrays[name].shape != (count,):
            raise ValueError(f"字段 {name} 的长度错误：{arrays[name].shape}")
    if str(arrays["variant"].item()) != "b0":
        raise ValueError("本脚本只用于 TCN-B0 预测文件")
    return path, arrays


def event_detection(arrays, scenario: int):
    """每个有施力 episode 中，target 区间至少一次报警即视为事件检出。"""
    target = arrays["detector_target"].astype(bool)
    detected = arrays["detected"].astype(bool)
    valid = arrays["loss_mask"].astype(bool)
    episode_id = arrays["episode_id"]
    scenario_id = arrays["scenario_id"]
    selected_episodes = np.unique(episode_id[scenario_id == scenario])
    events = []
    for episode in selected_episodes:
        locations = np.flatnonzero(episode_id == episode)
        region = locations[target[locations]]
        if len(region) == 0:
            raise ValueError(f"scenario={scenario} 的 episode={episode} 没有真实施力标签")
        events.append(bool(np.any(detected[region] & valid[region])))
    return {"events": len(events), "detected": int(sum(events)), "event_recall": sum(events) / max(len(events), 1)}


def main():
    args = parse_args()
    prediction_path, arrays = load(args.prediction)
    detected = arrays["detected"].astype(bool)
    valid = arrays["loss_mask"].astype(bool)
    target = arrays["detector_target"].astype(bool)
    scenario_id = arrays["scenario_id"].astype(np.int8)
    episode_id = arrays["episode_id"].astype(np.int64)

    normal = scenario_id == 0
    normal_valid = normal & valid
    normal_detected = normal_valid & detected
    # episode 间包含 warmup 造成的时间戳空档，必须逐 episode 累加有效窗口时长。
    detector_dt = float(arrays["stride"]) / 200.0
    normal_seconds = 0.0
    for episode in np.unique(episode_id[normal_valid]):
        normal_seconds += float(np.sum(normal_valid & (episode_id == episode))) * detector_dt
    normal_metrics = {
        "valid_windows": int(normal_valid.sum()),
        "detected_windows": int(normal_detected.sum()),
        "alarm_events": alarm_events(detected, valid, episode_id, normal),
        "duration_seconds": float(normal_seconds),
        "false_alarm_events_per_100s": alarm_events(detected, valid, episode_id, normal) / normal_seconds * 100.0,
    }

    stage = {name: event_detection(arrays, scenario) for scenario, name in SCENARIOS if scenario != 0}
    all_valid = valid
    all_target = target[all_valid]
    all_predicted = detected[all_valid]
    tp = int(np.sum(all_target & all_predicted)); fp = int(np.sum(~all_target & all_predicted))
    tn = int(np.sum(~all_target & ~all_predicted)); fn = int(np.sum(all_target & ~all_predicted))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    result = {
        "description": "独立泛化测试；模型与阈值均在采集前固定，结果不可用于再次调参。",
        "prediction_file": str(prediction_path),
        "threshold": float(arrays["threshold"]),
        "normal_motion": normal_metrics,
        "unseen_force_event_detection": stage,
        "all_valid_window_metrics": {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"threshold={result['threshold']:.2f}")
    print(f"normal: windows={normal_metrics['detected_windows']}/{normal_metrics['valid_windows']}, "
          f"false_alarm_events={normal_metrics['alarm_events']}, duration={normal_seconds:.1f}s")
    for name, metrics in stage.items():
        print(f"{name}: event_detect={metrics['detected']}/{metrics['events']}")
    all_metrics = result["all_valid_window_metrics"]
    print(f"all: F1={all_metrics['f1']:.4f}, P={precision:.4f}, R={recall:.4f}, FP={fp}")
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
