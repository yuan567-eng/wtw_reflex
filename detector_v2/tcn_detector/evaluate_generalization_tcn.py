"""对同一份泛化测试预测比较任意 TCN 变体，例如 B0 与 B3。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ((0, "complex_normal"), (1, "weak_20_35N"),
             (2, "boundary_35_45N"), (3, "strong_80_110N"))
REQUIRED = (
    "probability", "detected", "threshold", "variant", "stride", "window_start_index",
    "window_end_index", "timestamp_us", "detector_target", "loss_mask", "episode_id",
    "collection_episode_index", "disturbance_force_y", "scenario_id", "scenario_names",
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="比较 TCN 变体在独立泛化测试集上的表现")
    parser.add_argument("predictions", type=Path, nargs="+", help="同一份泛化测试的预测文件")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/tcn_b0_b3_generalization_5min.json",
    )
    return parser.parse_args()


def load(path: Path):
    path = resolve(path)
    with np.load(path, allow_pickle=False) as raw:
        missing = [name for name in REQUIRED if name not in raw.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        arrays = {name: np.asarray(raw[name]) for name in REQUIRED}
    length = len(arrays["probability"])
    for name in REQUIRED:
        if name not in ("threshold", "variant", "stride", "scenario_names") and arrays[name].shape != (length,):
            raise ValueError(f"{path} 的 {name} 长度错误")
    return path, arrays


def regions(mask: np.ndarray):
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []
    split = np.flatnonzero(np.diff(positions) != 1) + 1
    return [
        (int(positions[a]), int(positions[b - 1]))
        for a, b in zip(np.r_[0, split], np.r_[split, len(positions)])
    ]


def count_alarm_events(detected, valid, episode_id, selector):
    count = 0
    for episode in np.unique(episode_id[selector]):
        locations = np.flatnonzero(selector & (episode_id == episode))
        count += len(regions(detected[locations] & valid[locations]))
    return count


def event_metric(arrays, scenario_id: int):
    selected = arrays["scenario_id"] == scenario_id
    target = arrays["detector_target"].astype(bool)
    detected = arrays["detected"].astype(bool)
    valid = arrays["loss_mask"].astype(bool)
    episode_id = arrays["episode_id"]
    results = []
    for episode in np.unique(episode_id[selected]):
        indices = np.flatnonzero(episode_id == episode)
        force_indices = indices[target[indices]]
        if len(force_indices) == 0:
            raise ValueError(f"scenario={scenario_id} 含无真实施力标签的 episode")
        results.append(bool(np.any(detected[force_indices] & valid[force_indices])))
    return {"events": len(results), "detected": int(sum(results)),
            "event_recall": sum(results) / max(len(results), 1)}


def metrics(arrays):
    detected = arrays["detected"].astype(bool)
    valid = arrays["loss_mask"].astype(bool)
    target = arrays["detector_target"].astype(bool)
    scenario_id = arrays["scenario_id"]
    episode_id = arrays["episode_id"]
    selected = valid
    truth, prediction = target[selected], detected[selected]
    tp = int(np.sum(truth & prediction)); fp = int(np.sum(~truth & prediction))
    tn = int(np.sum(~truth & ~prediction)); fn = int(np.sum(truth & ~prediction))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)

    normal = scenario_id == 0
    normal_valid = normal & valid
    # 50 Hz 检测窗口有效时长，逐 episode 累加以排除 warmup 产生的时间戳间隙。
    normal_duration = float(normal_valid.sum()) * float(arrays["stride"]) / 200.0
    normal_alarm = count_alarm_events(detected, valid, episode_id, normal)
    return {
        "window_metrics": {
            "valid_windows": int(selected.sum()), "positive_windows": int(truth.sum()),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "normal_motion": {
            "valid_windows": int(normal_valid.sum()),
            "detected_windows": int(np.sum(detected & normal_valid)),
            "alarm_events": normal_alarm,
            "duration_seconds": normal_duration,
            "false_alarm_events_per_100s": normal_alarm / max(normal_duration, 1e-12) * 100.0,
        },
        "unseen_force_event_detection": {
            name: event_metric(arrays, scenario) for scenario, name in SCENARIOS if scenario != 0
        },
    }


def main():
    args = parse_args()
    loaded = [load(path) for path in args.predictions]
    reference_path, reference = loaded[0]
    match_fields = (
        "window_start_index", "window_end_index", "timestamp_us", "detector_target",
        "loss_mask", "episode_id", "collection_episode_index", "disturbance_force_y", "scenario_id",
    )
    rows = []
    for path, arrays in loaded:
        if any(not np.array_equal(reference[name], arrays[name]) for name in match_fields):
            raise ValueError(f"{path} 与 {reference_path} 并非同一份测试数据")
        row = {"variant": str(arrays["variant"].item()), "threshold": float(arrays["threshold"]),
               "prediction_file": str(path), **metrics(arrays)}
        rows.append(row)
    result = {
        "description": "模型和阈值均在采集此独立测试前固定；结果不可再用于调参。",
        "normal_rule": "complex_normal 的报警均为假报警；以连续检测窗口合并为一次报警事件。",
        "event_rule": "同一真实施力区间中至少一个有效窗口报警，即为该扰动事件检出。",
        "results": rows,
    }
    output = resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    for row in rows:
        window, normal, event = row["window_metrics"], row["normal_motion"], row["unseen_force_event_detection"]
        print(f"{row['variant']}: F1={window['f1']:.4f}, P={window['precision']:.4f}, R={window['recall']:.4f}, FP={window['confusion']['fp']}; "
              f"normal_false_alarm={normal['alarm_events']}/{normal['duration_seconds']:.1f}s; "
              f"weak={event['weak_20_35N']['detected']}/6, boundary={event['boundary_35_45N']['detected']}/6, strong={event['strong_80_110N']['detected']}/6")
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
