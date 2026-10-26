"""定位 B3 独立泛化测试中的复杂正常假报警与弱/临界扰动漏检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="分析 TCN-B3 泛化测试错误事件")
    parser.add_argument("--raw", type=Path,
                        default=PROJECT_ROOT / "detector_v2/dataset/raw/test_b3_generalization_5min.npz")
    parser.add_argument("--prediction", type=Path,
                        default=PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b3_on_generalization_5min.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation")
    return parser.parse_args()


def resolve(path: Path):
    return path if path.is_absolute() else PROJECT_ROOT / path


def regions(mask: np.ndarray):
    locations = np.flatnonzero(mask)
    if len(locations) == 0:
        return []
    breaks = np.flatnonzero(np.diff(locations) != 1) + 1
    return [(int(locations[a]), int(locations[b - 1]))
            for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(locations)])]


def local_time_us(raw, episode_index: int):
    locations = np.flatnonzero(raw["collection_episode_index"] == episode_index)
    return raw["timestamp_us"][locations[0]], locations


def command_summary(raw, raw_index: int):
    """返回报警峰值命令，以及相对 0.5 秒前的命令改变量。"""
    command = raw["commands"][raw_index].astype(float)
    previous = raw["commands"][max(0, raw_index - 100)].astype(float)
    return {
        "command_at_peak": command.round(4).tolist(),
        "command_delta_0p5s": (command - previous).round(4).tolist(),
        "command_delta_norm_0p5s": float(np.linalg.norm(command - previous)),
    }


def plot_false_alarms(pred, raw, false_events, output: Path):
    if not false_events:
        return
    columns = 2
    rows = int(np.ceil(len(false_events) / columns))
    figure, axes = plt.subplots(rows * 2, columns, figsize=(14, 3.8 * rows), squeeze=False)
    threshold = float(pred["threshold"])
    for event_index, event in enumerate(false_events):
        column, row = event_index % columns, (event_index // columns) * 2
        probability_axis, command_axis = axes[row, column], axes[row + 1, column]
        episode = event["episode_index"]
        selected = pred["collection_episode_index"] == episode
        locations = np.flatnonzero(selected)
        time = (pred["timestamp_us"][locations] - pred["timestamp_us"][locations[0]]) / 1e6
        probability_axis.plot(time, pred["probability"][locations], color="#2166ac", lw=1.2, label="Pd")
        probability_axis.axhline(threshold, color="#b2182b", ls="--", lw=1, label="threshold")
        probability_axis.axvspan(event["start_s"], event["end_s"], color="#ef8a62", alpha=.25, label="false alarm")
        probability_axis.set_ylim(-.03, 1.03); probability_axis.set_title(
            f"normal episode {episode}: false alarm {event_index + 1}, peak={event['peak_pd']:.3f}")
        probability_axis.set_ylabel("Pd"); probability_axis.grid(alpha=.25); probability_axis.legend(loc="upper right", fontsize=8)
        raw_start, raw_locations = local_time_us(raw, episode)
        raw_time = (raw["timestamp_us"][raw_locations] - raw_start) / 1e6
        commands = raw["commands"][raw_locations]
        command_axis.plot(raw_time, commands[:, 0], label="vx")
        command_axis.plot(raw_time, commands[:, 1], label="vy")
        command_axis.plot(raw_time, commands[:, 2], label="yaw")
        command_axis.set_xlabel("time in episode (s)"); command_axis.set_ylabel("command")
        command_axis.grid(alpha=.25); command_axis.legend(loc="upper right", fontsize=8)
    for index in range(len(false_events), rows * columns):
        axes[(index // columns) * 2, index % columns].axis("off")
        axes[(index // columns) * 2 + 1, index % columns].axis("off")
    figure.tight_layout(); figure.savefig(output, dpi=160); plt.close(figure)


def main():
    args = parse_args(); raw_path, pred_path = resolve(args.raw), resolve(args.prediction)
    with np.load(raw_path, allow_pickle=False) as source:
        raw = {name: np.asarray(source[name]) for name in ("timestamp_us", "commands", "settling", "collection_episode_index")}
    with np.load(pred_path, allow_pickle=False) as source:
        required = ("probability", "detected", "threshold", "window_end_index", "timestamp_us", "loss_mask",
                    "detector_target", "collection_episode_index", "scenario_id")
        missing = [name for name in required if name not in source.files]
        if missing: raise ValueError(f"预测文件缺少：{missing}")
        pred = {name: np.asarray(source[name]) for name in required}
    if not np.array_equal(pred["timestamp_us"], raw["timestamp_us"][pred["window_end_index"]]):
        raise ValueError("raw 与 prediction 时间戳不对齐")

    detected, valid = pred["detected"].astype(bool), pred["loss_mask"].astype(bool)
    normal = pred["scenario_id"] == 0
    false_events = []
    for episode in np.unique(pred["collection_episode_index"][normal]):
        locations = np.flatnonzero(normal & (pred["collection_episode_index"] == episode))
        for start, end in regions(detected[locations] & valid[locations]):
            global_indices = locations[start:end + 1]
            peak = global_indices[np.argmax(pred["probability"][global_indices])]
            timestamp0 = pred["timestamp_us"][locations[0]]
            raw_index = int(pred["window_end_index"][peak])
            false_events.append({
                "episode_index": int(episode),
                "start_s": float((pred["timestamp_us"][global_indices[0]] - timestamp0) / 1e6),
                "end_s": float((pred["timestamp_us"][global_indices[-1]] - timestamp0) / 1e6),
                "peak_s": float((pred["timestamp_us"][peak] - timestamp0) / 1e6),
                "peak_pd": float(pred["probability"][peak]),
                **command_summary(raw, raw_index),
            })

    missed_events = []
    for scenario in (1, 2):
        for episode in np.unique(pred["collection_episode_index"][pred["scenario_id"] == scenario]):
            locations = np.flatnonzero(pred["collection_episode_index"] == episode)
            target_locations = locations[pred["detector_target"][locations].astype(bool)]
            if np.any(detected[target_locations] & valid[target_locations]):
                continue
            examine = locations[(pred["detector_target"][locations].astype(bool)) | raw["settling"][pred["window_end_index"][locations]].astype(bool)]
            peak = examine[np.argmax(pred["probability"][examine])]
            start_timestamp = pred["timestamp_us"][locations[0]]
            missed_events.append({
                "scenario": "weak_20_35N" if scenario == 1 else "boundary_35_45N",
                "episode_index": int(episode),
                "force_interval_peak_pd": float(pred["probability"][target_locations].max()),
                "force_and_settling_peak_pd": float(pred["probability"][examine].max()),
                "peak_s": float((pred["timestamp_us"][peak] - start_timestamp) / 1e6),
                **command_summary(raw, int(pred["window_end_index"][peak])),
            })
    false_events.sort(key=lambda item: (item["episode_index"], item["start_s"]))
    args.output_dir = resolve(args.output_dir); args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tcn_b3_generalization_error_analysis.json"
    plot_path = args.output_dir / "tcn_b3_generalization_false_alarms.png"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"false_alarm_events": false_events, "missed_weak_or_boundary_events": missed_events}, file, ensure_ascii=False, indent=2)
    plot_false_alarms(pred, raw, false_events, plot_path)
    print(f"false_alarm_events={len(false_events)}, missed_weak_or_boundary_events={len(missed_events)}")
    print(f"json={json_path}\nplot={plot_path}")


if __name__ == "__main__":
    main()
