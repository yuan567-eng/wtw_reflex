"""诊断 12 个弱扰动事件中 B0/B1/B2 的 Pd 峰值与报警情况。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = {
    "b0": PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b0_test_5min.npz",
    "b1": PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b1_test_5min.npz",
    "b2": PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b2_test_5min.npz",
}
WEAK_EPISODE_INDICES = range(18, 30)
REQUIRED_FIELDS = (
    "probability", "detected", "threshold", "timestamp_us", "detector_target",
    "loss_mask", "episode_id", "collection_episode_index", "disturbance_force_y",
)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 B0/B1/B2 的弱扰动 Pd 曲线")
    parser.add_argument("--b0", type=Path, default=DEFAULT_PREDICTIONS["b0"])
    parser.add_argument("--b1", type=Path, default=DEFAULT_PREDICTIONS["b1"])
    parser.add_argument("--b2", type=Path, default=DEFAULT_PREDICTIONS["b2"])
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/weak_events_b0_b1_b2.png",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/weak_events_b0_b1_b2.json",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_prediction(path: Path):
    path = resolve_path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_FIELDS if name not in data.files]
        if missing:
            raise ValueError(f"{path} 缺少字段：{missing}")
        return path, {name: np.asarray(data[name]) for name in REQUIRED_FIELDS}


def assert_same_reference(reference, candidate, reference_path, candidate_path):
    for name in (
        "timestamp_us", "detector_target", "loss_mask", "episode_id",
        "collection_episode_index", "disturbance_force_y",
    ):
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"{candidate_path} 与 {reference_path} 的 {name} 不一致")


def contiguous_region(mask: np.ndarray):
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        raise ValueError("事件中没有 target 窗口")
    if np.any(np.diff(positions) != 1):
        raise ValueError("每个弱扰动 episode 应只有一个连续 target 区间")
    return int(positions[0]), int(positions[-1])


def event_record(event_index, reference, predictions):
    event_mask = reference["collection_episode_index"] == event_index
    local_indices = np.flatnonzero(event_mask)
    target_mask = reference["detector_target"][local_indices].astype(bool)
    target_start, target_end = contiguous_region(target_mask)
    timestamps = reference["timestamp_us"][local_indices].astype(np.float64) / 1e6
    relative_time = timestamps - timestamps[0]
    force_y = float(reference["disturbance_force_y"][local_indices[0]])
    # 施力区间以及其后 0.5 秒，用于区分“未响应”和“恢复期才出现响应”。
    context_mask = (relative_time >= relative_time[target_start]) & (
        relative_time <= relative_time[target_end] + 0.5
    )
    models = {}
    for name, arrays in predictions.items():
        pd = arrays["probability"][local_indices]
        detected = arrays["detected"][local_indices].astype(bool)
        models[name] = {
            "peak_pd_during_force": float(pd[target_start : target_end + 1].max()),
            "peak_pd_force_plus_recovery": float(pd[context_mask].max()),
            "detected_during_force": bool(detected[target_start : target_end + 1].any()),
            "detected_force_plus_recovery": bool(detected[context_mask].any()),
        }
    return {
        "episode_index": int(event_index),
        "force_y_N": force_y,
        "force_abs_N": abs(force_y),
        "force_time_s": [float(relative_time[target_start]), float(relative_time[target_end])],
        "models": models,
    }, local_indices, relative_time, target_start, target_end


def analyze(args):
    paths_and_arrays = {name: load_prediction(getattr(args, name)) for name in ("b0", "b1", "b2")}
    reference_path, reference = paths_and_arrays["b0"]
    predictions = {"b0": reference}
    for name in ("b1", "b2"):
        path, arrays = paths_and_arrays[name]
        assert_same_reference(reference, arrays, reference_path, path)
        predictions[name] = arrays
    threshold = float(reference["threshold"])

    records = []
    plot_data = []
    for episode_index in WEAK_EPISODE_INDICES:
        record, local_indices, relative_time, target_start, target_end = event_record(
            episode_index, reference, predictions
        )
        records.append(record)
        plot_data.append((record, local_indices, relative_time, target_start, target_end))

    args.output_figure = resolve_path(args.output_figure)
    args.output_json = resolve_path(args.output_json)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    colors = {"b0": "#1f77b4", "b1": "#ff7f0e", "b2": "#2ca02c"}
    figure, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
    for axis, (record, local_indices, time_s, target_start, target_end) in zip(axes.flat, plot_data):
        for name, arrays in predictions.items():
            axis.plot(time_s, arrays["probability"][local_indices], label=name.upper(), color=colors[name], linewidth=1.4)
        axis.axvspan(time_s[target_start], time_s[target_end], color="#d62728", alpha=0.16, label="force")
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.8, label="threshold")
        axis.set_title(f"E{record['episode_index']}: |Fy|={record['force_abs_N']:.1f} N")
        axis.set_xlim(max(0.0, time_s[target_start] - 0.5), time_s[target_end] + 0.6)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time within episode (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pd")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("Weak disturbances (20–40 N): Pd near true force interval", y=0.995)
    figure.legend(handles[:5], labels[:5], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.978))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.output_figure, dpi=160)
    plt.close(figure)

    result = {
        "threshold": threshold,
        "description": "弱扰动 episode 18--29；force_time_s 为窗口末帧相对该 episode 的时间。",
        "events": records,
    }
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    for name in ("b0", "b1", "b2"):
        peak = np.asarray([record["models"][name]["peak_pd_during_force"] for record in records])
        detected = sum(record["models"][name]["detected_during_force"] for record in records)
        print(f"{name}: force_peak Pd min/median/max={peak.min():.4f}/{np.median(peak):.4f}/{peak.max():.4f}, detected={detected}/12")
    print(f"figure={args.output_figure}")
    print(f"json={args.output_json}")
    return args.output_figure, args.output_json


if __name__ == "__main__":
    analyze(parse_args())
