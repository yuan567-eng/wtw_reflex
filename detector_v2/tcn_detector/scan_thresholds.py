"""对已有 TCN 预测的 Pd 扫描阈值，不重新训练模型。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from evaluate_tcn import event_metrics, load_prediction, window_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "detector_v2/tcn_detector/predictions/tcn_b1_test_5min.npz"
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.1, 0.81, 0.1), 2))


def parse_args():
    parser = argparse.ArgumentParser(description="扫描一个 TCN 预测文件的报警阈值")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "detector_v2/tcn_detector/evaluation/tcn_b1_threshold_scan.json",
    )
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("全部 thresholds 必须位于 (0,1)")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_threshold(arrays, threshold: float):
    """用同一 Pd 重建 detected，确保只有阈值这一变量改变。"""
    current = copy.copy(arrays)
    current["detected"] = arrays["probability"] >= threshold
    current["threshold"] = np.float32(threshold)
    return {
        "threshold": float(threshold),
        "window_metrics": window_metrics(current),
        "event_metrics": event_metrics(current),
    }


def scan(args):
    input_path, arrays = load_prediction(args.input)
    results = [evaluate_threshold(arrays, threshold) for threshold in sorted(set(args.thresholds))]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": str(arrays["variant"].item()),
        "prediction_file": str(input_path),
        "description": "同一组 Pd 的阈值扫描；训练、数据和窗口均不变。",
        "results": results,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("threshold | F1     | P      | R      | FP  | alarms | events | weak")
    for result in results:
        window = result["window_metrics"]
        event = result["event_metrics"]
        weak = event["stages"]["weak_20_40N"]
        print(
            f"{result['threshold']:9.2f} | {window['f1']:.4f} | "
            f"{window['precision']:.4f} | {window['recall']:.4f} | "
            f"{window['confusion']['fp']:3d} | {event['alarm_events']:6d} | "
            f"{event['detected']:2d}/30 | {weak['detected']:2d}/12"
        )
    print(f"已保存：{output}")
    return output


if __name__ == "__main__":
    scan(parse_args())
