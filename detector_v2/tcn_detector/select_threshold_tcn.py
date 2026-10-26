"""仅使用 checkpoint 保存的验证 episode 为 TCN 选择报警阈值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tcn import DisturbanceTCN
from window_dataset import DEFAULT_INPUTS, TCNWindowDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def parse_args():
    parser = argparse.ArgumentParser(description="从 TCN checkpoint 的验证集选择报警阈值")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须大于 0")
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("thresholds 必须位于 (0,1)")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 不可用 CUDA")
    return torch.device(requested)


def default_output(checkpoint: Path):
    return PROJECT_ROOT / "detector_v2/tcn_detector/evaluation" / f"{checkpoint.stem}_validation_threshold.json"


def metrics(probability, target, threshold):
    predicted = probability >= threshold
    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    tn = int(np.sum(~predicted & ~target))
    fn = int(np.sum(~predicted & target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def select_threshold(args):
    device = select_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = ("model_state_dict", "model_config", "variant", "window_size", "stride", "val_groups")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint 缺少字段：{missing}")
    dataset = TCNWindowDataset(args.inputs, checkpoint["variant"], checkpoint["window_size"], checkpoint["stride"])
    val_groups = {
        (int(item["source_index"]), int(item["episode_id"]))
        for item in checkpoint["val_groups"]
    }
    val_indices = []
    for index, window in enumerate(dataset.windows):
        source = dataset.sources[window.source_index]
        key = (window.source_index, int(source["episode_id"][window.end]))
        if key in val_groups:
            val_indices.append(index)
    if not val_indices:
        raise ValueError("未在当前输入数据中找到 checkpoint 的验证窗口")

    model = DisturbanceTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)
    probabilities, targets = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            masks = batch["loss_mask"].numpy().astype(bool)
            logits = model(features)
            probabilities.append(torch.sigmoid(logits).cpu().numpy()[masks])
            targets.append(batch["target"].numpy().astype(bool)[masks])
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets)
    results = [metrics(probability, target, threshold) for threshold in sorted(set(args.thresholds))]
    # 先最大 F1；若 F1 并列，优先更高召回；仍并列时优先较低阈值。
    best = max(results, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    output = resolve_path(args.output) if args.output is not None else default_output(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": checkpoint["variant"],
        "checkpoint": str(checkpoint_path),
        "selection_rule": "最大验证集窗口级 F1；并列时更高 recall；仍并列时更低 threshold。",
        "validation_windows": int(len(target)),
        "validation_positive_windows": int(target.sum()),
        "selected": best,
        "scan": results,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"variant={checkpoint['variant']}, validation_windows={len(target)}, positives={int(target.sum())}")
    for row in results:
        marker = " *" if row["threshold"] == best["threshold"] else ""
        print(f"threshold={row['threshold']:.2f} F1={row['f1']:.4f} P={row['precision']:.4f} R={row['recall']:.4f} FP={row['confusion']['fp']}{marker}")
    print(f"selected_threshold={best['threshold']:.2f}")
    print(f"output={output}")
    return output


if __name__ == "__main__":
    select_threshold(parse_args())
