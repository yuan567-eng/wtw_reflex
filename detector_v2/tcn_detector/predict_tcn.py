"""使用训练完成的 TCN 对一个 V2 episode 集合进行因果窗口预测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from tcn import DisturbanceTCN
from window_dataset import FEATURE_DIMS, build_features, find_episode_ranges, load_episode_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "detector_v2/dataset/raw/test_5min.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="使用 TCN V2 checkpoint 预测一个原始 V2 .npz 数据集")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None, help="默认使用 checkpoint 内阈值")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须大于 0")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("threshold 必须位于 (0,1)")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 不可用 CUDA")
    return torch.device(requested)


def default_output(checkpoint: Path, input_path: Path) -> Path:
    return (
        PROJECT_ROOT
        / "detector_v2/tcn_detector/predictions"
        / f"{checkpoint.stem}__{input_path.stem}_prediction.npz"
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = ("model_state_dict", "model_config", "variant", "window_size", "stride", "threshold")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint 缺少字段：{missing}")
    variant = checkpoint["variant"]
    if variant not in FEATURE_DIMS:
        raise ValueError(f"checkpoint variant 无效：{variant}")
    if int(checkpoint["model_config"].get("input_channels", -1)) != FEATURE_DIMS[variant]:
        raise ValueError("checkpoint 的 variant 与 input_channels 不一致")
    model = DisturbanceTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def predict(args):
    device = select_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    input_path = resolve_path(args.input)
    checkpoint, model = load_model(checkpoint_path, device)
    variant = checkpoint["variant"]
    window_size = int(checkpoint["window_size"])
    stride = int(checkpoint["stride"])
    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)
    arrays = load_episode_file(input_path)
    features = build_features(arrays, variant)

    windows = []
    for start_episode, end_episode in find_episode_ranges(arrays["episode_id"], input_path):
        last_start = end_episode - window_size
        for start in range(start_episode, last_start + 1, stride):
            windows.append((start, start + window_size - 1))
    if not windows:
        raise ValueError("没有可预测窗口，请检查数据长度和 checkpoint window_size")

    probabilities = []
    with torch.no_grad():
        for offset in range(0, len(windows), args.batch_size):
            current = windows[offset : offset + args.batch_size]
            batch = np.stack([features[start : end + 1].T for start, end in current])
            logits = model(torch.from_numpy(batch).to(device=device, dtype=torch.float32))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(probabilities).astype(np.float32, copy=False)
    end_indices = np.asarray([end for _, end in windows], dtype=np.int64)
    prediction = probability >= threshold

    result = {
        "probability": probability,
        "detected": prediction.astype(np.bool_),
        "threshold": np.float32(threshold),
        "variant": np.asarray(variant),
        "window_size": np.int32(window_size),
        "stride": np.int32(stride),
        "window_start_index": end_indices - window_size + 1,
        "window_end_index": end_indices,
        # 以下字段均为窗口末帧的真实信息；用于后续统一评估，绝不作为预测输入。
        "timestamp_us": arrays["timestamp_us"][end_indices],
        "detector_target": arrays["detector_target"][end_indices].astype(np.int8),
        "loss_mask": arrays["loss_mask"][end_indices].astype(np.bool_),
        "episode_id": arrays["episode_id"][end_indices].astype(np.int64),
    }
    # 这些元数据均不作为网络输入，仅随预测文件传递给不同测试协议的评估脚本。
    for optional in (
        "collection_episode_index", "disturbance_force_y", "scenario_id", "scenario_names",
    ):
        if optional in arrays:
            # scenario_names 是整个数据集的固定字典，不是逐帧数组。
            result[optional] = (
                arrays[optional]
                if optional == "scenario_names"
                else arrays[optional][end_indices]
            )

    output = resolve_path(args.output) if args.output is not None else default_output(checkpoint_path, input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f"checkpoint={checkpoint_path}")
    print(f"input={input_path}")
    print(f"variant={variant}, device={device}, windows={len(windows)}, threshold={threshold:.3f}")
    print(f"detected_windows={int(prediction.sum())}, output={output}")
    return output


if __name__ == "__main__":
    predict(parse_args())
