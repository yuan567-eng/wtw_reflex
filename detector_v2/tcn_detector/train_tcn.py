"""训练 Detector V2 的 TCN，并按独立 episode 划分训练/验证集。"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tcn import DisturbanceTCN, receptive_field
from window_dataset import DEFAULT_INPUTS, FEATURE_DIMS, TCNWindowDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Detector V2 的 B0/B1/B2 因果 TCN")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--variant", choices=tuple(FEATURE_DIMS), required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-progress", action="store_true", help="关闭 batch 级进度条")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0 or args.batch_size <= 0:
        parser.error("window-size、stride、batch-size 必须大于 0")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.hidden_channels <= 0:
        parser.error("epochs、learning-rate、hidden-channels 必须大于 0")
    if not 0 < args.val_fraction < 1 or not 0 < args.threshold < 1:
        parser.error("val-fraction 和 threshold 必须位于 (0,1)")
    if not 0 <= args.dropout < 1:
        parser.error("dropout 必须位于 [0,1)")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output(variant: str) -> Path:
    return PROJECT_ROOT / "detector_v2/tcn_detector/checkpoints" / f"tcn_{variant}_raw.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch 不可用 CUDA")
    return torch.device(requested)


def split_by_episode(dataset: TCNWindowDataset, val_fraction: float, seed: int):
    """分层划分正常/异常 episode，避免同一 episode 的窗口泄漏到验证集。"""
    grouped_windows = defaultdict(list)
    group_has_positive = defaultdict(bool)
    for window_index, window in enumerate(dataset.windows):
        source = dataset.sources[window.source_index]
        key = (window.source_index, int(source["episode_id"][window.end]))
        grouped_windows[key].append(window_index)
        if source["loss_mask"][window.end] and source["detector_target"][window.end] == 1:
            group_has_positive[key] = True

    normal_groups = sorted(key for key in grouped_windows if not group_has_positive[key])
    abnormal_groups = sorted(key for key in grouped_windows if group_has_positive[key])
    if len(normal_groups) < 2 or len(abnormal_groups) < 2:
        raise ValueError(
            "episode 级划分至少需要 2 个正常和 2 个异常 episode；"
            f"当前 normal={len(normal_groups)}、abnormal={len(abnormal_groups)}"
        )
    rng = random.Random(seed)
    rng.shuffle(normal_groups)
    rng.shuffle(abnormal_groups)

    def split(groups):
        val_count = min(max(1, int(round(len(groups) * val_fraction))), len(groups) - 1)
        return groups[val_count:], groups[:val_count]

    train_normal, val_normal = split(normal_groups)
    train_abnormal, val_abnormal = split(abnormal_groups)
    train_groups = train_normal + train_abnormal
    val_groups = val_normal + val_abnormal
    train_indices = [index for key in train_groups for index in grouped_windows[key]]
    val_indices = [index for key in val_groups for index in grouped_windows[key]]
    return train_indices, val_indices, train_groups, val_groups


def count_valid_labels(dataset: TCNWindowDataset, indices):
    labels = []
    for index in indices:
        window = dataset.windows[index]
        source = dataset.sources[window.source_index]
        if source["loss_mask"][window.end]:
            labels.append(int(source["detector_target"][window.end]))
    labels = np.asarray(labels, dtype=np.int64)
    return int((labels == 0).sum()), int((labels == 1).sum())


def epoch_metrics(model, loader, criterion, device, threshold, optimizer=None, progress=False, desc=""):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_valid = 0
    probabilities, targets = [], []
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterator:
            features = batch["features"].to(device=device, dtype=torch.float32)
            labels = batch["target"].to(device=device, dtype=torch.float32)
            masks = batch["loss_mask"].to(device=device, dtype=torch.bool)
            if not torch.any(masks):
                continue
            logits = model(features)
            loss = criterion(logits, labels)[masks].mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            valid_count = int(masks.sum().item())
            total_loss += float(loss.item()) * valid_count
            total_valid += valid_count
            probabilities.append(torch.sigmoid(logits[masks]).detach().cpu())
            targets.append(labels[masks].detach().cpu())
            if progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
    if total_valid == 0:
        raise RuntimeError("当前 epoch 没有有效窗口")
    probability = torch.cat(probabilities).numpy()
    target = torch.cat(targets).numpy().astype(np.int64)
    prediction = (probability >= threshold).astype(np.int64)
    tp = int(((prediction == 1) & (target == 1)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    tn = int(((prediction == 0) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": total_loss / total_valid,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def group_descriptions(groups):
    return [{"source_index": int(source), "episode_id": int(episode)} for source, episode in groups]


def train(args):
    set_seed(args.seed)
    device = select_device(args.device)
    dataset = TCNWindowDataset(args.inputs, args.variant, args.window_size, args.stride)
    train_indices, val_indices, train_groups, val_groups = split_by_episode(
        dataset, args.val_fraction, args.seed
    )
    train_negative, train_positive = count_valid_labels(dataset, train_indices)
    val_negative, val_positive = count_valid_labels(dataset, val_indices)
    if train_positive == 0 or val_positive == 0:
        raise RuntimeError("训练集和验证集都必须包含有效正样本")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)
    model_config = {
        "input_channels": FEATURE_DIMS[args.variant],
        "hidden_channels": args.hidden_channels,
        "kernel_size": 5,
        "dilations": (1, 2, 4, 8, 16),
        "dropout": args.dropout,
    }
    model = DisturbanceTCN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positive_weight = train_negative / train_positive
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(positive_weight, device=device, dtype=torch.float32),
    )
    output = resolve_path(args.output) if args.output is not None else default_output(args.variant)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={device} variant={args.variant} input_channels={model_config['input_channels']}")
    print(f"receptive_field={receptive_field(model_config['kernel_size'], model_config['dilations'])} frames")
    print(f"train episodes={len(train_groups)}, valid negative={train_negative}, positive={train_positive}")
    print(f"val episodes={len(val_groups)}, valid negative={val_negative}, positive={val_positive}")
    print(f"positive_weight={positive_weight:.4f}")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_result = epoch_metrics(model, train_loader, criterion, device, args.threshold, optimizer, not args.no_progress, f"train {epoch}/{args.epochs}")
        val_result = epoch_metrics(model, val_loader, criterion, device, args.threshold, None, not args.no_progress, f"val {epoch}/{args.epochs}")
        print(
            f"epoch={epoch:03d} train_loss={train_result['loss']:.6f} "
            f"val_loss={val_result['loss']:.6f} val_P={val_result['precision']:.3f} "
            f"val_R={val_result['recall']:.3f} val_F1={val_result['f1']:.3f} "
            f"val_FPR={val_result['false_positive_rate']:.3f} "
            f"confusion=[TP={val_result['tp']},FP={val_result['fp']},TN={val_result['tn']},FN={val_result['fn']}]"
        )
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "variant": args.variant,
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "threshold": args.threshold,
                    "train_groups": group_descriptions(train_groups),
                    "val_groups": group_descriptions(val_groups),
                    "best_val_loss": best_val_loss,
                },
                output,
            )
    print(f"最佳 checkpoint：{output}")
    print(f"best_val_loss={best_val_loss:.6f}")
    return output


if __name__ == "__main__":
    train(parse_args())
