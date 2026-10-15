"""按 episode 划分数据并训练最小 trotting 扰动 Detector。"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPTS = PROJECT_ROOT / "dataset/scripts"
if str(DATASET_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPTS))

from window_dataset import DEFAULT_INPUTS, DetectorWindowDataset
from tcn import DisturbanceTCN, receptive_field


DEFAULT_CHECKPOINT = PROJECT_ROOT / "tcn_detector/checkpoints/trotting_tcn.pt"
DEFAULT_NORMALIZATION_REFERENCE = DEFAULT_INPUTS[0]
NORMALIZED_STATS_FIELDS = (
    "center",
    "scale",
    "fitted_gait_mask",
    "command_center",
    "command_scale",
    "epsilon",
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练最小因果 TCN 扰动检测器")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--feature-mode",
        choices=("normalized", "raw"),
        default="normalized",
        help="使用归一化连续特征或原始连续特征",
    )
    parser.add_argument(
        "--normalization-reference",
        type=Path,
        default=DEFAULT_NORMALIZATION_REFERENCE,
        help="normalized模式写入checkpoint的正常数据归一化参数来源",
    )
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0:
        parser.error("window-size 和 stride 必须大于 0")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch-size 和 epochs 必须大于 0")
    if args.learning_rate <= 0:
        parser.error("learning-rate 必须大于 0")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction 必须位于 (0,1)")
    if not 0 < args.threshold < 1:
        parser.error("threshold 必须位于 (0,1)")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return torch.device(requested)


def split_by_episode(dataset, val_fraction, seed):
    """分别划分正常/扰动 episode，禁止同一 episode 的窗口跨集合。"""
    episode_windows = defaultdict(list)
    episode_has_positive = defaultdict(bool)
    for window_index, window in enumerate(dataset.windows):
        source = dataset.sources[window.source_index]
        episode_id = int(source["episode_id"][window.end])
        group_key = (window.source_index, episode_id)
        episode_windows[group_key].append(window_index)
        if source["loss_mask"][window.end] and source["detector_target"][window.end] == 1:
            episode_has_positive[group_key] = True

    normal_groups = sorted(key for key in episode_windows if not episode_has_positive[key])
    abnormal_groups = sorted(key for key in episode_windows if episode_has_positive[key])
    if len(normal_groups) < 2 or len(abnormal_groups) < 2:
        raise ValueError(
            "episode 级划分至少需要2个正常 episode和2个扰动 episode；"
            f"当前为 normal={len(normal_groups)}, abnormal={len(abnormal_groups)}"
        )

    rng = random.Random(seed)
    rng.shuffle(normal_groups)
    rng.shuffle(abnormal_groups)

    def split_groups(groups):
        val_count = max(1, int(round(len(groups) * val_fraction)))
        val_count = min(val_count, len(groups) - 1)
        return groups[val_count:], groups[:val_count]

    train_normal, val_normal = split_groups(normal_groups)
    train_abnormal, val_abnormal = split_groups(abnormal_groups)
    train_groups = train_normal + train_abnormal
    val_groups = val_normal + val_abnormal
    train_indices = [index for key in train_groups for index in episode_windows[key]]
    val_indices = [index for key in val_groups for index in episode_windows[key]]
    return train_indices, val_indices, train_groups, val_groups


def count_valid_labels(dataset, indices):
    targets = []
    for index in indices:
        window = dataset.windows[index]
        source = dataset.sources[window.source_index]
        if source["loss_mask"][window.end]:
            targets.append(int(source["detector_target"][window.end]))
    targets = np.asarray(targets, dtype=np.int64)
    positives = int(np.count_nonzero(targets == 1))
    negatives = int(np.count_nonzero(targets == 0))
    return negatives, positives


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_valid = 0
    all_probabilities = []
    all_targets = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            masks = batch["loss_mask"].to(device=device, dtype=torch.bool)
            if not torch.any(masks):
                continue

            logits = model(features)
            per_sample_loss = criterion(logits, targets)
            loss = per_sample_loss[masks].mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_count = int(masks.sum().item())
            total_loss += float(loss.item()) * valid_count
            total_valid += valid_count
            all_probabilities.append(torch.sigmoid(logits[masks]).detach().cpu())
            all_targets.append(targets[masks].detach().cpu())

    probabilities = torch.cat(all_probabilities).numpy()
    targets = torch.cat(all_targets).numpy().astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(np.sum((predictions == 1) & (targets == 1)))
    false_positive = int(np.sum((predictions == 1) & (targets == 0)))
    true_negative = int(np.sum((predictions == 0) & (targets == 0)))
    false_negative = int(np.sum((predictions == 0) & (targets == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    return {
        "loss": total_loss / total_valid,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def load_normalization_stats(path):
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in NORMALIZED_STATS_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"归一化参考文件缺少参数：{missing}")
        return {
            field: torch.as_tensor(np.asarray(data[field])).clone()
            for field in NORMALIZED_STATS_FIELDS
        }


def format_groups(groups):
    return [int(episode_id) for _, episode_id in groups]


def train(args):
    set_seed(args.seed)
    device = select_device(args.device)
    dataset = DetectorWindowDataset(
        args.inputs,
        args.window_size,
        args.stride,
        feature_mode=args.feature_mode,
    )
    train_indices, val_indices, train_groups, val_groups = split_by_episode(
        dataset, args.val_fraction, args.seed
    )
    train_negative, train_positive = count_valid_labels(dataset, train_indices)
    val_negative, val_positive = count_valid_labels(dataset, val_indices)
    if train_positive == 0 or val_positive == 0:
        raise RuntimeError("训练集和验证集都必须包含有效正样本")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_config = {
        "input_channels": 14,
        "hidden_channels": 32,
        "kernel_size": 5,
        "dilations": (1, 2, 4, 8, 16),
        "dropout": 0.1,
    }
    model = DisturbanceTCN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positive_weight = train_negative / train_positive
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device),
    )

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalization = (
        load_normalization_stats(resolve_input_path(args.normalization_reference))
        if args.feature_mode == "normalized"
        else None
    )
    best_val_loss = float("inf")

    print(f"device={device}")
    print(f"feature_mode={args.feature_mode}")
    print(f"receptive_field={receptive_field(model_config['kernel_size'], model_config['dilations'])}")
    print(f"train_episode_ids={format_groups(train_groups)}")
    print(f"val_episode_ids={format_groups(val_groups)}")
    print(f"train valid: negative={train_negative}, positive={train_positive}")
    print(f"val valid: negative={val_negative}, positive={val_positive}")
    print(f"positive_weight={positive_weight:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.threshold
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, None, args.threshold
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_P={val_metrics['precision']:.3f} "
            f"val_R={val_metrics['recall']:.3f} "
            f"val_F1={val_metrics['f1']:.3f} "
            f"val_FPR={val_metrics['false_positive_rate']:.3f} "
            f"confusion=[TP={val_metrics['tp']},FP={val_metrics['fp']},"
            f"TN={val_metrics['tn']},FN={val_metrics['fn']}]"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "threshold": args.threshold,
                    "feature_mode": args.feature_mode,
                    "normalization": normalization,
                    "train_episode_ids": format_groups(train_groups),
                    "val_episode_ids": format_groups(val_groups),
                    "best_val_loss": best_val_loss,
                },
                output_path,
            )

    print(f"最佳 checkpoint：{output_path}")
    print(f"best_val_loss={best_val_loss:.6f}")
    return output_path


def resolve_input_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    train(parse_args())
