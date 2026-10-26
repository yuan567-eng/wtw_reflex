"""训练 GRU Detector V2，并以 episode 为单位隔离验证集。"""
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

from gru import DisturbanceGRU, parameter_count
from window_dataset import DEFAULT_INPUTS, FEATURE_DIMS, GRUWindowDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def args_parser():
    parser = argparse.ArgumentParser(description="训练 V2 GRU B0/B1/B2")
    parser.add_argument("inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--variant", choices=tuple(FEATURE_DIMS), required=True)
    parser.add_argument("--window-size", type=int, default=100); parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--val-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--hidden-size", type=int, default=48); parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.1); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-progress", action="store_true"); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.window_size, args.stride, args.batch_size, args.epochs, args.hidden_size, args.num_layers) <= 0: parser.error("尺寸和 epochs 必须大于 0")
    if args.learning_rate <= 0 or not 0 < args.val_fraction < 1 or not 0 < args.threshold < 1 or not 0 <= args.dropout < 1: parser.error("训练参数取值无效")
    return args


def resolve(path): return path if path.is_absolute() else PROJECT_ROOT / path
def device_of(value):
    if value == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available(): raise RuntimeError("请求 CUDA，但 PyTorch 无 CUDA")
    return torch.device(value)
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_by_episode(dataset, fraction, seed):
    groups, positive = defaultdict(list), defaultdict(bool)
    for index, item in enumerate(dataset.windows):
        source = dataset.sources[item.source_index]; key = (item.source_index, int(source["episode_id"][item.end]))
        groups[key].append(index)
        if source["loss_mask"][item.end] and source["detector_target"][item.end] == 1: positive[key] = True
    normal, abnormal = sorted(k for k in groups if not positive[k]), sorted(k for k in groups if positive[k])
    if len(normal) < 2 or len(abnormal) < 2: raise ValueError(f"至少要 2 个正常和异常 episode，当前 {len(normal)}/{len(abnormal)}")
    rng = random.Random(seed); rng.shuffle(normal); rng.shuffle(abnormal)
    def split(items):
        n = min(max(1, round(len(items) * fraction)), len(items) - 1)
        return items[n:], items[:n]
    tr_n, va_n = split(normal); tr_a, va_a = split(abnormal); train, valid = tr_n + tr_a, va_n + va_a
    return [i for key in train for i in groups[key]], [i for key in valid for i in groups[key]], train, valid


def labels(dataset, indices):
    values = []
    for index in indices:
        item = dataset.windows[index]; source = dataset.sources[item.source_index]
        if source["loss_mask"][item.end]: values.append(int(source["detector_target"][item.end]))
    values = np.asarray(values); return int((values == 0).sum()), int((values == 1).sum())


def run_epoch(model, loader, loss_fn, device, threshold, optimizer=None, progress=False, desc=""):
    training = optimizer is not None; model.train(training); total_loss = 0.; total = 0; probs = []; targets = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False, disable=not progress):
            x = batch["features"].to(device, dtype=torch.float32); y = batch["target"].to(device, dtype=torch.float32); mask = batch["loss_mask"].to(device)
            if not mask.any(): continue
            logits = model(x); loss = loss_fn(logits, y)[mask].mean()
            if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
            amount = int(mask.sum()); total += amount; total_loss += loss.item() * amount
            probs.append(torch.sigmoid(logits[mask]).detach().cpu()); targets.append(y[mask].detach().cpu())
    p, y = torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(bool); pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum()); tn, fn = int((~pred & ~y).sum()), int((~pred & y).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"loss": total_loss / total, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "fpr": fp / max(fp + tn, 1), "tp":tp,"fp":fp,"tn":tn,"fn":fn}


def main():
    args = args_parser(); set_seed(args.seed); device = device_of(args.device)
    data = GRUWindowDataset(args.inputs, args.variant, args.window_size, args.stride)
    train_idx, val_idx, train_groups, val_groups = split_by_episode(data, args.val_fraction, args.seed)
    neg, pos = labels(data, train_idx); val_neg, val_pos = labels(data, val_idx)
    if not pos or not val_pos: raise RuntimeError("训练、验证集均需包含有效正样本")
    config = {"input_size": FEATURE_DIMS[args.variant], "hidden_size": args.hidden_size, "num_layers":args.num_layers, "dropout":args.dropout}
    model = DisturbanceGRU(**config).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(neg / pos, device=device))
    train_loader = DataLoader(Subset(data, train_idx), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(data, val_idx), batch_size=args.batch_size)
    output = resolve(args.output) if args.output else PROJECT_ROOT / "detector_v2/gru_detector/checkpoints" / f"gru_{args.variant}_raw.pt"; output.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} variant={args.variant} input_size={config['input_size']} parameters={parameter_count(model)}")
    print(f"train episodes={len(train_groups)} valid negative={neg} positive={pos}; val episodes={len(val_groups)} negative={val_neg} positive={val_pos}")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, args.threshold, optimizer, not args.no_progress, f"train {epoch}/{args.epochs}")
        val = run_epoch(model, val_loader, criterion, device, args.threshold, None, not args.no_progress, f"val {epoch}/{args.epochs}")
        print(f"epoch={epoch:03d} train_loss={train['loss']:.6f} val_loss={val['loss']:.6f} val_P={val['precision']:.3f} val_R={val['recall']:.3f} val_F1={val['f1']:.3f} val_FPR={val['fpr']:.3f} confusion=[TP={val['tp']},FP={val['fp']},TN={val['tn']},FN={val['fn']}]")
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model_state_dict": model.state_dict(), "model_config":config, "variant":args.variant, "window_size":args.window_size, "stride":args.stride, "threshold":args.threshold, "train_groups":[{"source_index":int(a),"episode_id":int(b)} for a,b in train_groups], "val_groups":[{"source_index":int(a),"episode_id":int(b)} for a,b in val_groups], "best_val_loss":best}, output)
    print(f"最佳 checkpoint：{output}\nbest_val_loss={best:.6f}")


if __name__ == "__main__": main()
