"""使用 GRU V2 checkpoint 对原始 V2 数据集作因果窗口预测。"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from gru import DisturbanceGRU
from window_dataset import FEATURE_DIMS, build_features, find_episode_ranges, load_episode_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    p = argparse.ArgumentParser(description="使用 V2 GRU 预测 .npz 数据集")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--input", type=Path, default=PROJECT_ROOT / "detector_v2/dataset/raw/test_5min.npz")
    p.add_argument("--output", type=Path); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--threshold", type=float)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto"); a = p.parse_args()
    if a.batch_size <= 0 or (a.threshold is not None and not 0 < a.threshold < 1): p.error("batch-size 或 threshold 无效")
    return a
def resolve(p): return p if p.is_absolute() else PROJECT_ROOT / p
def select_device(value):
    if value == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available(): raise RuntimeError("请求 CUDA，但 PyTorch 无 CUDA")
    return torch.device(value)

def main():
    args = parse_args(); device = select_device(args.device); checkpoint_path, input_path = resolve(args.checkpoint), resolve(args.input)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = ("model_state_dict","model_config","variant","window_size","stride","threshold")
    if [x for x in required if x not in checkpoint]: raise ValueError("checkpoint 字段不完整")
    variant = checkpoint["variant"]
    if variant not in FEATURE_DIMS or checkpoint["model_config"].get("input_size") != FEATURE_DIMS[variant]: raise ValueError("checkpoint 输入配置错误")
    model = DisturbanceGRU(**checkpoint["model_config"]).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    arrays = load_episode_file(input_path); features = build_features(arrays, variant); size, stride = int(checkpoint["window_size"]), int(checkpoint["stride"])
    windows = [(start,start+size-1) for begin,end in find_episode_ranges(arrays["episode_id"], input_path) for start in range(begin,end-size+1,stride)]
    if not windows: raise ValueError("没有可预测窗口")
    values=[]
    with torch.no_grad():
        for offset in range(0,len(windows),args.batch_size):
            batch=np.stack([features[a:b+1].T for a,b in windows[offset:offset+args.batch_size]])
            values.append(torch.sigmoid(model(torch.from_numpy(batch).to(device,dtype=torch.float32))).cpu().numpy())
    probability=np.concatenate(values).astype(np.float32); ends=np.asarray([b for _,b in windows]); threshold=float(checkpoint["threshold"] if args.threshold is None else args.threshold)
    result={"probability":probability,"detected":(probability>=threshold).astype(bool),"threshold":np.float32(threshold),"variant":np.asarray(variant),"window_size":np.int32(size),"stride":np.int32(stride),"window_start_index":ends-size+1,"window_end_index":ends,"timestamp_us":arrays["timestamp_us"][ends],"detector_target":arrays["detector_target"][ends].astype(np.int8),"loss_mask":arrays["loss_mask"][ends].astype(bool),"episode_id":arrays["episode_id"][ends].astype(np.int64)}
    # 预测不使用这两个字段；仅原样带到评估阶段定义 30 个真实事件及其力度分段。
    for name in ("collection_episode_index","disturbance_force_y"):
        if name in arrays: result[name]=arrays[name][ends]
    output=resolve(args.output) if args.output else PROJECT_ROOT / "detector_v2/gru_detector/predictions" / f"{checkpoint_path.stem}__{input_path.stem}_prediction.npz"; output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,**result)
    print(f"checkpoint={checkpoint_path}\ninput={input_path}\nvariant={variant}, device={device}, windows={len(windows)}, threshold={threshold:.3f}\ndetected_windows={int(result['detected'].sum())}, output={output}")

if __name__ == "__main__": main()
