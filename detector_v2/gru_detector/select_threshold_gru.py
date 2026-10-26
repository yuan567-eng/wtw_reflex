"""仅以 checkpoint 固化的验证 episode 为 GRU 选择报警阈值。"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gru import DisturbanceGRU
from window_dataset import DEFAULT_INPUTS, GRUWindowDataset

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS=tuple(np.round(np.arange(.05,.96,.05),2))
def parse_args():
    p=argparse.ArgumentParser(description="仅通过验证集为 V2 GRU 选阈值")
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("inputs",type=Path,nargs="*",default=list(DEFAULT_INPUTS));p.add_argument("--thresholds",type=float,nargs="+",default=DEFAULT_THRESHOLDS);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--output",type=Path);a=p.parse_args()
    if a.batch_size<=0 or any(not 0<x<1 for x in a.thresholds):p.error("阈值或 batch-size 无效")
    return a
def resolve(x):return x if x.is_absolute() else PROJECT_ROOT/x
def device_of(x):
    if x=="auto":return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x=="cuda" and not torch.cuda.is_available():raise RuntimeError("请求 CUDA，但 PyTorch 无 CUDA")
    return torch.device(x)
def calc(prob,target,threshold):
    pred=prob>=threshold;tp,fp,tn,fn=int((pred&target).sum()),int((pred&~target).sum()),int((~pred&~target).sum()),int((~pred&target).sum());p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return {"threshold":float(threshold),"precision":p,"recall":r,"f1":2*p*r/max(p+r,1e-12),"confusion":{"tp":tp,"fp":fp,"tn":tn,"fn":fn}}
def main():
    a=parse_args();device=device_of(a.device); path=resolve(a.checkpoint); ckpt=torch.load(path,map_location=device,weights_only=False)
    if [x for x in ("model_state_dict","model_config","variant","window_size","stride","val_groups") if x not in ckpt]:raise ValueError("checkpoint 不包含验证集划分")
    data=GRUWindowDataset(a.inputs,ckpt["variant"],int(ckpt["window_size"]),int(ckpt["stride"]));groups={(int(x["source_index"]),int(x["episode_id"])) for x in ckpt["val_groups"]};indices=[]
    for i,item in enumerate(data.windows):
        source=data.sources[item.source_index]
        if (item.source_index,int(source["episode_id"][item.end])) in groups:indices.append(i)
    if not indices:raise ValueError("未找到 checkpoint 对应的验证窗口")
    model=DisturbanceGRU(**ckpt["model_config"]).to(device);model.load_state_dict(ckpt["model_state_dict"]);model.eval();prob=[];target=[]
    with torch.no_grad():
        for batch in DataLoader(Subset(data,indices),batch_size=a.batch_size):
            mask=batch["loss_mask"].numpy().astype(bool);prob.append(torch.sigmoid(model(batch["features"].to(device,dtype=torch.float32))).cpu().numpy()[mask]);target.append(batch["target"].numpy().astype(bool)[mask])
    prob,target=np.concatenate(prob),np.concatenate(target);rows=[calc(prob,target,x) for x in sorted(set(a.thresholds))];best=max(rows,key=lambda x:(x["f1"],x["recall"],-x["threshold"]))
    output=resolve(a.output) if a.output else PROJECT_ROOT/"detector_v2/gru_detector/evaluation"/f"{path.stem}_validation_threshold.json";output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as f:json.dump({"variant":ckpt["variant"],"checkpoint":str(path),"selection_rule":"最大验证集窗口级 F1；并列时更高 recall；仍并列时更低 threshold。","validation_windows":int(len(target)),"validation_positive_windows":int(target.sum()),"selected":best,"scan":rows},f,ensure_ascii=False,indent=2)
    for x in rows:print(f"threshold={x['threshold']:.2f} F1={x['f1']:.4f} P={x['precision']:.4f} R={x['recall']:.4f} FP={x['confusion']['fp']}"+(" *" if x==best else ""))
    print(f"selected_threshold={best['threshold']:.2f}\noutput={output}")
if __name__=="__main__":main()
