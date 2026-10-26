"""统一评估 GRU B0/B1/B2 的窗口级与事件级指标。"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[2]
STAGES=(("original_40_80N",range(0,6)),("expanded_60_120N",range(6,18)),("weak_20_40N",range(18,30)))
NEEDED=("probability","detected","threshold","variant","window_size","stride","window_start_index","window_end_index","timestamp_us","detector_target","loss_mask","episode_id","collection_episode_index","disturbance_force_y")
def resolve(p): return p if p.is_absolute() else PROJECT_ROOT/p
def load(path):
    path=resolve(path)
    with np.load(path,allow_pickle=False) as raw:
        missing=[x for x in NEEDED if x not in raw.files]
        if missing: raise ValueError(f"{path} 缺少字段：{missing}")
        a={x:np.asarray(raw[x]) for x in NEEDED}
    n=len(a["probability"])
    if any(a[x].shape != (n,) for x in NEEDED if x not in ("threshold","variant","window_size","stride")): raise ValueError(f"{path} 预测长度不一致")
    return path,a
def regions(mask):
    pos=np.flatnonzero(mask)
    if not len(pos): return []
    cut=np.flatnonzero(np.diff(pos)!=1)+1; return [(int(pos[a]),int(pos[b-1])) for a,b in zip(np.r_[0,cut],np.r_[cut,len(pos)])]
def window(a):
    valid=a["loss_mask"].astype(bool); target=a["detector_target"].astype(bool)[valid]; pred=a["detected"].astype(bool)[valid]
    tp,fp,tn,fn=int((pred&target).sum()),int((pred&~target).sum()),int((~pred&~target).sum()),int((~pred&target).sum()); p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return {"valid_windows":int(valid.sum()),"positive_windows":int(target.sum()),"precision":p,"recall":r,"f1":2*p*r/max(p+r,1e-12),"false_positive_rate":fp/max(fp+tn,1),"confusion":{"tp":tp,"fp":fp,"tn":tn,"fn":fn}}
def events(a):
    target=a["detector_target"].astype(bool); detected=a["detected"].astype(bool); valid=a["loss_mask"].astype(bool); ep=a["episode_id"]; idx=a["collection_episode_index"]
    all_events=[]; alarms=[]
    for item in np.unique(ep):
        loc=np.flatnonzero(ep==item)
        for x,y in regions(target[loc]): all_events.append({"episode_index":int(idx[loc[x]]),"detected":bool((detected[loc[x:y+1]]&valid[loc[x:y+1]]).any())})
        for x,y in regions(detected[loc]&valid[loc]): alarms.append({"episode_index":int(idx[loc[x]]),"overlaps_target":bool(target[loc[x:y+1]].any())})
    if len(all_events)!=30: raise ValueError(f"测试应有30个事件，实际{len(all_events)}")
    stages={name:{"events":len(e:=[item for item in all_events if item["episode_index"] in ids]),"detected":sum(item["detected"] for item in e),"event_recall":sum(item["detected"] for item in e)/max(len(e),1)} for name,ids in STAGES}
    return {"events":30,"detected":sum(x["detected"] for x in all_events),"event_recall":sum(x["detected"] for x in all_events)/30,"alarm_events":len(alarms),"false_alarm_events":sum(not x["overlaps_target"] for x in alarms),"stages":stages}
def main():
    p=argparse.ArgumentParser(description="评估 V2 GRU B0/B1/B2"); p.add_argument("predictions",type=Path,nargs="+"); p.add_argument("--output",type=Path,default=PROJECT_ROOT/"detector_v2/gru_detector/evaluation/gru_b0_b1_b2_test_5min.json"); args=p.parse_args()
    loaded=[load(x) for x in args.predictions]; ref_path,ref=loaded[0]; results=[]
    compare=("window_start_index","window_end_index","timestamp_us","detector_target","loss_mask","episode_id","collection_episode_index","disturbance_force_y")
    for path,a in loaded:
        if any(not np.array_equal(ref[x],a[x]) for x in compare): raise ValueError(f"{path} 与 {ref_path} 不是同一测试集")
        results.append({"variant":str(a["variant"].item()),"prediction_file":str(path),"threshold":float(a["threshold"]),"window_metrics":window(a),"event_metrics":events(a)})
    output=resolve(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as f: json.dump({"description":"GRU V2 B0/B1/B2 独立5分钟测试比较。","event_rule":"真实施力区间内至少一个有效窗口报警即检出。","results":results},f,ensure_ascii=False,indent=2)
    for r in results: print(f"{r['variant']}: F1={r['window_metrics']['f1']:.4f}, P={r['window_metrics']['precision']:.4f}, R={r['window_metrics']['recall']:.4f}, FP={r['window_metrics']['confusion']['fp']}, alarm_events={r['event_metrics']['alarm_events']}, event_detect={r['event_metrics']['detected']}/30")
    print(f"已保存：{output}")
if __name__=="__main__": main()
