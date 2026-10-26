# GRU Detector V2

独立于 `tcn_detector` 的单向 GRU 扰动检测器。三组消融与 TCN 完全同定义：

| 方案 | 每帧输入 |
| --- | ---: |
| B0 | IMU[6] + commands[4] + gait one-hot[4] = 14 |
| B1 | B0 + `joint_pos - joint_default_pos`[12] = 26 |
| B2 | B1 + `joint_vel`[12] = 38 |

窗口保存为 `[B,C,100]`，在 `gru.py` 的入口转置为 GRU 接口所需的 `[B,100,C]`。单向 GRU 只读取过去到当前的帧；最后一个隐藏状态输出当前窗口末端的扰动概率 `Pd`。

## 执行顺序

先确认三个输入配置均能构造窗口：

```bash
conda run -n gym python detector_v2/gru_detector/window_dataset.py --variant b0
conda run -n gym python detector_v2/gru_detector/window_dataset.py --variant b1
conda run -n gym python detector_v2/gru_detector/window_dataset.py --variant b2
```

训练命令（每组独立运行，默认 CUDA 自动可用、100 epoch）：

```bash
conda run --no-capture-output -n gym python detector_v2/gru_detector/train_gru.py --variant b0 --device cuda --output detector_v2/gru_detector/checkpoints/gru_b0_raw_100ep.pt
```

训练完成后，对每个 checkpoint 仅在其验证集选阈值：

```bash
conda run --no-capture-output -n gym python detector_v2/gru_detector/select_threshold_gru.py --checkpoint detector_v2/gru_detector/checkpoints/gru_b0_raw_100ep.pt --device cuda
```

再用选出的阈值预测同一份五分钟测试集，最后将 B0/B1/B2 的预测文件传给 `evaluate_gru.py` 汇总。禁止根据测试集重新选阈值。
