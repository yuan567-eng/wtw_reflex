# TCN Detector V2

此目录只维护 TCN 的特征处理、窗口构建、训练、预测和评估代码。

模型权重、预测 `.npz` 和评估图像只保存在本地；可复核的实验配置和 JSON 指标应进入版本控制。

## B0 / B1 / B2 / B3 特征

`window_dataset.py` 独立实现 TCN 的原始尺度特征和因果窗口：

```text
B0: imu_raw[6] + commands[4] + gait_one_hot[4]              = 14 维
B1: B0 + (joint_pos - joint_default_pos)[12]                = 26 维
B2: B1 + joint_vel[12]                                      = 38 维
B3: B0 + joint_torque[12]                                   = 26 维
B3-delta: B3 + command_delta[4]                             = 30 维
```

窗口长度为 100 帧（200 Hz 下的 0.5 秒），stride 默认为 4 帧。窗口只在同一
`episode_id` 内构建，且以窗口末帧的 `detector_target` 和 `loss_mask` 作为监督标签。

## 训练

`tcn.py` 定义轻量因果 TCN；`train_tcn.py` 以 episode 为单位分层划分训练/验证集，
不会把同一个 episode 的相邻窗口泄漏到两侧。checkpoint 记录 variant、输入通道数、
窗口参数和实际 episode 划分。最佳模型以最低验证集 loss 保存。

## 预测

`predict_tcn.py` 根据 checkpoint 中保存的 variant 自动构造 B0/B1/B2/B3 特征，对一个
原始 V2 数据集逐窗口输出 `probability=Pd` 与 `detected=(Pd >= threshold)`。预测文件
同时保留窗口末帧标签、episode、力度阶段元数据，供 `evaluate_tcn.py` 进行统一评估。

## 评估

`evaluate_tcn.py` 接收一个或多个预测文件，先检查它们是否来自同一测试窗口和标签，
再输出窗口级 F1、精确率、召回率、FP 与报警事件数。事件级规则为：一次真实施力区间
内至少一个有效报警窗口即算检出；测试集按 episode 序号统计原范围、扩大范围与弱扰动
三个力度阶段。

## 弱扰动诊断

`analyze_weak_events.py` 针对测试集 episode 18--29，生成 B0/B1/B2 在真实施力附近的
Pd 对比图和逐事件 JSON。每个模型同时统计施力区间内的 Pd 峰值，以及施力加恢复期内
的峰值，用来区分“未响应”和“有响应但阈值未触发”。

## 阈值扫描

`scan_thresholds.py` 不重新训练模型，而是对既有预测的同一 Pd 扫描不同报警阈值，
同步输出窗口级指标、报警事件数、总事件检出和弱扰动检出率。默认输入为 B1，默认扫描
`0.1` 至 `0.8`。

## 验证集阈值选择

`select_threshold_tcn.py` 只读取 checkpoint 中保存的 `val_groups`，在验证窗口上扫描阈值。
选择规则为最大窗口级 F1；F1 并列时优先高召回、再优先低阈值。该步骤不读取独立 5 分钟
测试集，避免把测试集信息泄漏进阈值选择。
