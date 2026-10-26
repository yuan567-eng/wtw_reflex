# Detector V2 数据模块

此目录将保存 Detector V2 的专用采集脚本、数据格式说明和本地实验数据。

```text
scripts/       采集、归一化、窗口构建和绘图脚本
raw/           原始 episode 数据
normalized/    归一化后的数据
metadata/      数据版本、采集配置和字段说明
ablation/      弱扰动与特征消融实验数据
```

本目录不会修改或覆盖根目录 `dataset/` 中的第一版数据。

## V2 episode 格式

`scripts/collect_normal.py` 和 `scripts/collect_disturbance.py` 输出相同的训练字段；
两者仅在标签内容上不同：正常数据的 `detector_target` 与 `settling` 全为 0，
扰动数据则按物理步标记施力区间和恢复期。

除 V1 字段外，新增：

```text
joint_pos          [T, 12] float32，单位 rad
joint_vel          [T, 12] float32，单位 rad/s
joint_names        [12]    关节顺序
joint_default_pos  [12]    WTW 默认关节姿态，单位 rad
```

`joint_pos` 保存原始真实关节角；后续 TCN 或 GRU 可各自决定使用原始角度或
`joint_pos - joint_default_pos`，从而保持 B0、B1、B2 特征消融的独立性。

扰动采集中的标签语义：

```text
detector_target=1  实际向机身施加外力的物理步
settling=1         外力结束后的恢复期；该段 loss_mask=false，不作为训练监督
loss_mask=true     正常段与实际施力段均参加训练
```

## B0 / B1 / B2 正式实验采集

`scripts/collect_experiment.py` 固化本项目已确认的实验设计。每个命令只输出
一个合并 `.npz`；文件内部以 `episode_id` 和 `collection_episode_index` 区分独立
的 10 秒片段，窗口构建时不得跨越它们。

| preset | 输出文件 | 内容 |
|---|---|---|
| `train_normal_1min` | `raw/train_normal_1min.npz` | 6 个 10 秒无扰动 episode，随机且平滑的运动命令 |
| `train_abnormal_4min` | `raw/train_abnormal_4min.npz` | 24 个 10 秒 episode，随机命令；每段一次随机 40--80 N、3--7 秒的侧向扰动 |
| `test_5min` | `raw/test_5min.npz` | 30 个 10 秒 episode；每段第 5 秒扰动，依次为 40--80 N（6 次）、60--120 N（12 次）、20--40 N（12 次） |
| `final_b3_delta_5min` | `raw/final_b3_delta_5min.npz` | 仅用于冻结的 B3-delta 最终复验：12 段复杂正常运动，随后弱/临界/强未见扰动各 6 段；必须使用全新命令和扰动随机种子 |

训练与测试集都会保存 `disturbance_force_y[T]`。该字段只用于按力度范围统计
事件检出率，**不能**作为 B0/B1/B2 的网络输入。
