## 2026-10-07 修改记录

### 当前目标

实现固定 gait 正常数据采集的最小环境，并验证六维等价 IMU 的通道、频率与基础数值语义。

### 修改内容

- 新增外挂 `ImuCollectionEnv`，在不修改 WTW 源码的前提下，每个 5 ms 物理步计算一次机体系比力和角速度。
- 新增 `dataset/scripts/collect.py`，支持选择固定 gait、可视化/无头采集和 `.npz` 保存。
- 新增 gait 名称到整数 ID 的固定映射。

### 修改原因

原 WTW 策略以 50 Hz 更新动作，但 Detector 数据需要从内部 200 Hz 物理步获取，不能只在策略控制步读取状态。采集逻辑全部放在 `dataset/scripts/` 外挂模块中，原项目文件保持不变。

### 测试方法

```bash
python -m py_compile dataset/scripts/imu_collection_env.py \
  dataset/scripts/collect.py

TORCH_EXTENSIONS_DIR=/tmp/wtw_torch_extensions \
conda run -n gym python dataset/scripts/collect.py \
  --gait trotting --warmup-steps 50 --steps 20 \
  --headless --sim-device cpu --output /tmp/wtw_imu_trot_short.npz
```

### 测试结果

- Python 语法检查通过。
- CPU smoke test 成功保存 `80 × 6` IMU 数据。
- 相邻时间戳均为 `5000 us`，实际采样率为 `200 Hz`。
- 水平静止公式测试输出 `[0, 0, 9.81, 0, 0, 0]`。
- 短时 trotting 数据没有 NaN/Inf，z 轴平均比力约为 `10.36 m/s²`。
- 当前执行环境不可见 CUDA 设备，GPU + actuator network 正式采集尚未验证。

### 当前状态

固定 gait 正常 IMU 采集的最小实现已完成；等待在有 GPU 的 Isaac Gym 环境中验证正式采集路径。

### 下一步

使用 CUDA 运行一次可视化 trotting 采集，检查机器人运动、六通道曲线与输出文件后，再决定是否进入多 gait 批量采集。

## 2026-10-07 修改记录（二）

### 当前目标

为已采集的六维 IMU episode 增加独立可视化工具。

### 修改内容

- 新增 `dataset/scripts/plot.py`。
- 支持自动选择最新 `.npz`、指定原始/归一化字段、保存 PNG 和交互显示。
- 输出样本数、时长、采样率及六通道统计信息。
- 在图中预留 disturbance、switching、settling 和 loss mask 背景标记。

### 修改原因

采集结果保存在 `.npz` 中，无法直接检查六通道时序、采样频率和数据范围。

### 测试方法

```bash
python dataset/scripts/plot.py \
  dataset/raw/normal_gait/trotting/episode_1791361911742507.npz \
  --output /tmp/trotting_imu.png
```

### 测试结果

- 成功读取 `2000 × 6` trotting episode。
- 检测到时长 `9.995 s`、采样率 `200 Hz`、`gait_id=[1]`。
- 成功生成 `2062 × 1256` PNG，六通道曲线连续且没有 NaN/Inf。
- `a_z` 均值约为 `9.81 m/s²`，符合当前比力定义。

### 当前状态

已完成并通过测试。

### 下一步

由开发者检查图像；确认后再决定是否增加分通道视图或进入下一任务。

## 2026-10-07 修改记录（三）

### 当前目标

将四种固定 gait 的正常 episode 合并为一个总数据集，同时保留清晰的 episode 边界。

### 修改内容

- 新增 `dataset/scripts/total.py`。
- 按 pronking、trotting、bounding、pacing、gait_switching 的固定顺序发现输入文件。
- 合并前检查统一字段、数组 shape、IMU 有限值、时间戳、gait 标签、二分类标签和 episode ID。
- 检查不同输入文件的 `episode_id` 不重复，防止后续 TCN 窗口边界失效。
- 默认输出 `dataset/raw/normal_gait/total_gait_data/normal_total.npz`，并防止意外覆盖已有输出。

### 修改原因

训练前需要把多个 gait episode 汇总到统一文件中，但不能丢失 episode 边界，否则后续滑动窗口可能错误地跨越两次独立采集。

### 测试方法

```bash
python -m py_compile dataset/scripts/total.py
python dataset/scripts/total.py
```

另外逐字段比较总数据集与四份源数据，并检查 `episode_id` 的切换位置和已有输出防覆盖行为。

### 测试结果

- 成功合并 4 个 episode，共 `8000 × 6` 条 IMU 样本。
- gait_id 0、1、2、3 各 2000 条样本。
- 总文件全部 8 个字段与源文件逐元素一致。
- episode 边界位于索引 2000、4000、6000，边界保留正确。
- 不带 `--overwrite` 再次运行时能够拒绝覆盖已有输出。

### 当前状态

正常多 gait 数据合并功能已完成并通过测试。

### 下一步

讨论并实现 `normalization.py`，只使用正常数据为四种 gait 分别计算 Q1/Q99 中心与尺度。

## 2026-10-07 修改记录（四）

### 当前目标

修复 `plot.py` 无法绘制多 episode 总数据集的问题。

### 修改内容

- 将 `episode_id` 纳入可选字段读取与 shape 校验。
- 总数据集改为按 `episode_id` 分段检查时间戳，每个 episode 内仍要求严格递增。
- 根据各 episode 的原始时间戳构造连续显示时间轴。
- 分 episode 绘制曲线，避免在两个独立 episode 之间画出虚假的连接线。
- 使用竖线标记 episode 边界，并在终端输出 episode 数量。

### 修改原因

多个独立采集的 episode 各自使用自己的仿真时间戳，合并后全局时间戳会在边界处重新开始。原脚本错误地要求总文件全局时间戳严格递增。

### 测试方法

```bash
python -m py_compile dataset/scripts/plot.py
conda run -n gym python dataset/scripts/plot.py
conda run -n gym python dataset/scripts/plot.py \
  dataset/raw/normal_gait/trotting/episode_1791361911742507.npz \
  --output /tmp/trotting_plot_regression.png
```

### 测试结果

- 默认命令成功绘制 `normal_total.npz`。
- 正确识别 4 个 episode、8000 个样本、39.995 秒和 200 Hz。
- 图中在 10、20、30 秒附近正确标记三个 episode 边界，且曲线不跨边界连接。
- 单 episode trotting 文件回归测试通过，仍为 2000 个样本、9.995 秒和 200 Hz。

### 当前状态

总数据集和单 episode 数据均可正常绘制，问题已修复。

### 下一步

继续讨论并实现 `normalization.py`。

## 2026-10-07 修改记录（五）

### 当前目标

使用正常稳定数据为四种 gait 分别计算鲁棒归一化参数，并生成正常归一化数据集。

### 修改内容

- 新增 `dataset/scripts/normalization.py`。
- 每个 gait、每个 IMU 通道分别计算 Q1/Q99，并据此计算 `center[4,6]` 和 `scale[4,6]`。
- 只使用 `loss_mask=True`、`detector_target=0`、非 switching、非 settling 的样本拟合统计量。
- 默认最小尺度为加速度 `0.1 m/s²`、角速度 `0.01 rad/s`，`epsilon=1e-6`。
- 不裁剪归一化结果，保留超出 Q1/Q99 范围的异常幅值。
- 输出保留原总数据集的全部字段，并新增 `imu_normalized`、`center`、`scale` 和 `epsilon`。

### 修改原因

不同 gait 的正常 IMU 中心和尺度明显不同，需要先按 gait 对齐，Detector 才能更关注相对正常基线的偏离程度。

### 测试方法

```bash
python -m py_compile dataset/scripts/normalization.py
python dataset/scripts/normalization.py
conda run -n gym python dataset/scripts/plot.py \
  dataset/raw/normal_gait/normalization_data/normal_normalized.npz \
  --field imu_normalized
```

另外使用输出文件保存的 `center`、`scale`、`epsilon` 重新计算全部样本，并与 `imu_normalized` 逐元素比较。

### 测试结果

- 成功生成 `8000 × 6` 的 `imu_normalized`。
- 四个 gait 的六个通道在拟合数据上的 Q1/Q99 均接近 `-1/+1`。
- 输出中的全部原始字段与 `normal_total.npz` 逐元素一致。
- 使用保存参数复算的结果与输出一致，没有 NaN/Inf。
- 归一化结果存在超出 `[-1,1]` 的值，确认没有进行裁剪。
- 总归一化数据可正常绘图，四个 episode 边界正确。
- 已有输出防误覆盖测试通过。

### 当前状态

正常数据的 gait-conditioned 鲁棒归一化已完成并通过测试。

### 下一步

扩充不同速度和转向命令下的正常数据，重新拟合更具代表性的最终归一化参数；之后再进入扰动采集环境设计。

## 2026-10-07 修改记录（六）

### 当前目标

允许通过命令行调节归一化使用的上下分位数，便于比较不同尺度设置。

### 修改内容

- `normalization.py` 新增 `--low-quantile` 和 `--high-quantile` 参数，默认仍为 `0.01/0.99`。
- 增加参数范围与顺序检查，要求 `0 < low < high < 1`。
- 输出文件新增标量字段 `quantile_low`、`quantile_high`，记录生成数据时实际使用的参数。
- 终端统计信息根据当前参数显示对应的归一化分位点。

### 修改原因

不同分位范围会改变 scale 和归一化波形幅值，需要支持重复实验并记录每份数据的生成参数。

### 测试方法

```bash
python -m py_compile dataset/scripts/normalization.py
python dataset/scripts/normalization.py \
  --low-quantile 0.02 \
  --high-quantile 0.98 \
  --output /tmp/normal_normalized_q02_q98.npz
```

同时测试 `low=0.99, high=0.01` 的非法组合。

### 测试结果

- `0.02/0.98` 成功生成 `8000 × 6` 归一化数据。
- 输出正确保存 `quantile_low=0.02`、`quantile_high=0.98`。
- 四个 gait 的对应上下分位点均接近 `-1/+1`，无 NaN/Inf。
- 非法分位数组合能够在处理数据前被拒绝。
- 原 `normal_normalized.npz` 未被覆盖。

### 当前状态

归一化分位数现已支持安全调节和结果追溯。

### 下一步

使用不同输出文件比较候选分位范围；最终参数应结合正常误报率和扰动可分性确定，而不是只根据波形外观选择。

## 2026-10-07 修改记录（七）

### 当前目标

允许使用完整最小值/最大值范围归一化，并将其设为默认配置。

### 修改内容

- `--low-quantile` 默认值由 `0.01` 改为 `0.0`。
- `--high-quantile` 默认值由 `0.99` 改为 `1.0`。
- 合法范围改为 `0 <= low < high <= 1`，允许使用分位数端点。
- 终端输出使用完整的 `normalized quantile` 描述，避免把 `Q1` 与第一百分位混淆。

### 修改原因

开发者需要以每个 gait、每个通道的实际最小值和最大值作为默认归一化边界，同时保留命令行参数用于后续实验。

### 测试方法

```bash
python -m py_compile dataset/scripts/normalization.py
python dataset/scripts/normalization.py \
  --output /tmp/normal_normalized_default_minmax.npz
```

另外测试小于 0、大于 1，以及上下分位数相等的非法参数。

### 测试结果

- 无需指定分位数时正确使用 `0.0/1.0`。
- 成功生成 `8000 × 6` 归一化数据，无 NaN/Inf。
- 四个 gait 的六个通道最小值/最大值均接近 `-1/+1`。
- 输出正确保存 `quantile_low=0.0`、`quantile_high=1.0`。
- 越界和相等分位数均能被正确拒绝。
- 原项目内的归一化数据文件未被覆盖。

### 当前状态

完整 min/max 归一化已经成为默认配置，分位数仍可通过命令行调节。

### 下一步

如需更新项目内的 `normal_normalized.npz`，由开发者确认后使用 `--overwrite` 重新生成并绘图检查。

## 2026-10-07 修改记录（八）

### 当前目标

升级正常 episode 格式，在 200 Hz 下保存与 IMU 严格时间对齐的运动命令。

### 修改内容

- 修改 `dataset/scripts/collect.py`，新增 `commands[T,4]` 字段。
- 命令通道顺序固定为 `vx_command`、`vy_command`、`yaw_rate_command`、`gait_frequency`。
- 在每次 50 Hz 控制步执行前保存实际命令，并重复到随后四个 200 Hz 物理步样本。
- 增加 commands shape、有限值和终端输出检查。
- 按开发者要求删除 `dataset/raw/normal_gait` 下 12 个旧格式数据与图片文件，保留目录结构。

### 修改原因

Detector 需要利用时间对齐的运动命令区分正常主动运动和外部扰动；旧数据没有命令字段，不再进入最终训练集。

### 测试方法

```bash
python -m py_compile \
  dataset/scripts/collect.py \
  dataset/scripts/imu_collection_env.py

TORCH_EXTENSIONS_DIR=/tmp/wtw_torch_extensions \
conda run -n gym python dataset/scripts/collect.py \
  --gait trotting \
  --warmup-steps 50 \
  --steps 20 \
  --headless \
  --sim-device cpu \
  --output /tmp/wtw_trotting_commands_smoke.npz
```

### 测试结果

- 成功生成 `imu_raw[80,6]` 和 `commands[80,4]`。
- commands 内容为 `[0.5, 0.0, 0.0, 3.0]`，类型为 float32。
- 每个 50 Hz 控制步的命令正确对应连续四条 IMU 样本。
- 时间戳间隔保持 `5000 us`，采样率为 200 Hz。
- 新 episode 共九个统一字段，没有 NaN/Inf。
- 旧格式 normal_gait 文件已全部清空。

### 当前状态

固定命令下的 IMU-command 时间对齐采集已完成并通过 CPU 冒烟测试；正式 CUDA 路径仍需开发者运行确认。

### 下一步

先在 CUDA 可视化环境采集一份固定 trotting 新格式数据并检查；通过后再实现连续随机命令调度。`total.py` 和 `normalization.py` 尚未适配 commands，在适配前不要用于新数据。

## 2026-10-07 修改记录（九）

### 当前目标

让可视化脚本同时显示新 episode 中与 IMU 时间对齐的运动命令。

### 修改内容

- `plot.py` 读取并校验可选的 `commands[T,4]` 字段。
- 检测到 commands 时自动增加第三个子图。
- 第三个子图显示 `vx_command`、`vy_command`、`yaw_rate_command` 和 `gait_frequency`。
- 终端摘要增加四个命令通道的最小值和最大值。
- 命令图沿用 episode 分段绘制、边界竖线和状态背景标记。
- 保持对不含 commands 的旧格式文件兼容。

### 修改原因

需要直观确认每个控制命令的时间变化，并与 200 Hz IMU 响应在同一时间轴上比较。

### 测试方法

```bash
python -m py_compile dataset/scripts/plot.py
conda run -n gym python dataset/scripts/plot.py \
  /tmp/wtw_trotting_commands_smoke.npz \
  --output /tmp/wtw_trotting_commands_smoke.png
```

### 测试结果

- 成功读取 `commands[80,4]` 并生成三行子图。
- 命令图正确显示固定水平线：`vx=0.5`、`vy=0`、`yaw_rate=0`、`gait_frequency=3.0`。
- IMU、命令共用相同的 0.395 秒时间轴，采样率识别为 200 Hz。
- IMU 和 commands 均通过 shape 与有限值检查。
- 不含 commands 的旧格式 `/tmp/wtw_imu_trot_short.npz` 回归绘图通过，仍生成原两行 IMU 图。

### 当前状态

新格式 episode 已可同时可视化 IMU 和运动命令。

### 下一步

开发者完成正式 trotting 采集后，使用 `plot.py dataset/raw/normal_gait/trotting` 检查最新数据；确认无误后进入随机命令调度。

## 2026-10-07 修改记录（十）

### 当前目标

为固定 gait 正常采集增加连续随机运动命令，先验证 trotting 的命令调度、平滑过渡和 200 Hz 对齐。

### 修改内容

- `collect.py` 新增 `--command-mode fixed|random`，默认保持 fixed。
- 随机模式支持独立配置 vx、vy、yaw_rate、gait_frequency 范围。
- 默认范围来自实际 checkpoint：`vx[-1,1]`、`vy[-0.6,0.6]`、`yaw[-1,1]`、步频 `[2,4]`。
- 新增随机种子、目标保持时间和线性过渡时间参数。
- 默认目标保持 1～3 秒，过渡时间 0.5 秒。
- 禁止设置超过 checkpoint 训练范围的随机命令。
- 随机模式下检查 warmup 初始命令位于随机范围，避免第一次过渡越界。
- 每个控制周期生成一条命令，并继续严格重复到对应四个 IMU 物理步。

### 修改原因

离散固定工况无法覆盖连续运动命令空间。随机采样可覆盖更多命令组合，平滑过渡则用于模拟正常命令变化并避免无意义的瞬时冲击。

### 测试方法

```bash
TORCH_EXTENSIONS_DIR=/tmp/wtw_torch_extensions \
conda run -n gym python dataset/scripts/collect.py \
  --gait trotting \
  --command-mode random \
  --command-seed 7 \
  --vx 0.0 \
  --vx-range -0.2 0.2 \
  --vy-range -0.1 0.1 \
  --yaw-rate-range -0.2 0.2 \
  --gait-frequency-range 2.8 3.2 \
  --command-hold-seconds 0.2 0.3 \
  --command-transition-seconds 0.1 \
  --warmup-steps 50 \
  --steps 100 \
  --headless \
  --sim-device cpu \
  --output /tmp/wtw_trotting_random_commands_smoke.npz
```

### 测试结果

- 成功生成 `imu_raw[400,6]` 和 `commands[400,4]`，采样率 200 Hz。
- 每个控制命令严格重复到连续四条 IMU 样本。
- 100 个控制周期内检测到 29 次逐周期命令变化，包含过渡段和保持段。
- 四个命令通道全部位于测试指定范围内。
- 最大单控制步变化量约为 `[0.0456, 0.0357, 0.0453, 0.0468]`，平滑过渡生效。
- 命令与 IMU 三行可视化正常，机器人未 reset。
- 初始命令超出自定义随机范围时能够在启动仿真前报错。

### 当前状态

随机命令调度已在 CPU trotting 冒烟测试中通过；完整 checkpoint 范围和 CUDA actuator network 尚待开发者正式验证。

### 下一步

先在 CUDA 可视化环境使用保守随机范围采集 10 秒 trotting；确认机器人稳定和命令曲线正确后，再逐步扩大到完整训练范围。

## 2026-10-07 修改记录（十一）

### 当前目标

验收开发者采集的 trotting 新格式数据，并让 `total.py` 合并时完整保留 commands。

### 修改内容

- 检查一份固定命令和一份随机命令 trotting episode。
- `total.py` 的必需字段新增 `commands`。
- 合并前校验 `commands[T,4]` shape 与 NaN/Inf。
- 合并输出和终端摘要增加 commands shape。

### 修改原因

当前决定先跑通 trotting 单步态检测流程。新数据中的运动命令是 Detector 的时序输入，合并阶段不能丢失。

### 测试方法

```bash
python -m py_compile dataset/scripts/total.py
python dataset/scripts/total.py
conda run -n gym python dataset/scripts/plot.py \
  dataset/raw/normal_gait/total_gait_data/normal_total.npz
```

另外逐字段比较合并文件和两份源 episode，并检查 command 的 4:1 对齐及 episode 边界。

### 测试结果

- 两份 episode 均为 2000 条、200 Hz、约 10 秒，九字段完整且无 NaN/Inf。
- 固定 episode 使用 `[0.5,0,0,3.0]`，随机 episode 检测到 99 次逐控制周期命令变化。
- 成功合并为 `imu_raw[4000,6]`、`commands[4000,4]`，共 20 秒。
- 九个字段与源文件逐元素一致。
- command 仍严格每四条 IMU 对应一个控制命令。
- episode 边界位于索引 2000，合并可视化正确。

### 当前状态

trotting 正常数据的采集、命令对齐、可视化和合并流程已经跑通。

### 下一步

将 `normalization.py` 改为支持当前只有 trotting 的数据，并分别生成 `imu_normalized` 与 `commands_normalized`；其他 gait 暂不要求存在。

## 2026-10-07 修改记录（十二）

### 当前目标

在只存在 trotting 数据时完成 IMU 与运动命令归一化，为单步态 Detector 流程提供输入。

### 修改内容

- `normalization.py` 必需字段新增 `commands[T,4]`。
- IMU 只拟合数据中实际存在的 gait，不再要求四种 gait 同时存在。
- 继续保留 `center[4,6]`、`scale[4,6]` 固定结构。
- 新增 `fitted_gait_mask[4]`，当前只有 gait_id=1 标记为已拟合。
- 未出现 gait 使用 center=0、scale=1 的恒等占位参数，但不允许被当作有效 baseline。
- 新增 `commands_normalized[T,4]`。
- command 按 checkpoint 固定范围归一化：center=`[0,0,0,3]`，scale=`[1,0.6,1,1]`。
- command 超出 checkpoint 训练范围时直接报错。
- `plot.py --field imu_normalized` 自动绘制 `commands_normalized`，而不是原始命令。

### 修改原因

当前先跑通 trotting 单步态流程。IMU baseline 依赖 gait，而命令通道必须保持跨数据集一致的物理含义，因此二者采用不同归一化方式。

### 测试方法

```bash
python -m py_compile \
  dataset/scripts/normalization.py \
  dataset/scripts/plot.py

python dataset/scripts/normalization.py

conda run -n gym python dataset/scripts/plot.py \
  dataset/raw/normal_gait/normalization_data/normal_normalized.npz \
  --field imu_normalized
```

另外使用输出中保存的参数重新计算全部 IMU 和 command，并逐元素比较。

### 测试结果

- 成功生成 `imu_normalized[4000,6]` 和 `commands_normalized[4000,4]`。
- `fitted_gait_ids=[1]`，mask 为 `[False,True,False,False]`。
- trotting 六个 IMU 通道的最小值/最大值均接近 `-1/+1`。
- command 固定范围映射结果有限且位于当前采集覆盖区间内。
- 全部原始九字段与 `normal_total.npz` 逐元素一致。
- IMU 和 command 使用保存参数复算的结果与输出一致。
- 归一化 IMU、归一化 command 和两个 episode 边界均可正确绘制。

### 当前状态

trotting 正常数据的采集、合并、IMU 归一化和 command 归一化流程已完成。

### 下一步

设计 trotting 扰动采集：确定扰动施加方式、时间标签范围和恢复期 mask，再实现最小外力扰动 episode。

## 2026-10-07 修改记录（十三）

### 当前目标

先跑通 trotting 单步态的最小扰动采集：在 base 质心施加一段侧向外力，并让扰动标签、恢复期和 200 Hz IMU 严格对齐。

### 修改内容

- 只修改 `dataset/scripts/imu_collection_env.py` 与 `dataset/scripts/collect.py`，未修改 WTW 原项目文件。
- 外挂环境新增定时 base 质心外力，输入为机体系 `[F_x,F_y,F_z]`，每个物理步转换到环境坐标系后施加。
- 采集器新增 `--dataset-type normal|disturbance`。
- 新增扰动开始时间、持续时间、恢复时间及 y 轴外力参数。
- 扰动 episode 默认写入 `dataset/raw/abnormal_gait/<gait>/`。
- 施力期间 `detector_target=1` 且参与训练。
- 施力后的恢复期 `settling=True`、`loss_mask=False`、`detector_target=0`。
- 正常模式保持原有九字段格式和原有输出目录不变。

### 修改原因

当前只验证二分类 Detector 的完整流程，不区分扰动类型。使用真实物理外力可以让 IMU 响应由 PhysX 动力学产生，同时保持正常和异常 episode 的数据格式完全一致。

### 测试方法

```bash
python -m py_compile \
  dataset/scripts/collect.py \
  dataset/scripts/imu_collection_env.py

TORCH_EXTENSIONS_DIR=/tmp/wtw_torch_extensions conda run -n gym \
python dataset/scripts/collect.py \
  --dataset-type disturbance \
  --gait trotting \
  --command-mode fixed \
  --warmup-steps 50 \
  --steps 100 \
  --disturbance-start-seconds 0.8 \
  --disturbance-duration-seconds 0.1 \
  --disturbance-recovery-seconds 0.3 \
  --disturbance-force-y 60 \
  --headless \
  --sim-device cpu \
  --output /tmp/trotting_disturbance_smoke.npz
```

另外运行短 normal episode 回归测试，并用 `plot.py` 绘制扰动 episode。

### 测试结果

- 扰动 episode 共 400 帧，采样率 200 Hz，全部 IMU 有限且时间间隔恒为 5000 us。
- `detector_target=1` 精确位于索引 160--179，共 20 帧，对应 0.1 秒。
- `settling=True` 和 `loss_mask=False` 精确位于索引 180--239，共 60 帧，对应 0.3 秒。
- 扰动与恢复期不重叠。
- 施力前 20 帧的 `a_y` 均值约为 0.20 m/s²，施力期间约为 4.61 m/s²，证明外力已进入物理仿真。
- `plot.py` 可正确显示红色扰动区、蓝色恢复区和灰色忽略区。
- normal 回归 episode 中扰动、恢复期均为 0，原采集模式未被破坏。

### 当前状态

固定 trotting、固定运动命令、单次侧向外力的最小异常数据采集流程已经跑通。

### 下一步

在 CUDA 可视化环境中采集一份正式的 10 秒 trotting 扰动 episode，人工确认机器人受力表现和曲线后，再决定是否批量扩充左右方向与不同外力强度。

## 2026-10-07 修改记录（十四）

### 当前目标

合并 trotting 扰动 episode，并严格复用正常数据的归一化参数生成异常归一化数据。

### 修改内容

- 使用 `total.py --category abnormal_gait` 合并扰动 episode。
- 生成 `dataset/raw/abnormal_gait/total_gait_data/abnormal_total.npz`。
- `normalization.py` 新增 `--reference-normalized` 参数。
- 未指定参考文件时保持原行为：从正常稳定数据拟合参数。
- 指定参考文件时只读取其中的 IMU 与 command 归一化参数，不从当前输入重新拟合。
- 对参考文件的字段、shape、scale、epsilon 和 gait 覆盖范围进行校验。
- 生成 `dataset/raw/abnormal_gait/normalization_data/abnormal_normalized.npz`。

### 修改原因

扰动数据如果自行拟合中心和尺度，会把异常幅值部分吸收到 baseline 中。正常与扰动必须使用同一套正常基线参数，才能让归一化后的偏离程度具有一致含义。

### 测试方法

```bash
python dataset/scripts/total.py --category abnormal_gait

python dataset/scripts/normalization.py \
  --input dataset/raw/abnormal_gait/total_gait_data/abnormal_total.npz \
  --output dataset/raw/abnormal_gait/normalization_data/abnormal_normalized.npz \
  --reference-normalized dataset/raw/normal_gait/normalization_data/normal_normalized.npz

python dataset/scripts/plot.py \
  dataset/raw/abnormal_gait/normalization_data/abnormal_normalized.npz \
  --field imu_normalized
```

另外逐字段比较正常与异常归一化文件，并比较异常 total 与 normalized 中的九个原始字段。

### 测试结果

- 合并结果包含 1 个 episode、2000 帧、200 Hz、约 10 秒。
- 正常与异常归一化文件的字段集合完全相同。
- `center`、`scale`、`fitted_gait_mask`、command 参数、epsilon 和分位数参数逐元素相同。
- 异常 total 的九个原始字段在归一化后逐元素保持不变。
- 归一化结果不含 NaN 或 Inf，并可正常绘图。
- 扰动标签保持为 20 帧，恢复期与 mask 保持为 100 帧。

### 当前状态

trotting 正常与扰动数据均已完成采集、合并和归一化，且二者具有完全相同的文件字段结构。

### 下一步

实现统一窗口数据加载器：由 `gait_id` 生成 `gait_one_hot[4]`，拼接为逐帧 14 维特征，并按 episode 边界生成长度 100、步长 4 的窗口。

## 2026-10-07 修改记录（十五）

### 当前目标

实现正常与扰动数据共用的 Detector 滑动窗口加载器。

### 修改内容

- 新增 `dataset/scripts/window_dataset.py`。
- 读取 `imu_normalized[6]`、`commands_normalized[4]` 和 `gait_id`。
- 将 `gait_id` 转换为 `gait_one_hot[4]`。
- 拼接得到逐帧 `features[T,14]`。
- 以 100 帧窗口、4 帧步长按 episode 分段生成窗口。
- 标签和 `loss_mask` 均取窗口最后一帧，表示当前时刻检测结果。
- Dataset 输出 TCN 所需的 `[14,100]`，DataLoader 输出 `[B,14,100]`。
- 保留恢复期窗口及其 `loss_mask=False`，供训练阶段执行 masked loss。
- 增加字段、shape、有限值、gait ID、二分类标签和 episode 连续性校验。

### 修改原因

Detector 在 50 Hz 时刻使用最近 0.5 秒的 200 Hz 数据做当前扰动判断，因此每个窗口需要 100 帧，并每收到 4 帧新 IMU 推理一次。窗口不能跨越两个 episode。

### 测试方法

```bash
conda run -n gym python -m py_compile dataset/scripts/window_dataset.py
conda run -n gym python dataset/scripts/window_dataset.py
```

另外逐窗口检查 one-hot、shape、末端标签和 episode 边界。

### 测试结果

- 输入2个归一化文件，共3个 episode。
- 共生成1428个窗口，其中有效窗口1403个、忽略窗口25个。
- 0.1秒扰动在4帧步长下生成5个有效正样本窗口。
- 当前 trotting 的 one-hot 唯一值为 `[0,1,0,0]`，每行和均为1。
- 单窗口 shape 为 `[14,100]`。
- batch size 32 时 DataLoader shape 为 `[32,14,100]`。
- 所有窗口末端标签与原数据一致，且没有窗口跨越 episode。

### 当前状态

从统一归一化数据到 TCN 输入窗口的数据链路已经跑通。

### 下一步

在训练前先设计 episode 级训练集/验证集划分，并解决当前只有5个正样本窗口、无法可靠训练和验证的问题。

## 2026-10-07 修改记录（十六）

### 当前目标

实现最小因果 TCN 训练闭环，并用 episode 级验证检查当前数据是否足以泛化。

### 修改内容

- 新增 `detector/tcn.py`。
- TCN 使用5个因果残差卷积块，dilation 为 `1,2,4,8,16`。
- 输入为 `[B,14,100]`，输出每个窗口的一个二分类 logit。
- 新增 `detector/train.py`。
- 正常与扰动 episode 分别进行训练集/验证集划分，禁止窗口随机泄漏。
- 使用 `BCEWithLogitsLoss(reduction="none")`、训练集正样本权重和 `loss_mask`。
- 验证输出 precision、recall、F1、FPR 和混淆矩阵。
- checkpoint 同时保存模型结构、权重、窗口参数、阈值、episode 划分和正常归一化参数。

### 修改原因

当前目标是先验证从 `[100,14]` 窗口到扰动概率的完整训练链路。episode 级划分能够真实暴露模型是否只记住某一次扰动，而不会因高度重叠窗口泄漏得到虚假高分。

### 测试方法

```bash
conda run -n gym python -m py_compile detector/tcn.py detector/train.py

conda run -n gym python detector/train.py \
  --device cpu \
  --epochs 30 \
  --batch-size 64
```

训练后重新加载 checkpoint，恢复模型并执行 `[1,14,100]` 前向推理。

### 测试结果

- 模型参数量为23393，时间感受野为125帧，能够覆盖100帧窗口。
- 训练集与验证集各包含1个正常、1个扰动 episode。
- 两边各有922个有效负窗口和5个有效正窗口。
- 训练集正样本权重为184.4。
- 训练 loss 从1.5069下降到接近0，说明训练链路和反向传播正常。
- 最佳验证 loss 为1.5471，出现在第1个 epoch。
- 验证集5个正窗口始终没有检出，recall=0。
- 当前训练扰动和验证扰动方向相反，结果说明模型只记住了单个方向/episode，数据不足以支撑泛化。
- checkpoint 已保存到 `detector/checkpoints/trotting_tcn.pt`，可成功恢复，输出 shape 为 `[1]` 且数值有限。

### 当前状态

数据加载、episode 划分、TCN训练、masked loss、验证指标和checkpoint保存的端到端训练流程已跑通；当前 checkpoint 仅是流程产物，不能用于实际检测。

### 下一步

为训练集和验证集分别补充正、负两个方向的多个扰动 episode，并覆盖不同扰动时刻或运动命令；随后重新进行 episode 级训练与验证。

## 2026-10-07 修改记录（十七）

### 当前目标

让 `collect.py` 一次运行能够批量生成多个独立 episode，并随机化扰动强度和开始时间、平衡左右方向。

### 修改内容

- `collect.py` 新增 `--episodes` 批量数量参数。
- 新增 `--disturbance-force-range MIN MAX`。
- 新增 `--disturbance-start-range MIN MAX`。
- 新增 `--disturbance-direction fixed|balanced|random`。
- 新增独立的 `--disturbance-seed`。
- 新增 `--output-dir`，用于将批量测试输出到指定目录。
- `balanced` 模式按照 episode 正负交替，保证方向数量平衡。
- 每个 episode 都重新 reset、warmup、生成命令调度器并单独保存。
- random command 模式为每个 episode 使用不同但可复现的 command seed。
- 保持原有单 episode 参数和九字段 `.npz` 格式兼容。
- 批量结束后输出 episode 数和有效数据总时长。

### 修改原因

训练与验证必须按 episode 划分，因此10分钟数据不应保存为一个超长文件。独立 episode 既能避免窗口泄漏，也便于某个 episode 失败时单独排查和剔除。

### 测试方法

```bash
python -m py_compile dataset/scripts/collect.py

TORCH_EXTENSIONS_DIR=/tmp/wtw_torch_extensions conda run -n gym \
python dataset/scripts/collect.py \
  --dataset-type disturbance \
  --gait trotting \
  --episodes 2 \
  --steps 100 \
  --warmup-steps 10 \
  --disturbance-force-range 40 60 \
  --disturbance-start-range 0.4 0.8 \
  --disturbance-direction balanced \
  --disturbance-duration-seconds 0.1 \
  --disturbance-recovery-seconds 0.2 \
  --disturbance-seed 7 \
  --headless \
  --sim-device cpu \
  --output-dir /tmp/wtw_batch_smoke_20261007
```

### 测试结果

- 一次进程成功生成2个独立 `.npz`，episode ID 不重复。
- 每个文件为400帧、200 Hz，时间间隔恒为5000 us。
- 两个 episode 分别采样到 `+52.50 N` 和 `-55.51 N`。
- 两者各有20帧扰动标签和40帧恢复期标签。
- 施力期间 `a_y` 均值分别约为 `+3.53` 和 `-3.16 m/s²`，方向正确。
- 两个文件均保持统一九字段格式。
- 批量摘要正确报告2个 episode 和0.067分钟有效数据。

### 当前状态

批量扰动采集模式已经实现并通过短CPU冒烟测试。正式数据仍应使用CUDA和 actuator network 配置采集。

### 下一步

在CUDA环境采集10分钟正式 trotting 扰动数据：保存为60个10秒 episode，随后合并、复用正常参数归一化并重新训练。

## 2026-10-07 修改记录（十八）

### 当前目标

修复长批量采集中第二个 episode 偶发的 IMU 时间戳非5000 us问题，并支持从已完成序号断点续采。

### 问题原因

原时间戳使用 `累计物理步数 × 浮点 physics_dt` 计算。批量运行后累计步数增大，float dt 的微小表示误差可能使相邻时间戳偶发变为4999或5001 us，从而触发严格连续性检查。该问题不影响物理采样频率，但会破坏数据时间戳约束。

### 修改内容

- 在外挂环境初始化时将物理周期一次性换算为整数微秒。
- 时间戳改为 `累计物理步数 × 整数周期微秒`。
- `collect.py` 新增 `--episode-index-offset`。
- 断点续采时跳过已经消费的扰动随机参数。
- command seed 和 balanced 正负方向也按照全局 episode 序号继续。

### 测试方法

使用CPU连续采集2个完整10秒 episode，每个 episode 另有100控制步 warmup，总物理步数超过原GPU报错位置。

### 测试结果

- 两个完整 episode 均成功保存，各2000帧。
- 全部相邻时间戳严格为5000 us。
- 扰动标签均为20帧，恢复期均为100帧。
- 随机参数序列为 `+73.40 N @ 5.386 s`、`-51.55 N @ 3.172 s`，balanced方向正确。
- 批量过程未再触发时间戳连续性错误。

### 当前状态

用户已成功保存原批次第1个 episode。该文件通过保存前校验，可以保留；使用 offset=1 续采剩余29个即可完成固定命令的30个 episode。

### 下一步

CUDA续采固定命令剩余29个 episode，然后另采30个 random-command episode。

## 2026-10-07 修改记录（十九）

### 当前目标

验收10分钟批量扰动数据，重新合并、归一化并检查训练/验证 episode 分布。

### 数据验收结果

- 扰动目录共有62个 episode、124000帧、10分20秒有效数据。
- fixed-command episode 32个，random-command episode 30个。
- 正方向31个，负方向31个。
- 全部文件均为2000帧、200 Hz，时间间隔严格为5000 us。
- 每个 episode 均有20帧扰动标签和100帧恢复期。
- 未发现 NaN、Inf、重复/多值 episode ID 或字段 shape 错误。

### 数据处理

- 重新生成 `abnormal_total.npz`：62个 episode、124000帧。
- 复用正常数据参数重新生成 `abnormal_normalized.npz`。
- 随机命令归一化范围保持在 checkpoint 的 `[-1,1]` 范围内。
- 新窗口总数30464，有效窗口28914，忽略窗口1550，正样本窗口310。

### Episode划分检查

- 训练集与验证集各32个 episode，集合无重叠。
- 训练集：1正常、19正方向、12负方向、15 fixed、16 random。
- 验证集：1正常、12正方向、19负方向、17 fixed、14 random。
- 训练集和验证集各有14302个有效负窗口、155个有效正窗口。

### 当前状态

训练集和验证集均覆盖正负扰动方向及 fixed/random 命令，已经具备重新训练最小TCN的条件。

### 下一步

在CUDA上重新运行 `detector/train.py`，检查验证召回率、误报率和最佳checkpoint。

## 2026-10-07 修改记录（二十）

### 当前目标

在完整验证 episode 上输出连续 `Pd(t)`，评估事件检出、检测延迟和误报事件频率。

### 修改内容

- 新增 `detector/evaluate.py`。
- 读取 checkpoint 中记录的验证 episode，避免评估训练数据。
- 批量输出每个50 Hz窗口的扰动概率。
- 统计窗口级 precision、recall、F1、FPR和混淆矩阵。
- 将连续超阈值窗口合并为一次报警事件。
- 统计施力期间事件检出率、恢复期结束前检出率和检测延迟。
- 统计有效正常时段中的每分钟误报事件数。
- 保存 `detector/evaluation/summary.json` 和验证 episode 概率曲线。

### 测试方法

```bash
MPLCONFIGDIR=/tmp/wtw_matplotlib conda run -n gym \
python detector/evaluate.py \
  --device cpu \
  --batch-size 256 \
  --max-plots 6
```

### 测试结果

- 验证集32个 episode，其中31个包含扰动。
- 窗口级 Precision=0.8361、Recall=0.9871、F1=0.9053、FPR=0.002098。
- 混淆矩阵为 TP=153、FP=30、TN=14272、FN=2。
- 31/31个扰动 episode 均在施力期间检出，事件级 Recall=1.0。
- 平均检测延迟8.55 ms，中位数5 ms，最大25 ms。
- 4.767分钟有效正常时段中出现26次单阈值误报事件，即5.454次/分钟。
- 概率曲线能够正确显示扰动区、恢复期和 `Pd(t)`。

### 当前状态

TCN对验证扰动的事件检出和延迟表现良好，但0.5单阈值的误报事件频率过高，尚不能直接驱动 Reflex。

### 下一步

基于验证集概率设计去抖和双阈值迟滞状态机，并比较不同阈值组合下的事件召回、检测延迟和每分钟误报次数。

## 2026-10-07 修改记录（二十一）

### 当前目标

提供一个脚本，对任意单个统一格式数据集执行预测，并在原有IMU图上叠加模型检测结果。

### 修改内容

- 新增 `detector/predict.py`。
- 从 checkpoint 读取正常 IMU/command 归一化参数，不依赖输入文件提前归一化。
- 由原始 `imu_raw`、`commands`、`gait_id` 构造14维特征。
- 按 episode 边界生成100帧窗口，以4帧步长执行50 Hz预测。
- 输出独立预测文件，包含时间戳、窗口索引、episode ID、`Pd`、0/1检测结果、阈值及可用真值。
- 若输入带标签，自动打印 precision、recall、F1和混淆矩阵。
- `--plot` 自动调用现有绘图逻辑生成四面板叠加图。
- `plot.py` 新增 `--prediction`，在原IMU、角速度和command图下增加 `Pd(t)`、阈值和 detected 面板。
- 预测产物统一写入 `detector/predictions/`，避免污染 raw episode 目录并影响 `total.py`。

### 测试方法

```bash
MPLCONFIGDIR=/tmp/wtw_matplotlib conda run -n gym \
python detector/predict.py \
  dataset/raw/abnormal_gait/trotting/episode_1791370304508388.npz \
  --device cpu \
  --plot
```

### 测试结果

- 输入2000帧、10秒 episode，成功生成476个50 Hz预测窗口。
- 检测到7个连续正窗口，合并为1次报警事件。
- 该 episode 的有效窗口指标为 P=1、R=1、F1=1，TP=5、FP=0、TN=446、FN=0。
- 最大 `Pd=1.0`，概率峰值和 detected 区域与真实扰动区对齐。
- 成功生成预测 `.npz` 和四面板检测叠加图。

### 当前状态

已经可以对任意一个统一格式 episode 执行离线预测，并直观看到模型输出是否与真实扰动对应。

### 下一步

用独立于训练/验证集的新采集 episode 做测试；确认预测接口后再加入去抖与双阈值迟滞。

## 2026-10-07 修改记录（二十二）

### 当前目标

将批量采集输出从“每个 episode 一个 `.npz`”调整为“一次运行只输出一个 `.npz`”。

### 修改内容

- `collect.py` 每个 episode 仍独立 reset、warmup、采集和生成唯一 `episode_id`。
- 各 episode 数据先保存在内存中，整次运行结束后沿时间轴统一拼接。
- 一次运行最终只写一个 `collection_<id>_<N>ep.npz`。
- 文件内部通过 `episode_id[T]` 保留全部 episode 边界。
- `--output` 现在允许批量采集直接指定唯一输出文件。
- `--output-dir` 表示唯一批次文件所在目录，不再表示多个 episode 文件目录。
- `total.py` 改为允许一个输入文件包含多个连续 episode。
- `total.py` 逐 episode 校验时间戳、连续区段及跨文件 episode ID 重复。
- 保持原来“一个文件一个 episode”的旧数据兼容。

### 修改原因

开发者希望一次采集任务只管理一个文件。窗口加载仍然依据 `episode_id` 分段，因此合并到一个文件不会造成窗口跨 episode，也不会影响 episode 级训练/验证划分。

### 测试方法

使用CPU一次采集3个短 episode，指定同一个输出目录，然后检查文件数量、字段、episode区段和 `total.py` 校验。

### 测试结果

- 一次运行只生成1个 `.npz`。
- 文件包含1200帧、3个不同 episode ID。
- 每个 episode 恰好400帧、20帧扰动标签、40帧恢复期。
- 每个 episode 内时间间隔严格为5000 us。
- 九字段格式保持不变。
- 新版 `total.py` 成功识别并校验文件中的3个 episode。

### 当前状态

今后的批量采集已经符合“一次运行一个 `.npz`”；现有62个历史原始文件未删除，合并后的 `abnormal_total.npz` 本身已经是单文件。

### 注意事项

由于唯一文件在整次运行结束后才写入，如果采集中途进程异常退出，本次尚未写盘的 episode 会丢失。这是单文件输出相对逐 episode 保存的代价。

## 2026-10-07 修改记录（二十三）

### 当前目标

采集一个独立的5分钟测试集：每10秒一次扰动，并依次测试训练扰动力范围、扩大50%的范围和缩小50%的范围。

### 测试集设计

- 总计30个episode，每个episode保存10秒，合计5分钟有效数据。
- 每个episode第5秒施加一次扰动，因此合并时间轴上相邻扰动间隔10秒。
- 前6个episode（1分钟）使用40至80 N。
- 中间12个episode（2分钟）使用60至120 N。
- 后12个episode（2分钟）使用20至40 N。
- 扰动方向正负交替，共15次正向、15次负向。
- 每个episode独立reset和warmup，窗口通过`episode_id`禁止跨越边界。
- 整次采集仅输出一个`.npz`，并存放在独立的`dataset/test/`目录，避免混入训练数据。

### 修改内容

- `collect.py`增加`--disturbance-force-profile test_5min_three_stage`。
- 三阶段模式校验30个episode、每段10秒、扰动固定从第5秒开始。
- 以`--disturbance-force-range 40 80`为原范围，按episode顺序自动应用1.0、1.5和0.5倍范围。
- `predict.py`增加`--phase-episodes BASE EXPANDED REDUCED`。
- 预测时除整体指标外，可分别打印三种扰动力范围的precision、recall、F1和混淆矩阵。

### 测试结果

- Python语法检查通过。
- 固定随机种子下，6/12/12段的随机力分别全部落入40至80、60至120、20至40 N。
- 正负扰动方向数量为15/15。
- 三阶段episode顺序统计测试通过。
- 使用现有`abnormal_total.npz`进行回归预测成功，共输出29512个50 Hz预测窗口。

## 2026-10-07 修改记录（二十四）

### 当前目标

解决5分钟测试集的叠加图过密问题，将时间轴平均拆成5张图。

### 修改内容

- `plot.py`增加`--parts`参数。
- `predict.py`增加`--plot-parts`参数，并传递给统一绘图接口。
- 多图输出自动添加`part01of05`至`part05of05`后缀。
- 预测只执行一次，预测结果仍只有一个`.npz`，仅PNG按时间范围拆分。

### 测试结果

- Python语法检查通过。
- 使用620秒、62个episode的现有数据和29512个预测窗口完成实际绘图。
- 成功生成5张连续、不重叠的检测叠加图。
- 对5分钟测试集使用5份时，每张图约覆盖60秒。

## 2026-10-08 修改记录（二十五）

### 当前目标

为数据集结构优化讨论准备一份无扰动 demo 和一份带扰动 demo。

### 修改内容

- 新增 `dataset/demo/demo_normal.npz`。
- 新增 `dataset/demo/demo_abnormal.npz`。
- 两份数据均使用 trotting、固定前进命令、10秒有效采集和200 Hz IMU采样。
- 扰动 demo 在第5秒施加60 N侧向力，持续0.1秒，随后标记0.5秒恢复期。

### 测试方法

- 检查两个文件的字段列表、shape、时间戳间隔、gait ID及标签数量。

### 测试结果

- 两个文件均包含统一的9个原始字段和2000帧数据。
- `imu_raw` 为 `[2000, 6]`，`commands` 为 `[2000, 4]`。
- 时间戳间隔均为5000 us，即200 Hz。
- 正常 demo 的扰动和恢复期标签均为0。
- 扰动 demo 包含20帧扰动标签和100帧恢复期标签。

### 当前状态

两个数据格式 demo 已生成并通过结构检查。由于当前执行环境没有可用GPU，本次采用CPU PhysX与P控制器采集，仅用于结构和字段展示，不作为正式训练数据。

## 2026-10-08 修改记录（二十六）

### 当前目标

归一化正常与扰动 demo，并确保二者使用同一套正常基线参数。

### 修改内容

- 使用 `demo_normal.npz` 拟合 trotting 的 IMU `center/scale`。
- 生成 `dataset/demo/demo_normal_normalized.npz`。
- 使用正常 demo 保存的参数转换 `demo_abnormal.npz`，不对扰动数据重新拟合。
- 生成 `dataset/demo/demo_abnormal_normalized.npz`。

### 测试方法

- 比较归一化文件中的原始 IMU 与输入文件。
- 比较正常、扰动归一化文件的 `center/scale/epsilon`。
- 用保存的参数重新计算扰动 IMU，并与 `imu_normalized` 比较。

### 测试结果

- 两份文件均保留原始数据，新增归一化数据和统计参数。
- 两份文件的归一化参数完全相同。
- 正常数据六个通道均约映射到 `[-1, 1]`。
- 扰动数据允许超出 `[-1, 1]`，说明异常幅值没有被裁剪或重新压缩。
- 手工重算结果与文件内 `imu_normalized` 一致。

### 当前状态

两个 demo 已使用同一套正常基线完成归一化并通过校验。

## 2026-10-08 修改记录（二十七）

### 当前目标

完成归一化消融实验第1步：采集并验收正常训练集、扰动训练集和独立测试集。

### 数据集

- `dataset/ablation/raw/train_normal_1min.npz`：1分钟、6个episode。
- `dataset/ablation/raw/train_abnormal_4min.npz`：4分钟、24个episode。
- `dataset/ablation/raw/test_5min.npz`：5分钟、30个episode。

### 测试方法

- 检查统一9字段、数组shape、有限数值和gait ID。
- 按episode检查帧数及5000 us时间戳间隔。
- 检查扰动标签、恢复期标签、loss mask和随机命令范围。
- 检查三份文件的episode ID是否互不重复。

### 测试结果

- 三份数据分别为12000、48000和60000帧，均为200 Hz。
- 每个episode均为2000帧、10秒，episode数量分别为6、24和30。
- 正常集没有扰动和恢复期标签。
- 两份扰动数据的每个episode均包含20帧扰动和100帧恢复期。
- 随机命令覆盖设定范围内的前后、侧移、转向和步频变化。
- 三份文件的episode ID互不重复，训练集和测试集相互独立。

### 当前限制

当前原始文件没有保存每个episode实际抽样到的扰动力和开始时间，只能通过采集命令、随机种子和episode顺序复现，无法仅从NPZ审计扰动力分布。该问题记为后续数据格式优化项，不影响本轮归一化消融的特征构建。

### 当前状态

消融实验第1步已经通过，可以进入第2步：从完全相同的训练和测试episode构建normalized与raw两条特征支路。

## 2026-10-08 修改记录（二十八）

### 当前目标

完成归一化消融实验第2步：从相同原始数据构建normalized和raw两条训练/测试特征支路。

### 修改内容

- 使用正常训练集拟合归一化参数，生成 `dataset/ablation/normalized/train_normal_1min_normalized.npz`。
- 扰动训练集和测试集复用同一组正常参数，分别生成对应normalized文件。
- `window_dataset.py`增加`feature_mode`：
  - `normalized`使用`imu_normalized + commands_normalized + gait_one_hot`。
  - `raw`使用`imu_raw + commands + gait_one_hot`。
- 命令行增加`--feature-mode normalized|raw`。
- 默认模式仍为`normalized`，保持原有调用兼容。

### 测试方法

- 分别加载两条训练支路并检查窗口、标签及batch shape。
- 逐项比较两条支路的窗口索引、target、loss mask、episode ID和one-hot。
- 对测试集执行相同的一致性检查。

### 测试结果

- 两条训练支路均生成14280个窗口，其中13680个有效、600个忽略、120个有效正样本。
- 两条测试支路均生成14280个窗口。
- 单窗口均为`[14,100]`，batch均为`[B,14,100]`。
- raw与normalized的窗口索引、标签、mask、episode和gait one-hot完全一致。
- 两条支路只有前10维连续特征的尺度不同，满足单变量消融要求。

### 当前状态

第2步已经通过。下一步是在训练脚本中显式记录feature mode，并用相同划分和随机种子分别训练30 epoch。

## 2026-10-08 修改记录（二十九）

### 当前目标

完成归一化消融实验第3步：以相同设置分别训练normalized和raw模型30 epoch。

### 修改内容

- `train.py`增加`--feature-mode normalized|raw`。
- 训练时将feature mode传给窗口数据集，并写入checkpoint。
- normalized checkpoint保存训练正常集的归一化参数。
- raw checkpoint明确保存`normalization=None`，避免后续预测误用归一化。
- 两组均使用相同TCN结构、seed=0、batch size=256、validation fraction=0.2和30 epoch。

### 测试方法

- 两种模式各执行1 epoch冒烟训练。
- 检查checkpoint中的feature mode和归一化参数状态。
- 比较训练/验证episode、模型结构、窗口长度和stride。
- 冒烟测试通过后顺序执行两组正式30 epoch训练。

### 测试结果

- 两组训练/验证episode完全一致：训练24个、验证6个。
- 两组有效训练样本均为10854个负样本和95个正样本。
- 两组有效验证样本均为2706个负样本和25个正样本。
- normalized模型保存为`detector/checkpoints/ablation_tcn_normalized.pt`，最低验证损失为0.257433。
- raw模型保存为`detector/checkpoints/ablation_tcn_raw.pt`，最低验证损失为0.053706。
- 两个checkpoint的TCN结构、窗口`100`、stride`4`及数据划分完全一致。

### 当前状态

第3步已经完成。验证损失暂时显示raw更低，但这不能直接作为消融结论；下一步必须让预测脚本按checkpoint中的feature mode处理同一份独立5分钟测试集，再比较测试指标。

## 2026-10-08 修改记录（三十）

### 当前目标

完成归一化消融实验第4步：两个模型预测同一份独立5分钟测试集并比较指标。

### 修改内容

- `predict.py`读取checkpoint中的`feature_mode`。
- normalized模型使用checkpoint内训练归一化参数，raw模型完全跳过归一化。
- 指标增加Accuracy和Balanced Accuracy，同时保留Precision、Recall、F1与混淆矩阵。
- 两个模型均按6/12/12个episode统计原范围、扩大50%和缩小50%三段结果。
- 新增`detector/evaluation/ablation_normalization_comparison.json`保存对比结果。

### 测试方法

- 两个模型分别预测同一个`dataset/ablation/raw/test_5min.npz`。
- 比较两份预测的时间戳、窗口索引、episode、target和loss mask。
- 检查checkpoint特征模式与实际预测预处理一致。

### 测试结果

- 两组均预测14280个50 Hz窗口，其中13530个有效、750个恢复期窗口被忽略。
- normalized：Accuracy=0.9891，Balanced Accuracy=0.8066，P=0.5082，R=0.6200，F1=0.5586，FP=90。
- raw：Accuracy=0.9827，Balanced Accuracy=0.8594，P=0.3618，R=0.7333，F1=0.4846，FP=194。
- normalized将误报窗口从194降到90，整体F1提高0.0740，单阈值报警事件从151降到82。
- raw召回率更高，但以更多误报为代价。
- 两份预测的窗口、episode、标签和mask完全一致，满足公平比较要求。

### 当前结论

在本次单seed、单次训练、0.5固定阈值实验中，归一化改善了整体Accuracy、Precision和F1，并显著减少误报；raw模型获得更高Recall与Balanced Accuracy。归一化的价值主要体现为抑制误报和提高预测质量，而不是提高所有指标。该结果仍需多seed重复实验和阈值扫描后才能形成稳健结论。

## 2026-10-08 修改记录（三十一）

### 当前目标

将两个消融模型的检测阈值从0.5调整为0.8并重新测试。

### 修改内容

- 不重新训练模型，不改变`Pd(t)`，仅在预测时使用`--threshold 0.8`。
- 两个模型继续使用同一份5分钟测试集和6/12/12三阶段划分。
- 新增`detector/evaluation/ablation_threshold_08_comparison.json`。
- 新预测文件使用`threshold_08`命名，不覆盖0.5阈值结果。

### 测试结果

- normalized：Accuracy=0.9919，P=0.6429，R=0.6000，F1=0.6207，FP=50，报警事件=57。
- raw：Accuracy=0.9886，P=0.4907，R=0.7000，F1=0.5769，FP=109，报警事件=93。
- 相比0.5阈值，两者的误报和报警事件均减少，Precision与F1提高，Recall降低。
- 0.8阈值下normalized仍具有更高Accuracy、Precision和F1；raw仍具有更高Recall与Balanced Accuracy。

### 当前状态

0.8固定阈值对比已经完成。该阈值来自人工试验而非验证集选择，因此只能用于观察阈值效应，不能作为最终无偏测试结论。

## 2026-10-08 修改记录（三十二）

### 当前目标

按照“每次扰动只要有一个扰动预测点命中就算成功”的规则重新计算事件级指标。

### 事件定义

- 测试集包含30次真实扰动，每个episode一次。
- 对每次扰动，检查其0.1秒真实施力区间内的5个50 Hz预测点。
- 至少一个点`detected=1`即认为该扰动事件成功检出。
- 首次命中时间减去施力开始时间定义为检测延迟。

### 测试结果

- threshold=0.8时，normalized和raw均检出23/30次，事件级Recall均为0.7667。
- 40至80 N阶段两者均检出6/6次。
- 60至120 N阶段两者均检出12/12次。
- 20至40 N阶段两者均检出5/12次，全部7次漏检均来自弱扰动。
- normalized成功事件平均延迟按三阶段分别为48.33、26.67、43.00 ms。
- raw成功事件平均延迟按三阶段分别为21.67、18.33、23.00 ms。

### 当前结论

按事件级成功标准，两种模型的检出次数相同；raw首次响应更快，normalized的窗口级误报更少。窗口级Recall描述扰动区间内命中了多少预测点，事件级Recall描述30次扰动中成功响应了多少次，两者含义不同。

## 2026-10-08 修改记录（三十三）

### 当前目标

按事件级规则补充threshold=0.5的检出率和检测延迟。

### 测试结果

- normalized检出23/30次，事件级Recall=0.7667。
- raw检出25/30次，事件级Recall=0.8333。
- 40至80 N阶段两者均检出6/6次。
- 60至120 N阶段两者均检出12/12次。
- 20至40 N阶段normalized检出5/12次，raw检出7/12次。
- normalized三阶段平均延迟分别为48.33、26.67、35.00 ms。
- raw三阶段平均延迟分别为21.67、16.67、29.29 ms。

### 当前结论

阈值从0.8降到0.5后，normalized的事件检出次数没有增加；raw额外检出2次弱扰动，但伴随更多窗口误报和更多报警段。最终阈值选择需要在弱扰动检出与误报频率之间权衡。

## 2026-10-08 修改记录（三十四）

### 当前目标

将normalized和raw消融模型的训练预算提高到100 epoch，并在相同5分钟测试集上重新比较。

### 训练设置

- 两组继续使用相同seed、训练/验证episode、TCN结构、batch size和validation fraction。
- checkpoint仍按最低验证损失保存，而不是保存第100轮。
- 当前执行环境无法访问CUDA，因此实际使用CPU训练；这只影响耗时，不改变实验设置。

### 训练结果

- normalized最低验证损失为0.056373，出现在第46轮。
- raw最低验证损失为0.011748，出现在第88轮。
- normalized后期训练损失继续下降、验证损失升高，表现出明显过拟合。

### 测试结果

- threshold=0.5：normalized F1=0.6135、FP=76；raw F1=0.7020、FP=46。
- threshold=0.8：normalized F1=0.6600、FP=51；raw F1=0.7197、FP=35。
- 四种模型/阈值组合的事件级检出均为24/30。
- 两个模型均检出6/6次40至80 N、12/12次60至120 N、6/12次20至40 N扰动。
- raw在三段中的平均首次检测延迟均不高于normalized。
- 完整结果保存至`detector/evaluation/ablation_100ep_comparison.json`。

### 当前结论

在当前单步态、单seed、同分布数据和100 epoch训练预算下，raw模型的窗口级F1、Precision、误报和响应速度均优于normalized，事件级检出次数相同。因此本实验没有证明当前归一化方案带来收益，反而说明TCN在训练充分时可以直接适应当前原始量纲。该结论不能外推到多步态、跨速度范围、仿真到实机或多seed情形。

## 2026-10-15 修改记录（三十五）

### 当前目标

为后续 TCN 与 GRU 的独立架构对比整理目录边界，避免在同一训练脚本中通过模型参数切换不同网络。

### 修改内容

- 将原 `detector/` 目录重命名为 `tcn_detector/`。
- 同步更新 TCN 训练、预测和评估脚本的默认产物路径。
- 同步更新已保存评估 JSON 中的 checkpoint 路径。
- 为未来 `gru_detector/` 的 checkpoint、预测结果和评估图片补充 `.gitignore` 规则。

### 当前状态

TCN 实现保持不变；后续将在独立的 `gru_detector/` 目录中实现 GRU，并与 TCN 使用相同数据、窗口与评估口径进行对比。

## 2026-10-15 修改记录（三十六）

### 当前目标

实现 GRU 对比实验的第一步：独立定义与现有 TCN 输入输出一致的扰动检测网络。

### 修改内容

- 新增 `gru_detector/gru.py`。
- 输入保持为数据加载器输出的 `[B, 14, 100]`，在网络内部转为 GRU 所需的 `[B, 100, 14]`。
- 使用两层单向 GRU，`hidden_size=48`、`dropout=0.1`，取窗口末端隐藏状态经线性层输出一个 logit。
- 新增参数量统计函数，用于控制 TCN/GRU 架构对比的容量差异。

### 测试方法

```bash
conda run -n gym python -c "..."
```

### 测试结果

- 默认 GRU 参数量为 23377，与当前 TCN 的 23393 基本一致。
- `[3, 14, 100]` 输入可输出 `[3]` logit。
- 二分类损失有限且可完成反向传播。

### 下一步

实现独立的 `gru_detector/train_gru.py`，复用相同滑动窗口、episode 划分、类别权重和训练超参数。

## 2026-10-15 修改记录（三十七）

### 当前目标

补齐独立 GRU 的训练、预测和评估链路，并使用消融实验数据验证完整接口。

### 修改内容

- 新增 `gru_detector/train_gru.py`、`predict_gru.py` 和 `evaluate_gru.py`。
- 三个入口均只加载 `DisturbanceGRU`；GRU 脚本会拒绝非 `model_type=gru` 的 checkpoint，避免误用 TCN 模型。
- GRU checkpoint 保存 `model_type`、模型配置、窗口参数、特征模式、归一化参数和 episode 划分信息。
- 评估脚本按 checkpoint 中保存的 `feature_mode` 重建窗口数据，raw 模型不会误用归一化特征。

### 测试方法

- 使用 `dataset/ablation/raw/train_normal_1min.npz` 与 `train_abnormal_4min.npz` 进行 1 epoch 冒烟训练。
- 使用 `dataset/ablation/raw/test_5min.npz` 执行预测，并按 6/12/12 个 episode 输出三段指标。
- 对训练数据的验证 episode 执行独立事件级评估。

### 测试结果

- 训练、预测、评估脚本均可运行。
- 冒烟训练的 episode 划分与 TCN 消融实验一致：训练 24 个、验证 6 个。
- 5 分钟测试集生成 14280 个 50 Hz 预测窗口。
- 当前环境 GPU 驱动不可通信，PyTorch CUDA device count 为 0；因此尚未启动约 100 epoch 的正式 CPU 训练。

### 下一步

在 GPU 可用时按 TCN 对比设置训练 `ablation_gru_raw_100ep.pt`；若使用 CPU，需要确认可接受约 45 分钟的训练时间。

## 2026-10-15 修改记录（三十八）

### 当前目标

完成 GRU 的 100 epoch 正式训练，并在独立 5 分钟测试集上与 raw-TCN 进行架构对比。

### 训练设置

- 训练数据：`dataset/ablation/raw/train_normal_1min.npz` 与 `train_abnormal_4min.npz`。
- 测试数据：`dataset/ablation/raw/test_5min.npz`。
- 输入为 raw 的 `[14, 100]` 窗口，stride 为 4，即 50 Hz 推理。
- GRU 为两层单向结构，`hidden_size=48`、`dropout=0.1`、参数量 23377。
- 使用相同 seed、episode 划分、类别权重、学习率和 100 epoch 预算；正式训练由可访问 GPU 的本地终端完成。

### 测试结果

- checkpoint 最低验证损失为 0.185513。
- threshold=0.5：窗口级 F1=0.3524、FP=411；事件级检出 27/30，弱扰动（20至40 N）检出 9/12。
- threshold=0.8：窗口级 F1=0.4467、FP=236；事件级检出 25/30，弱扰动检出 7/12。
- 两档阈值均检出全部 6/6 次原范围和 12/12 次扩大范围扰动。
- 完整结果保存至 `gru_detector/evaluation/ablation_gru_raw_100ep_comparison.json`；阈值 0.5 的五段概率叠加图保存在同目录。

### 当前结论

与 raw-TCN 相比，当前 GRU 在 threshold=0.5 下提高了事件级检出率（27/30 对 24/30）并提升弱扰动检出（9/12 对 6/12），但产生显著更多误报窗口和报警事件（FP=411、报警段=304），窗口级 F1 明显低于 raw-TCN 的 0.7020。当前 GRU 更敏感但稳定性不足；后续应优先分析阈值、类别权重、GRU 容量或去抖迟滞，而不是直接宣称 GRU 优于 TCN。

## 2026-10-15 修改记录（三十九）

### 当前目标

在 GRU 正式训练时显示 batch 级进度，便于确认长时间训练没有停滞。

### 修改内容

- `gru_detector/train_gru.py` 的训练与验证 DataLoader 均由 `tqdm` 包装。
- 每个 epoch 显示 `train` 和 `val` 两个 batch 进度条；epoch 结束后仍输出原有 loss、F1 和混淆矩阵。

### 测试结果

- 使用两批最小内存数据验证进度条可创建。
- 训练损失可完成反向传播，4 个有效样本均被正确统计。

### 使用说明

- 通过 `conda run --no-capture-output -n gym ...` 运行，才能实时看到终端进度条。

## 2026-10-15 修改记录（四十）

### 当前目标

完成 GRU 的 raw / normalized 输入消融：在固定网络结构和训练预算下，确认 IMU 与命令归一化对独立 5 分钟测试集的影响。

### 实验设置

- 两组均为两层单向 GRU，`hidden_size=48`、`dropout=0.1`、100 epoch。
- 两组使用相同训练 / 验证 episode 划分、窗口 `[14, 100]`、stride=4、seed 和测试集。
- raw 组直接输入原始 IMU 与命令；normalized 组使用仅由正常训练集拟合得到的归一化参数。
- 窗口级指标排除 `loss_mask=false` 的恢复 / 忽略窗口；事件级检出定义为：一次真实施力区间中至少有一个窗口报警。

### 测试结果

- normalized-GRU 最低验证损失为 `0.266514`。
- threshold=0.5：窗口级 F1=0.3296、FP=372，事件级检出 23/30；原范围 6/6、扩大范围 11/12、弱扰动 6/12。
- threshold=0.8：窗口级 F1=0.4062、FP=211，事件级检出 22/30；原范围 6/6、扩大范围 11/12、弱扰动 5/12。
- 完整结果写入 `gru_detector/evaluation/ablation_gru_normalized_100ep_comparison.json`；阈值 0.5 的五段概率叠加图写入同目录。

### 当前结论

在当前 GRU 配置与训练数据下，归一化确实略降低了误报（0.5：411 到 372；0.8：236 到 211），但同时明显降低了召回率、事件检出率和弱扰动检出率。因此不能据此声称归一化提升了 GRU 效果；更合理的结论是：当前归一化方案与 GRU 的组合需要继续检查统计量、特征尺度和训练配置。

### 下一步

将四组结果（raw / normalized × TCN / GRU）统一按窗口级和事件级指标比较，再决定是否调整归一化方法或网络结构。
