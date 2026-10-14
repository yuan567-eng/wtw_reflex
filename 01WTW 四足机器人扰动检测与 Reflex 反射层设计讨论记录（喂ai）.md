# WTW 四足机器人扰动检测与 Reflex 反射层设计讨论记录

# WTW 四足机器人扰动检测与 Reflex 反射层设计讨论记录



> 主题：基于 WTW（Walk These Ways）的四足机器人扰动检测、Residual Reflex
> 
> Layer 与 PPO 训练方案\\
> 
> 整理时间：2026\-10\-06
> 
> 

---

现有 WTW locomotion policy

对部分突发外部扰动的恢复能力可能仍然不足，因此考虑在原有 WTW

控制器之上增加一个高速 **Reflex Layer（反射层）**。

核心控制形式：

$\mathbf{y}_t
=
\mathbf{y}_t^{WTW}
+
\mathbf{y}_t^{reflex}$

进一步引入门控系数：

$\mathbf{y}_t
=
\mathbf{y}_t^{WTW}
+
\alpha_t\Delta\mathbf{y}_t^{reflex}$

其中

- $\mathbf{y}_t^{WTW}$：WTW 原始 locomotion policy 输出；

- $\mathbf{y}_t^{reflex}$：反射层给出的 residual（残差）action；

- $\alpha_t$：Reflex Gate，决定反射层是否介入以及介入强度。

正常情况下：

$\alpha_t \approx 0$

因此：

$\mathbf{y}_t \approx \mathbf{y}_t\^{WTW}$

发生明显扰动后：

$\alpha_t \rightarrow 1$

Reflex Layer 开始辅助 WTW 进行快速恢复。

因此整个系统可以概括为：

$WTW\ Nominal\ Locomotion
+
Disturbance\ Detector
+
Residual\ Reflex$

---

## Detector部分

### 扰动检测器的定位

最初考虑使用 6 轴 IMU 检测：

- 外部踢击；

- 碰撞；

- 足端突然踩空；

- 腿部被障碍物绊住或撞击；

- 其他瞬态外部扰动。

IMU 输入：

$\mathbf x_t =
a_x,a_y,a_z,\omega_x,\omega_y,\omega_z$

Detector 的主要输出：

$P_d=P(\text{disturbance}\mid X_{t-N:t})$

最重要的认识是：

> Detector 本身不是最终目的，而是 Reflex Layer 的触发器（Reflex Gate）。
> 
> 

即：

$IMU \rightarrow Detector \rightarrow
\alpha_t \rightarrow Reflex$

Detector 回答：

> **什么时候需要反射？**
> 
> 

Reflex Policy 回答：

> **发生扰动以后应该怎么反射？**
> 
> 



---

### IMU 跨步态归一化



不同 gait 下正常 IMU 的幅值和分布不同，因此不能直接使用统一固定阈值。

最初提出：

$x_{\text{consistent}} = \dfrac{x_{\text{gait}}}
{\|x_{\max}-x_{\min}\|}$

进一步改进为每个 IMU channel 单独进行中心化和尺度归一化：

$\tilde{x}_{t,i}
=
\dfrac{x_{t,i}-c_{g,i}}
{s_{g,i}+\epsilon}$

其中：

$c_{g,i} = \dfrac{x^{high}_{g,i}+x^{low}_{g,i}}{2}

s_{g,i} = \dfrac{x^{high}_{g,i}-x^{low}_{g,i}}{2}$

注意：

- 六个 IMU channel 分别归一化；

- 不建议用一个统一 vector norm 对六维信号共同缩放。

    

---

### 使用 Quantile 而不是绝对 Min/Max

正常运动中偶尔也会出现较大的瞬时尖峰。

如果直接使用：

$x_{\min},x_{\max}$

少量异常值会严重扩大 normal range。

因此采用 robust quantile，例如：

$Q_{1\%},\quad Q_{99\%}$

构造：

$c_g= \dfrac{Q_{99,g}+Q_{1,g}}{2}

s_g= \dfrac{Q_{99,g}-Q_{1,g}}{2}$

最终：

$\tilde{x}
=
\dfrac{x-c_g}
{s_g+\epsilon}$

关键原则：

离线估计 Normal Baseline 时极端尾部样本不应该决定 normal range。

在线检测时不要将归一化结果 clip 到：

$-1,1$

例如：

$\tilde{x}=3.5$

恰恰可能是非常重要的 disturbance evidence。

---

### 同一步态下的环境 Domain Shift

即使 gait 不变：

$p(\mathbf x\mid gait)$



环境也不是固定的。

平地、斜坡、碎石、沙地等环境会造成：

$p(\mathbf x\mid gait,environment)$

发生变化。

其中斜坡还会导致重力在 IMU body frame 中的投影发生改变。

如果能够获得姿态估计，可以考虑：

$\mathbf a_{linear} = \mathbf a_{imu} -
R\^\top\mathbf g$

进行重力补偿。

但更一般的问题仍然是环境/context 的变化。

因此采用两部分 baseline：

Offline Gait Prior

预先统计每种 gait 的 robust normal distribution：

$(c_g,s_g)$

Online Context Estimate

在线根据近期正常 IMU 数据估计：

$(c_t^{online},s_t^{online})$

融合：

$c_t = (1-\alpha)c_g + \alpha c_t\^{online}

s_t = (1-\alpha)s_g + \alpha s_t\^{online}$

其中 gait information 只作为：

$\text{Prior}$

而不是绝对 ground truth。

即使上游 gait classifier / gait switching 出现错误，online adaptation

仍然能够提供一定的容错能力。



---

### Fast/Slow Timescale Separation

核心假设：

- 慢变化

例如：

$flat\rightarrow slope$

或者 terrain statistics 缓慢改变。

由：

$Online Context Adaptation$

处理。

- 快变化

例如：

- kick；

- collision；

- sudden support loss；

- snag。

由：

$Causal Disturbance Detector$

处理。

因此可以理解为：

$Disturbance
=
Current\ Dynamic\ Response
-
Expected\ Normal\ Response$

---

### Online Baseline 的 Update Gate



必须防止扰动本身污染 normal baseline。

例如机器人被踢后，如果立即将大幅 IMU response 更新到 baseline：

$s_t\^{online}\uparrow$

那么 detector 可能逐渐认为：

> 被踢也是正常的。
> 
> 

因此设置 update gate。

当：

$P_d<T_{safe}$

且状态持续稳定一定时间时：

$Update Baseline$

当：

$P_d>T_{freeze}$

则：

$(c_t^{online},s_t^{online})
=
(c_{t-1}^{online},s_{t-1}^{online})$

即冻结 baseline。

可进一步研究的 streaming robust estimator 包括：

- Sliding\-window quantile；

- P² online quantile；

- EWMA；

- MAD。

    

---

### Gait Switching 状态机

上游 gait controller 可以提供：

- `gait_id`

- `switching`

在 gait transition 时，IMU

本身可能产生较大的正常瞬态，因此需要避免误判。

建议状态机：

```Plain Text
NORMAL
  ↓
SWITCHING
  ↓
SETTLING
  ↓
NORMAL
```

`switching=1` 时：

- 暂时抑制 disturbance alarm；

- 继续采集 IMU；

- 可以利用该阶段重新初始化/适配 context。

切换结束后不能立即恢复检测，而应该经过：

$T_{settle}$

如果 temporal detector window 为：

$T_w$

初始可以取：

$T_{settle}\ge T_w$

并根据真实残余振荡调整。

---

### Detector 的实时性要求



Detector 必须是 causal：

$P_d(t) = f( x_{t-N+1},\ldots,x_t )$

不能使用未来数据。

总检测延迟：

$T_{detect}
=
T_{sensor}
+
T_{evidence}
+
T_{network}
+
T_{decision}$

例如 IMU：

$f_s=200Hz$

则：

$T_s=5ms$

注意：

> 0\.5 s temporal window 并不意味着 500 ms detection latency。
> 
> 

历史 buffer 始终存在，新的 5 ms sample 到来后即可重新推理。

初步设计目标：

- 明显 kick/collision：20\-\-50 ms；

- 强冲击：20\-\-50 ms；

- leg snag：30\-\-100 ms；

- support loss / 踩空：50\-\-150 ms。

报警可以要求连续 2\-\-4 个 sample 超过阈值，以减少 false positive。

检测延迟指标：

$T_d=t_{alarm}-t_{disturbance}$

建议报告：

- Mean detection latency；

- P95 latency；

- Detection rate；

- False\-positive rate。

    

---

### WTW \+ Reflex 总体控制结构

由于 WTW 对部分强扰动恢复可能不够理想，因此增加一个并行 residual reflex path。

```Plain Text
Command
                       │
                       ↓
                ┌─────────────┐
                │     WTW     │
                └──────┬──────┘
                       │
                    y_WTW
                       │
                       ↓
IMU → Detector → Gate → Σ → Robot
                  ↑    ↑
                  │    │
                  │  y_reflex
                  │    ↑
                  └ Reflex Policy
```

---

### 为什么不推荐纯 0/1 Gate

第一版可以：



$\alpha=

\begin{cases}
0&normal\\
1&disturbance
\end{cases}$



但最终更适合连续 gate：

$\alpha\in[0,1]$

例如：

$\alpha(P_d) =

\begin{cases}
0,&P_d<T_L\\
\dfrac{P_d-T_L}{T_H-T_L},
&T_L\le P_d<T_H\\
1,&P_d\ge T_H
\end{cases}$

这样能够：

- 设置 dead zone；

- 防止轻微 detector noise 不断修改 WTW action；

- 使 Reflex 介入更加平滑。

---

## Reflex 部分

### Reflex 的核心问题

假设 WTW 输出：

$\mathbf{y}\^{WTW} = y_1,\ldots,y_{12}$

Reflex 输出：

$\Delta\mathbf{y}\^{R} =
\Delta y_1,\ldots,\Delta y_{12}$

那么真正的问题变成：

> 当前机器人受到扰动以后，应该给 12 个关节增加多少 residual correction？
> 
> 

这可以采用：

1. 人工 Rule\-Based Reflex；

2. Model\-Based Reflex；

3. Learned Residual Reflex Policy；

4. Teacher/Imitation 等方法。

当前推荐第一版：

$Residual\ Reinforcement\ Learning$

---

### Residual Reflex Policy

冻结原始 WTW：

$a_t\^{WTW} = \pi_{WTW}(o_t)$

Reflex：

$\Delta a_t = \pi_R(o_t\^R)$

最终：

$a_t
=
a_t^{WTW}
+
\alpha_t\Delta a_t$

Reflex 不重新学习 locomotion。

它只学习：

$\text{在 WTW action 基础上应该额外修改多少？}$

因此整个学习空间比重新训练完整 locomotion policy 小很多。

---

### 仿真 Ground Truth 是关键优势

在 Isaac Gym 中，外部扰动由我们主动施加，因此天然知道：

$t_d,\quad \mathbf F_d,\quad
\mathbf r_d,\quad \Delta t_d$

甚至：

$\boldsymbol\tau_d =
\mathbf r_d\times\mathbf F_d$

因此可以同时开发：

- Detector；

- Reflex。

而不需要等 Detector 完成以后才训练 Reflex。

---

### Perfect Detector / Ground\-Truth Gate

第一阶段直接使用 simulator ground truth：

$z_t=

\begin{cases}
1,&disturbance/recovery\\
0,&normal
\end{cases}$

控制：

$a_t
=
a_t^{WTW}
+
z_t\Delta a_t^{R}$

这样首先回答：

> 如果 100% 知道什么时候发生扰动，Reflex 是否真的能够提高 WTW
> 
> 的抗扰能力？
> 
> 

如果：

$WTW+Reflex_{GT}$

仍然没有超过：

$WTW$

说明问题主要在 Reflex，而不是 Detector。

---

### Detector 独立训练

仿真时可以同步记录：

${ IMU, gait, terrain, t_d, F_d, \Delta t_d }$

Detector：

$X_{t-N:t}\^{IMU} \rightarrow P_d$

重点指标：

$Recall,\quad FPR,\quad T_{detect}$

Detector 的目标不只是高分类 accuracy，而是：

$\text{快速、可靠地触发 Reflex}$

---

### 最终合体

Ground\-truth gate：

$z_t$

最终替换为：

$\alpha_t=f(P_d)$

得到：

$a_t
=
a_t^{WTW}
+
\alpha_t\Delta a_t^R$

建议实验：

1. WTW；

2. WTW \+ Reflex \(Ground Truth\)；

3. WTW \+ Reflex \(IMU Detector\)；

4. WTW \+ Reflex \(Always On\)。

第四组可以回答：

> 为什么需要 Detector？Residual policy 永远工作是否也可以？
> 
> 

理想结果：

$WTW < WTW+Reflex_{Detector} \approx WTW+Reflex_{GT}$

并且：

$WTW+Reflex_{AlwaysOn}$

在正常 locomotion 上可能存在额外干扰。

---

### 仿真数据应记录完整 Disturbance Ground Truth

不要只记录：

$disturbance=0/1$

建议保存：

$D_t=
[
flag,
F_x,F_y,F_z,
\tau_x,\tau_y,\tau_z,
position,
duration
]$

即使 V1 Detector 只输出：

$P_d$

后续也可以升级为：

$IMU \rightarrow
[
P_d,
\hat{\mathbf d},
\hat I
]$

即：

- 是否发生扰动；

- 扰动方向；

- 扰动强度。

这些信息对 Reflex Policy 可能非常有价值。

---

### Reflex Observation

Detector 可以严格限制为 IMU\-only，但 Reflex Controller 没必要只使用

IMU。

推荐 V1：

$o_t^R
=
[
\omega_t,
g_t^{proj},
q_t-q_0,
\dot q_t,
a_t^{WTW}
]$

如果四足机器人为 12 DoF：

$3+3+12+12+12=42$

约 42 维。

其中：

- \(`\omega`\\\_t\)：body angular velocity；

- \(g\_t\\^\{proj\}\)：projected gravity；

- \(q\_t\-q\_0\)：关节位置偏差；

- \(`\dot `q\_t\)：关节速度；

- \(a\_t\\^\{WTW\}\)：WTW 当前准备执行的 action。

加入：

$a_t\^{WTW}$

非常重要，因为 Reflex 可以知道：



> WTW 原本准备怎么控制机器人？
> 
> 



---

### Reflex Action

如果 WTW 输出为 12\-D joint target：

$a^{WTW}\in\mathbb R^{12}$

则：

$\Delta a^R\in\mathbb R^{12}$

最终：

$a = a\^{WTW} + \alpha\Delta a\^R$

Residual action 必须限制幅值：

$\Delta a_i\^R \in
[-\Delta a_i^{max},+\Delta a_i^{max}]$

可以采用：

$\Delta a^R
=
s_R\tanh(z)$

防止 Reflex 逐渐取代 WTW。



---

### MLP 与 PPO 的关系



这是讨论中一个重要概念澄清：

$MLP\neq PPO$

两者不是二选一。

- MLP

是神经网络结构：

$o_t\^R \rightarrow MLP \rightarrow \Delta a_t$

例如：

$42 \rightarrow
256 \rightarrow
128 \rightarrow
64 \rightarrow
12$

- PPO

是用于训练这个 MLP 参数的强化学习算法。

因此完整描述应该是：

$PPO\text{-}trained\ MLP\ Residual\ Reflex\ Policy$

可以理解为：

- MLP：大脑长什么样；

- PPO：如何训练这个大脑。

---

### Actor\-Critic

PPO 通常采用 Actor\-Critic。

Actor：

$o_t\^R \rightarrow MLP_{Actor} \rightarrow
\Delta a$

Critic：

$o_t\^R \rightarrow MLP_{Critic} \rightarrow V(s)$

PPO 根据 rollout 和 reward 更新网络参数。

实机部署时通常只需要 Actor：

$Robot State \rightarrow Reflex Actor \rightarrow
\Delta a$

---

### Reflex Reward

目标不是单纯：

> 不摔倒。
> 
> 

而是：

$\text{受到扰动后快速恢复到原来的 WTW locomotion 状态}$

可以设计：

$r_t = w_vr_{vel} +w_or_{ori} +w_hr_h +w_sr_{survive}
-\lambda_1\|\Delta a_t\|\^2
-\lambda*2\|\Delta a_t-\Delta a*{t-1}\|\^2$

- 速度恢复

$r_{vel} = \exp \left( -k_v \|v-v^{cmd}\|^2
\right)$

机器人受到扰动后仍应该继续完成原 locomotion command。

- 姿态恢复

$r_{ori} = \exp[-k_o(\phi^2+\theta^2)]$

- 身体高度

$r_h = \exp[-k_h(h-h_0)^2]$

防止机器人学会通过趴下等方式获得 survival reward。



- Residual Action Penalty

$r_{residual}
=
-\lambda\|\Delta a\|^2$

目的是形成：

$Minimal\ Intervention$

即 WTW 能解决的问题，Reflex 尽量不介入。



---

### Recovery Time

定义恢复条件，例如：

$|\phi| < \phi_{th}

|\theta| < \theta_{th}

\|v-v\^{cmd}\|<v_{th}$

并持续：

$T_{stable}$

则认为：

$Recovered$

定义：

$T_{recovery}
=
t_{recovered}
-
t_{disturbance}$

核心比较：

$T_{recovery}^{WTW+Reflex}
<
T_{recovery}^{WTW}$

---

### Trigger 与 Release 必须分开

Reflex 不能只在外力存在的几十毫秒内工作。

例如：

$T_{disturbance}=50ms$

机器人可能需要：

$T_{recovery}=300ms$

才能恢复。

因此：

$F=0$

并不意味着：

$Reflex=OFF$

正确状态：

```Plain Text
NORMAL
  ↓
Disturbance Trigger
  ↓
REFLEX ON
  ↓
Recovery
  ↓
Stable
  ↓
REFLEX OFF
```

即：

$Trigger\neq Release$

Detector 负责 Trigger。

Recovery State Machine 负责 Release。

---

### Reflex 平滑退出



恢复以后不一定直接：

$\alpha:1\rightarrow0$

可以使用：

$\alpha_t = e\^{-(t-t_r)/\tau}$

使 Reflex：

$1 \rightarrow0.8 \rightarrow0.5
\rightarrow0.2 \rightarrow0$

最终形成：

$WTW
\rightarrow
Reflex\ Assist
\rightarrow
WTW$

的平滑控制权交接。

---

### Isaac Gym 中如何训练 Reflex



整体训练循环：

```Plain Text
Reset
  ↓
WTW 正常 locomotion
  ↓
随机等待
  ↓
随机施加 disturbance
  ↓
Ground-Truth Trigger
  ↓
WTW + Reflex
  ↓
Recovery / Fall
  ↓
Reward
  ↓
PPO Update
  ↓
只更新 Reflex Policy
```

WTW：

$\text{Frozen}$

Reflex：

$\text{Trainable}$

概念上：

```Python
with torch.no_grad():
    action_wtw = wtw_policy(obs)

delta_action = reflex_policy(reflex_obs)

action = action_wtw + alpha * delta_action
```

PPO 只更新：

$\theta_R$

---

### 训练数据不需要提前采集

Reflex 使用 online RL，因此不需要传统 supervised dataset。

Isaac Gym 可以并行生成 rollout。

例如：

$N_{env}=4096$

不同环境同时施加：

```Plain Text
Env 1    → 左侧 push
Env 2    → 右侧 push
Env 3    → 前向 push
Env 4    → 后向 push
Env 5    → 小扰动
Env 6    → 中扰动
...
Env 4096 → 随机 disturbance
```

随机化：

$F\sim U(F_{min},F_{max})

\theta\sim U(0,2\pi)

t_d\sim U(t_{min},t_{max})

T_d\sim U(T_{min},T_{max})$



---

### Curriculum Learning

不建议训练一开始就使用极强扰动。

因为初始：

$\pi_R\approx Random$

如果每次扰动后机器人立即摔倒：

$episode\rightarrow terminate$

PPO 很难获得高质量 recovery trajectory。

建议：

- Level 1

轻微 base push。

- Level 2

中等 base push。

- Level 3

强 base push。

- Level 4

更复杂扰动：

- trip；

- foot support loss；

- step\-down；

- unexpected obstacle contact；

- rough terrain \+ disturbance。

    

---

### Reflex V1 建议

第一版不要同时解决所有问题。

建议：

$Frozen\ WTW
+
Random\ Base\ Push
+
Ground\ Truth\ Trigger
+
Residual\ PPO$

Reflex Actor：

$42 \rightarrow
256 \rightarrow
128 \rightarrow
64 \rightarrow
12$

输出：

$\Delta a = s_R\tanh(z)$

执行：

$a
=
a^{WTW}
+
\alpha\Delta a$

第一项实验只回答：



> 在完全知道扰动发生时刻的情况下，Residual Reflex 能否让 WTW
> 
> 更快、更稳定地恢复？
> 
> 



---

### Evaluation



建议至少记录：



$Fall Rate

T_{recovery}

\max\|\phi\|

\max\|\theta\|

\|v-v\^{cmd}\|$



以及 Reflex effort：

$\sum_t\|\Delta a_t\|\^2$



对比：

$WTW$

和：

$WTW+Reflex$

时，应使用：

- 相同 random seed；

- 相同扰动力；

- 相同扰动方向；

- 相同作用位置；

- 相同作用时刻；

- 相同 terrain；

- 相同 locomotion command。

这样才能得到公平的 paired comparison。

---

## 当前总体研究框架

最终形成三个核心模块：



$\underbrace{WTW}_{Nominal\ Locomotion}
+
\underbrace{Detector}_{When?}
+
\underbrace{Reflex}_{How?}$

数学上：

$a_t
=
\pi_{WTW}(o_t)
+
\alpha_t
\pi_R
\left(
o_t^R,
\pi_{WTW}(o_t)
\right)$

其中：

WTW

负责：

$\text{正常 locomotion}$

Detector

负责：

$\text{什么时候需要 Reflex}$

Reflex

负责：

$\text{如何产生 residual recovery action}$

Recovery State Machine

负责：

$\text{什么时候结束 Reflex 并将控制权完全交还 WTW}$



---



## 推荐开发顺序

建议按照以下顺序推进：

1. 保持原 WTW checkpoint 不变；

2. 在 Isaac Gym 中实现随机 base push；

3. 保存完整 disturbance ground truth；

4. 实现 Ground\-Truth Reflex Gate；

5. 实现 12\-D bounded residual MLP；

6. 使用 PPO 只训练 Reflex Actor/Critic；

7. 比较 WTW 与 WTW \+ GT Reflex；

8. 逐步增加 disturbance strength；

9. 增加 trip / support loss 等复杂扰动；

10. 独立训练 IMU disturbance detector；

11. 用 IMU Detector 替换 GT trigger；

12. 加入 online context adaptation；

13. 最终进行 sim\-to\-real 测试。

    

