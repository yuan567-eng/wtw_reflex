# 可复用突发扰动仿真环境

此目录将此前分散在实验 1、2、5 的场景统一起来。它只负责**场景与扰动注入**：不包含 WTW checkpoint、Detector 网络或 Reflex 网络，也不会运行任何反射检测功能。

当前场景通过 `scenario_config.py` 内的 `SIM_ENV_CONFIG.scenario` 切换：

| 值 | 行为 | T 键 |
|---|---|---|
| `normal` | 无扰动平地正常行走 | 无作用 |
| `slip` | 指定足端低摩擦并施加切向力 | 每次追加一次打滑脉冲 |
| `kick` | 对机身指定位置、方向、大小的踢击 | 每次追加一次踢击 |
| `foothold_loss` | 高平台边缘自然踩空 | 放行机器人开始前进 |
| `leg_fault` | 指定腿的三个关节冻结或零力矩 | 触发一次故障并保持到回合结束 |
| `sudden_geometry` | 前方地面下沉或环形障碍突然升起 | 随机或按配置触发几何事件 |

WTW 中可直接运行：

```bash
python sim_env/run_wtw_scenario.py
```

## 被其他项目复用时

`UnifiedScenarioEnv` 目前是 Go2/WTW 的一个适配版本，因为它继承了 `NormalWalkEnv`。但其场景参数、按键语义、事件状态变量均在本目录。迁移到另一 Isaac Gym 足式项目时，只需要将其基类替换为对方的环境，并确保对方暴露：

- `gym`、`sim`、`envs`、`env_origins`、`actor_handles`；
- `base_pos`、`base_quat`、`foot_positions`、`feet_indices`；
- `root_states`、`joint_pos_target`、`default_dof_pos`、`num_actions`；
- 一个可在每控制步调用的 `_post_physics_step_callback`。

这样扰动本身无需重写；只需做一次机器人状态字段的适配。
