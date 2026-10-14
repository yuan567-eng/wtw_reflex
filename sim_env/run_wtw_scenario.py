#!/usr/bin/env python3
"""使用冻结 WTW 策略运行 sim_env 的统一场景。

所有参数均在 scenario_config.py 的 SIM_ENV_CONFIG 中修改；此脚本不解析命令行参数。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Isaac Gym 需要在 torch 之前由环境模块导入。
from sim_env import SIM_ENV_CONFIG, UnifiedScenarioEnv
from snn.scripts.benchmark_common import make_flat_wtw_env, printable, run_one_episode, save_video


def main():
    config = SIM_ENV_CONFIG
    print("统一仿真环境配置：", config)
    env, policy = make_flat_wtw_env(
        PROJECT_ROOT, config.policy_label, UnifiedScenarioEnv, config, config.num_envs, config.headless
    )
    run_one_episode(env, policy, record_video=config.record_video)
    print("统一仿真环境指标：", printable(env.env.get_scenario_metrics()))
    if config.draw_trajectory:
        env.env.save_trajectory_plot(
            PROJECT_ROOT / config.trajectory_image_path,
            f"统一环境：{config.scenario} 轨迹",
            config.chinese_font_path,
        )
    if config.record_video:
        save_video(env, PROJECT_ROOT / config.video_path, config.video_fps)


if __name__ == "__main__":
    main()
