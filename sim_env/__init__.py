"""可迁移的 Isaac Gym 突发扰动场景库。

这里不保存 WTW 策略，也不保存 Detector / Reflex 模型；它只负责构造
"正常运动 + 可控突发事件"。因此其他足式机器人项目可以复用场景定义，
只需为自己的基础环境实现 README 中列出的适配接口。
"""

from .scenario_config import SIM_ENV_CONFIG, SimEnvConfig, SimTaskConfig
from .unified_scenario_env import UnifiedScenarioEnv

__all__ = ["SIM_ENV_CONFIG", "SimTaskConfig", "SimEnvConfig", "UnifiedScenarioEnv"]
