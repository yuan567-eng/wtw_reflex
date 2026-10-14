"""统一扰动仿真环境的唯一参数入口。

修改文件最底部的 ``SIM_ENV_CONFIG`` 后，直接运行
``python sim_env/run_wtw_scenario.py`` 即可；不需要命令行参数。
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple

# 正常场景 + 原实验 2 的四类事件 + 原实验 5 的两类突发几何事件。
ScenarioName = Literal[
    "normal", "slip", "kick", "foothold_loss", "leg_fault", "sudden_geometry"
]
GeometryEventName = Literal["random", "ground_drop", "obstacle_ring"]
HoleShape = Literal["rectangle", "circle"]
JointFaultMode = Literal["hold_position", "zero_torque"]


@dataclass
class SlipConfig:
    """指定足端打滑：低摩擦 + 一段短时切向牵引力。"""

    # 0=FL，1=RL，2=FR，3=RR；(-1,) 表示每个回合随机选一条腿。
    target_legs: Tuple[int, ...] = (-1,)
    duration_s: float = 0.20
    tangential_force_xy: Tuple[float, float] = (0.0, 85.0)
    foot_friction: float = 0.05


@dataclass
class KickConfig:
    """模拟机身受到一脚短时踢击。"""

    duration_s: float = 0.20
    force_magnitude_n: float = 180.0
    # 方向以机身坐标表示：先按 RPY 旋转局部 +z 轴得到施力方向。
    direction_rpy: Tuple[float, float, float] = (-1.5708, 0.0, 0.0)
    # 施力点相对机身重心的偏移 (x, y, z)，单位 m。
    application_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.16)


@dataclass
class FootholdLossConfig:
    """机器人起步于高平台；按 T 后开始前进并越过平台边缘。"""

    platform_height: float = 0.50
    platform_size_xy: Tuple[float, float] = (2.5, 2.5)


@dataclass
class LegFaultConfig:
    """指定腿在按 T 后不再执行新的动作。"""

    target_legs: Tuple[int, ...] = (-1,)
    mode: JointFaultMode = "hold_position"


@dataclass
class GroundDropConfig:
    """由小块拼成的平台；按 T 后前方部分地面下沉形成坑洞。"""

    platform_size_xy: Tuple[float, float] = (10.0, 10.0)
    platform_height: float = 0.30
    tile_size: float = 0.20
    hole_shape: HoleShape = "rectangle"
    hole_size_xy: Tuple[float, float] = (0.60, 0.40)
    hole_forward_offset_range: Tuple[float, float] = (0.85, 0.85)
    hole_lateral_offset_range: Tuple[float, float] = (-0.08, 0.08)
    drop_depth: float = 0.60


@dataclass
class ObstacleRingConfig:
    """初始隐藏、按 T 后升起并围住机器人的低矮环形障碍。"""

    height: float = 0.08
    thickness: float = 0.10
    segments: int = 32
    foot_clearance: float = 0.12
    min_inner_radius: float = 0.48
    max_inner_radius: float = 0.72


@dataclass
class SimTaskConfig:
    """统一场景运行所需的正常行走、显示和录像参数。

    这里刻意不继承 snn.env 的 WalkTaskConfig：后者包含 Reflex Detector
    的配置，而 sim_env 仅提供场景，不应携带检测器的任何概念或参数。
    """

    # ── WTW 的固定正常行走任务 ──────────────────────────────────────
    command_vx: float = 0.5
    command_vy: float = 0.0
    command_yaw: float = 0.0
    body_height_cmd: float = 0.0
    step_frequency_cmd: float = 3.0
    gait_phase_cmd: float = 0.5
    gait_offset_cmd: float = 0.0
    gait_bound_cmd: float = 0.0
    gait_duration_cmd: float = 0.5
    footswing_height_cmd: float = 0.08
    pitch_cmd: float = 0.0
    roll_cmd: float = 0.0
    stance_width_cmd: float = 0.25

    # ── 回合与可视化 ────────────────────────────────────────────────
    episode_length_s: float = 18.0
    draw_trajectory: bool = True
    trajectory_image_path: str = "snn/results/sim_env_trajectory.png"
    chinese_font_path: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    camera_offset: Tuple[float, float, float] = (-3.0, -3.0, 1.8)
    camera_target_height: float = 0.35

    # ── 可选录像 ────────────────────────────────────────────────────
    record_video: bool = True
    video_path: str = "snn/results/sim_env_rollout.mp4"
    video_fps: int = 25
    video_frame_stride: int = 2
    video_width_px: int = 640
    video_height_px: int = 480


@dataclass
class SimEnvConfig(SimTaskConfig):
    """统一场景环境配置；所有可调参数均集中于此。"""

    policy_label: str = "gait-conditioned-agility/pretrain-go2/train"
    num_envs: int = 1
    headless: bool = False
    # normal 不需要按 T；其余场景均通过 T 手动触发。
    scenario: ScenarioName = "normal"
    # sudden_geometry 专用：random 表示按 T 时在两类几何事件中随机选择。
    geometry_event: GeometryEventName = "random"
    ground_drop_probability: float = 0.5
    # slip / kick 已默认可反复按 T；此项控制突发几何是否允许重复触发。
    allow_multiple_geometry_triggers: bool = False
    slip: SlipConfig = field(default_factory=SlipConfig)
    kick: KickConfig = field(default_factory=KickConfig)
    foothold_loss: FootholdLossConfig = field(default_factory=FootholdLossConfig)
    leg_fault: LegFaultConfig = field(default_factory=LegFaultConfig)
    ground_drop: GroundDropConfig = field(default_factory=GroundDropConfig)
    obstacle_ring: ObstacleRingConfig = field(default_factory=ObstacleRingConfig)


# ============================ 你主要修改这里 ============================
SIM_ENV_CONFIG = SimEnvConfig(
    command_vx=0.5,
    episode_length_s=18.0,
    headless=False,             # False 才能使用窗口的 T 键触发事件。
    scenario="sudden_geometry",          # normal / slip / kick / foothold_loss / leg_fault / sudden_geometry
    geometry_event="random",    # sudden_geometry 时：random / ground_drop / obstacle_ring
    ground_drop_probability=0.5,
    allow_multiple_geometry_triggers=False,
    trajectory_image_path="snn/results/sim_env_trajectory.png",
    video_path="snn/results/sim_env_rollout.mp4",
)
