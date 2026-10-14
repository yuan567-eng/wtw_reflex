"""采集固定 gait 的正常或单次侧向外力扰动六维 IMU 数据。"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pickle as pkl
import sys
import time
from pathlib import Path

import isaacgym

assert isaacgym
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from imu_collection_env import ImuCollectionEnv


GAIT_COMMANDS = {
    "pronking": [0.0, 0.0, 0.0],
    "trotting": [0.5, 0.0, 0.0],
    "bounding": [0.0, 0.5, 0.0],
    "pacing": [0.0, 0.0, 0.5],
}
DETECTOR_COMMAND_INDICES = (0, 1, 2, 4)
RANDOM_COMMAND_DEFAULT_RANGES = np.array(
    [[-1.0, 1.0], [-0.6, 0.6], [-1.0, 1.0], [2.0, 4.0]],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="采集固定 gait 的 200 Hz IMU 数据")
    parser.add_argument(
        "--dataset-type",
        choices=("normal", "disturbance"),
        default="normal",
        help="正常 episode 或单次外力扰动 episode",
    )
    parser.add_argument("--gait", choices=tuple(GAIT_COMMANDS), default="trotting")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="一次连续采集的独立 episode 数量",
    )
    parser.add_argument("--steps", type=int, default=500, help="保存的策略控制步数")
    parser.add_argument("--warmup-steps", type=int, default=100, help="采集前的稳定步数")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--gait-frequency", type=float, default=3.0)
    parser.add_argument(
        "--command-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="固定命令或连续随机命令",
    )
    parser.add_argument("--command-seed", type=int, default=0, help="随机命令种子")
    parser.add_argument("--vx-range", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument("--vy-range", type=float, nargs=2, default=(-0.6, 0.6))
    parser.add_argument("--yaw-rate-range", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument("--gait-frequency-range", type=float, nargs=2, default=(2.0, 4.0))
    parser.add_argument(
        "--command-hold-seconds",
        type=float,
        nargs=2,
        default=(1.0, 3.0),
        metavar=("MIN", "MAX"),
        help="随机目标到达后的保持时间范围",
    )
    parser.add_argument(
        "--command-transition-seconds",
        type=float,
        default=0.5,
        help="旧命令线性过渡到随机目标的时间",
    )
    parser.add_argument(
        "--disturbance-start-seconds",
        type=float,
        default=5.0,
        help="从正式采集开始到施力开始的时间",
    )
    parser.add_argument(
        "--disturbance-duration-seconds",
        type=float,
        default=0.1,
        help="侧向外力持续时间",
    )
    parser.add_argument(
        "--disturbance-recovery-seconds",
        type=float,
        default=0.5,
        help="施力结束后不参与训练的恢复期",
    )
    parser.add_argument(
        "--disturbance-force-y",
        type=float,
        default=60.0,
        help="机体系 y 轴外力，正负号决定左右方向，单位 N",
    )
    parser.add_argument(
        "--disturbance-force-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="随机扰动力绝对值范围；不提供时使用 disturbance-force-y 的绝对值",
    )
    parser.add_argument(
        "--disturbance-force-profile",
        choices=("constant", "test_5min_three_stage"),
        default="constant",
        help="恒定范围，或5分钟测试集的原范围/扩大50%%/缩小50%%三阶段",
    )
    parser.add_argument(
        "--disturbance-start-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="随机扰动开始时间范围；不提供时使用 disturbance-start-seconds",
    )
    parser.add_argument(
        "--disturbance-direction",
        choices=("fixed", "balanced", "random"),
        default="fixed",
        help="固定符号、正负交替平衡或随机方向",
    )
    parser.add_argument("--disturbance-seed", type=int, default=0, help="批量扰动参数种子")
    parser.add_argument(
        "--episode-index-offset",
        type=int,
        default=0,
        help="断点续采时跳过的批次序号，同时保持随机参数和正负交替连续",
    )
    parser.add_argument("--headless", action="store_true", help="关闭 Isaac Gym 可视化窗口")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument(
        "--label", default="gait-conditioned-agility/pretrain-go2/train",
        help="runs/ 下的预训练策略路径",
    )
    parser.add_argument("--output", type=Path, default=None, help="可选的输出 .npz 路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选输出目录；一次运行的全部 episode 合并保存为一个 .npz",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.warmup_steps < 0:
        parser.error("--episodes、--steps 必须大于 0，--warmup-steps 不能小于 0")
    if args.episode_index_offset < 0:
        parser.error("--episode-index-offset 不能小于 0")
    if args.output is not None and args.output_dir is not None:
        parser.error("--output 和 --output-dir 不能同时使用")
    for name, bounds in (
        ("--vx-range", args.vx_range),
        ("--vy-range", args.vy_range),
        ("--yaw-rate-range", args.yaw_rate_range),
        ("--gait-frequency-range", args.gait_frequency_range),
        ("--command-hold-seconds", args.command_hold_seconds),
    ):
        if bounds[0] > bounds[1]:
            parser.error(f"{name} 必须满足 MIN <= MAX")
    if args.gait_frequency_range[0] <= 0:
        parser.error("--gait-frequency-range 必须大于 0")
    if args.command_hold_seconds[0] < 0:
        parser.error("--command-hold-seconds 不能小于 0")
    if args.command_transition_seconds < 0:
        parser.error("--command-transition-seconds 不能小于 0")
    if args.disturbance_start_seconds < 0:
        parser.error("--disturbance-start-seconds 不能小于 0")
    if args.disturbance_duration_seconds <= 0:
        parser.error("--disturbance-duration-seconds 必须大于 0")
    if args.disturbance_recovery_seconds < 0:
        parser.error("--disturbance-recovery-seconds 不能小于 0")
    if args.dataset_type == "disturbance" and args.disturbance_force_y == 0:
        parser.error("disturbance 模式下 --disturbance-force-y 不能为 0")
    if args.disturbance_force_range is not None:
        force_min, force_max = args.disturbance_force_range
        if force_min <= 0 or force_min > force_max:
            parser.error("--disturbance-force-range 必须满足 0 < MIN <= MAX")
    if args.disturbance_force_profile == "test_5min_three_stage":
        if args.dataset_type != "disturbance":
            parser.error("test_5min_three_stage 只能用于 disturbance 数据")
        if args.disturbance_force_range is None:
            parser.error("test_5min_three_stage 必须提供 --disturbance-force-range")
        if args.episodes != 30 or args.episode_index_offset != 0:
            parser.error("test_5min_three_stage 必须使用 --episodes 30 且 offset=0")
    if args.disturbance_start_range is not None:
        start_min, start_max = args.disturbance_start_range
        if start_min < 0 or start_min > start_max:
            parser.error("--disturbance-start-range 必须满足 0 <= MIN <= MAX")
    requested_ranges = np.array(
        [args.vx_range, args.vy_range, args.yaw_rate_range, args.gait_frequency_range],
        dtype=np.float32,
    )
    if np.any(requested_ranges[:, 0] < RANDOM_COMMAND_DEFAULT_RANGES[:, 0]) or np.any(
        requested_ranges[:, 1] > RANDOM_COMMAND_DEFAULT_RANGES[:, 1]
    ):
        parser.error(
            "随机命令范围不能超过 checkpoint 训练范围："
            "vx[-1,1]、vy[-0.6,0.6]、yaw_rate[-1,1]、gait_frequency[2,4]"
        )
    initial_command = np.array(
        [args.vx, args.vy, args.yaw_rate, args.gait_frequency], dtype=np.float32
    )
    if args.command_mode == "random" and (
        np.any(initial_command < requested_ranges[:, 0])
        or np.any(initial_command > requested_ranges[:, 1])
    ):
        parser.error(
            "随机模式的初始 --vx/--vy/--yaw-rate/--gait-frequency "
            "必须位于对应随机范围内"
        )
    return args


class RandomCommandScheduler:
    """在 50 Hz 控制周期上生成可复现的随机目标和线性过渡。"""

    def __init__(self, initial_command, command_ranges, hold_seconds, transition_seconds, dt, seed):
        self.current = np.asarray(initial_command, dtype=np.float32).copy()
        self.start = self.current.copy()
        self.target = self.current.copy()
        self.command_ranges = np.asarray(command_ranges, dtype=np.float32)
        self.hold_seconds = hold_seconds
        self.transition_steps = max(1, int(round(transition_seconds / dt)))
        self.rng = np.random.default_rng(seed)
        self.transition_index = self.transition_steps
        self.hold_steps_remaining = 0
        self.dt = dt

    def _sample_hold_steps(self):
        seconds = self.rng.uniform(self.hold_seconds[0], self.hold_seconds[1])
        return max(1, int(round(seconds / self.dt)))

    def _start_transition(self):
        self.start = self.current.copy()
        self.target = self.rng.uniform(
            self.command_ranges[:, 0], self.command_ranges[:, 1]
        ).astype(np.float32)
        self.transition_index = 0

    def next(self):
        if self.transition_index >= self.transition_steps and self.hold_steps_remaining <= 0:
            self._start_transition()

        if self.transition_index < self.transition_steps:
            self.transition_index += 1
            ratio = self.transition_index / self.transition_steps
            self.current = self.start + ratio * (self.target - self.start)
            if self.transition_index == self.transition_steps:
                self.current = self.target.copy()
                self.hold_steps_remaining = self._sample_hold_steps()
        else:
            self.hold_steps_remaining -= 1
        return self.current.copy()


def find_logdir(label: str) -> Path:
    candidates = sorted(Path(path) for path in glob.glob(str(PROJECT_ROOT / "runs" / label / "*")))
    for candidate in candidates:
        if (candidate / "parameters.pkl").is_file() and (candidate / "checkpoints/body_latest.jit").is_file():
            return candidate
    raise FileNotFoundError(f"未找到可用的预训练策略目录：runs/{label}")


def load_policy(logdir: Path):
    body = torch.jit.load(str(logdir / "checkpoints/body_latest.jit"))
    adaptation_module = torch.jit.load(str(logdir / "checkpoints/adaptation_module_latest.jit"))

    def policy(obs):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        return body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))

    return policy


def load_training_config(logdir: Path, args):
    # CPU smoke test 需要使用已转换为 CPU tensor 的配置；GPU 正式采集沿用原配置。
    config_name = "parameters_cpu.pkl" if args.sim_device == "cpu" else "parameters.pkl"
    with (logdir / config_name).open("rb") as file:
        if args.sim_device == "cpu":
            original_loader = torch.storage._load_from_bytes
            torch.storage._load_from_bytes = lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
            try:
                saved_cfg = pkl.load(file)["Cfg"]
            finally:
                torch.storage._load_from_bytes = original_loader
        else:
            saved_cfg = pkl.load(file)["Cfg"]
    for key, value in saved_cfg.items():
        if hasattr(Cfg, key):
            for child_key, child_value in value.items():
                setattr(getattr(Cfg, key), child_key, child_value)

    # 最小验证环境：单机器人、平地、无扰动、无域随机化。
    for name in (
        "push_robots", "randomize_friction", "randomize_gravity", "randomize_restitution",
        "randomize_motor_offset", "randomize_motor_strength", "randomize_friction_indep",
        "randomize_ground_friction", "randomize_base_mass", "randomize_Kd_factor",
        "randomize_Kp_factor", "randomize_joint_friction", "randomize_com_displacement",
        "randomize_lag_timesteps",
    ):
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)

    Cfg.noise.add_noise = False
    Cfg.env.num_envs = 1
    Cfg.env.num_recording_envs = 1
    Cfg.env.episode_length_s = max(30.0, (args.warmup_steps + args.steps + 10) * 0.02)
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.measure_heights = False
    Cfg.terrain.teleport_robots = False
    Cfg.commands.resampling_time = Cfg.env.episode_length_s + 10.0
    Cfg.control.control_type = "P" if args.sim_device == "cpu" else "actuator_net"
    Cfg.asset.flip_visual_attachments = True
    if args.sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False


def get_initial_command(args):
    return np.array(
        [args.vx, args.vy, args.yaw_rate, args.gait_frequency], dtype=np.float32
    )


def get_random_command_ranges(args):
    return np.array(
        [args.vx_range, args.vy_range, args.yaw_rate_range, args.gait_frequency_range],
        dtype=np.float32,
    )


def set_motion_command(env: ImuCollectionEnv, args, detector_command):
    env.commands[:, 0] = float(detector_command[0])
    env.commands[:, 1] = float(detector_command[1])
    env.commands[:, 2] = float(detector_command[2])
    env.commands[:, 3] = 0.0
    env.commands[:, 4] = float(detector_command[3])
    env.commands[:, 5:8] = torch.tensor(GAIT_COMMANDS[args.gait], device=env.device)
    env.commands[:, 8] = 0.5
    env.commands[:, 9] = 0.08
    env.commands[:, 10] = 0.0
    env.commands[:, 11] = 0.0
    env.commands[:, 12] = 0.25


def step_policy(env, base_env, policy, obs, args, requested_command):
    set_motion_command(base_env, args, requested_command)
    # 命令必须在 env.step() 前保存；这正是随后四个物理步实际使用的控制上下文。
    detector_command = (
        base_env.commands[0, list(DETECTOR_COMMAND_INDICES)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    with torch.no_grad():
        actions = policy(obs)
    obs, _, done, info = env.step(actions)
    if bool(done[0].item()):
        raise RuntimeError("数据采集期间机器人发生 reset；本 episode 不应保存。")
    return obs, info, detector_command


def sample_disturbance_parameters(args, rng, episode_index):
    """为一个 episode 生成可复现且方向可平衡的扰动参数。"""
    if args.disturbance_force_range is None:
        magnitude = abs(args.disturbance_force_y)
    else:
        force_low, force_high = args.disturbance_force_range
        if args.disturbance_force_profile == "test_5min_three_stage":
            if episode_index < 6:
                scale_factor = 1.0
            elif episode_index < 18:
                scale_factor = 1.5
            elif episode_index < 30:
                scale_factor = 0.5
            else:
                raise ValueError("test_5min_three_stage 的 episode_index 必须位于0~29")
            force_low *= scale_factor
            force_high *= scale_factor
        magnitude = float(rng.uniform(force_low, force_high))

    if args.disturbance_direction == "fixed":
        direction = 1.0 if args.disturbance_force_y > 0 else -1.0
    elif args.disturbance_direction == "balanced":
        direction = 1.0 if episode_index % 2 == 0 else -1.0
    else:
        direction = float(rng.choice((-1.0, 1.0)))

    if args.disturbance_start_range is None:
        start_seconds = args.disturbance_start_seconds
    else:
        start_seconds = float(rng.uniform(*args.disturbance_start_range))
    return start_seconds, direction * magnitude


def resolve_output_path(args, batch_id_value):
    if args.output is not None:
        return args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if args.output_dir is not None:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
        )
    else:
        gait_group = "normal_gait" if args.dataset_type == "normal" else "abnormal_gait"
        output_dir = PROJECT_ROOT / "dataset/raw" / gait_group / args.gait
    return output_dir / f"collection_{batch_id_value}_{args.episodes}ep.npz"


def collect_one_episode(
    args,
    env,
    base_env,
    policy,
    fixed_command,
    control_dt,
    gait_id_value,
    episode_index,
    sequence_index,
    disturbance_start,
    disturbance_force_y,
):
    """reset、warmup并采集一个 episode，最后由批次统一保存。"""
    obs = env.reset()
    display_index = episode_index + 1
    for _ in tqdm(
        range(args.warmup_steps),
        desc=f"warmup {display_index}/{args.episodes}",
    ):
        obs, _, _ = step_policy(env, base_env, policy, obs, args, fixed_command)

    if args.dataset_type == "disturbance":
        base_env.arm_body_force_disturbance(
            start_delay_seconds=disturbance_start,
            duration_seconds=args.disturbance_duration_seconds,
            recovery_seconds=args.disturbance_recovery_seconds,
            body_force_xyz=[0.0, disturbance_force_y, 0.0],
        )

    scheduler = None
    if args.command_mode == "random":
        scheduler = RandomCommandScheduler(
            initial_command=fixed_command,
            command_ranges=get_random_command_ranges(args),
            hold_seconds=args.command_hold_seconds,
            transition_seconds=args.command_transition_seconds,
            dt=control_dt,
            seed=args.command_seed + sequence_index,
        )

    imu_chunks, timestamp_chunks, command_chunks = [], [], []
    disturbance_chunks, settling_chunks = [], []
    for _ in tqdm(
        range(args.steps),
        desc=f"collect {display_index}/{args.episodes}",
    ):
        requested_command = scheduler.next() if scheduler is not None else fixed_command
        obs, info, detector_command = step_policy(
            env, base_env, policy, obs, args, requested_command
        )
        imu_chunks.append(info["imu_raw"][:, 0, :])
        timestamp_chunks.append(info["imu_timestamp_us"])
        disturbance_chunks.append(info["disturbance_active"])
        settling_chunks.append(info["disturbance_settling"])
        # 一个 50 Hz 控制命令对应 decimation 个 200 Hz IMU 样本。
        command_chunks.append(
            np.repeat(
                detector_command[None, :],
                repeats=len(info["imu_timestamp_us"]),
                axis=0,
            )
        )

    imu_raw = np.concatenate(imu_chunks, axis=0).astype(np.float32, copy=False)
    timestamp_us = np.concatenate(timestamp_chunks, axis=0).astype(np.int64, copy=False)
    commands = np.concatenate(command_chunks, axis=0).astype(np.float32, copy=False)
    disturbance_active = np.concatenate(disturbance_chunks).astype(np.bool_, copy=False)
    settling = np.concatenate(settling_chunks).astype(np.bool_, copy=False)
    expected_samples = args.steps * Cfg.control.decimation
    if imu_raw.shape != (expected_samples, 6):
        raise RuntimeError(f"IMU shape 错误：得到 {imu_raw.shape}，期望 {(expected_samples, 6)}")
    if commands.shape != (expected_samples, 4):
        raise RuntimeError(
            f"commands shape 错误：得到 {commands.shape}，期望 {(expected_samples, 4)}"
        )
    if disturbance_active.shape != (expected_samples,) or settling.shape != (expected_samples,):
        raise RuntimeError("扰动事件标签没有与 200 Hz IMU 样本逐点对齐")
    if not np.isfinite(imu_raw).all() or not np.isfinite(commands).all():
        raise RuntimeError("IMU 或 commands 包含 NaN/Inf")

    intervals = np.diff(timestamp_us)
    expected_interval_us = int(round(Cfg.sim.dt * 1_000_000))
    if len(intervals) and not np.all(intervals == expected_interval_us):
        raise RuntimeError("IMU 时间戳不连续，未达到稳定的物理步采样频率")

    if args.dataset_type == "disturbance":
        expected_active = int(round(args.disturbance_duration_seconds / Cfg.sim.dt))
        expected_settling = int(round(args.disturbance_recovery_seconds / Cfg.sim.dt))
        if int(disturbance_active.sum()) != expected_active:
            raise RuntimeError(
                f"扰动标签数量错误：得到 {disturbance_active.sum()}，期望 {expected_active}"
            )
        if int(settling.sum()) != expected_settling:
            raise RuntimeError(
                f"恢复期标签数量错误：得到 {settling.sum()}，期望 {expected_settling}"
            )
    elif disturbance_active.any() or settling.any():
        raise RuntimeError("normal 模式不应出现扰动或恢复期标签")

    episode_id_value = int(time.time_ns() // 1_000)
    sample_count = len(timestamp_us)
    episode = {
        "timestamp_us": timestamp_us,
        "imu_raw": imu_raw,
        "commands": commands,
        "gait_id": np.full(sample_count, gait_id_value, dtype=np.int8),
        "switching": np.zeros(sample_count, dtype=np.bool_),
        "settling": settling,
        "detector_target": disturbance_active.astype(np.int8),
        "loss_mask": ~settling,
        "episode_id": np.full(sample_count, episode_id_value, dtype=np.int64),
    }
    actual_rate = 1_000_000.0 / float(np.median(intervals)) if len(intervals) else float("nan")
    print(
        f"episode {display_index}/{args.episodes} 采集完成 | "
        f"samples={sample_count}, rate={actual_rate:.1f}Hz, "
        f"target={int(disturbance_active.sum())}, settling={int(settling.sum())}"
    )
    if args.dataset_type == "disturbance":
        print(
            f"  force_y={disturbance_force_y:.2f}N, start={disturbance_start:.3f}s, "
            f"duration={args.disturbance_duration_seconds:.3f}s"
        )
    return episode


def collect_episodes(args):
    logdir = find_logdir(args.label)
    load_training_config(logdir, args)
    control_dt = float(Cfg.sim.dt * Cfg.control.decimation)
    collection_seconds = args.steps * control_dt
    if args.disturbance_force_profile == "test_5min_three_stage":
        if not np.isclose(collection_seconds, 10.0, atol=1e-6):
            raise ValueError(
                "test_5min_three_stage 要求每个 episode 恰好10秒；"
                f"当前为 {collection_seconds:.3f} 秒"
            )
        if args.disturbance_start_range is not None or not np.isclose(
            args.disturbance_start_seconds, 5.0, atol=1e-6
        ):
            raise ValueError(
                "test_5min_three_stage 要求固定在每个 episode 第5秒施加扰动"
            )
    maximum_start = (
        args.disturbance_start_seconds
        if args.disturbance_start_range is None
        else args.disturbance_start_range[1]
    )
    if args.dataset_type == "disturbance":
        maximum_schedule_end = (
            maximum_start
            + args.disturbance_duration_seconds
            + args.disturbance_recovery_seconds
        )
        if maximum_schedule_end > collection_seconds + 1e-9:
            raise ValueError(
                "扰动范围和恢复期超出采集时长："
                f"最晚结束于 {maximum_schedule_end:.3f}s，采集仅 {collection_seconds:.3f}s"
            )

    base_env = ImuCollectionEnv(sim_device=args.sim_device, headless=args.headless, cfg=Cfg)
    env = HistoryWrapper(base_env)
    policy = load_policy(logdir)
    fixed_command = get_initial_command(args)
    mapping_path = PROJECT_ROOT / "dataset/metadata/gait_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as file:
        gait_id_value = int(json.load(file)[args.gait])

    rng = np.random.default_rng(args.disturbance_seed)
    # 断点续采时先消费已完成 episode 对应的随机数，使参数序列可复现。
    if args.dataset_type == "disturbance":
        for skipped_index in range(args.episode_index_offset):
            sample_disturbance_parameters(args, rng, skipped_index)
    episodes = []
    for episode_index in range(args.episodes):
        sequence_index = args.episode_index_offset + episode_index
        if args.dataset_type == "disturbance":
            disturbance_start, disturbance_force_y = sample_disturbance_parameters(
                args, rng, sequence_index
            )
        else:
            disturbance_start, disturbance_force_y = 0.0, 0.0
        episodes.append(
            collect_one_episode(
                args,
                env,
                base_env,
                policy,
                fixed_command,
                control_dt,
                gait_id_value,
                episode_index,
                sequence_index,
                disturbance_start,
                disturbance_force_y,
            )
        )

    # 一次运行只产生一个文件；episode_id 保留各段边界，窗口仍禁止跨 episode。
    merged = {
        field: np.concatenate([episode[field] for episode in episodes], axis=0)
        for field in episodes[0]
    }
    episode_ids = np.unique(merged["episode_id"])
    if len(episode_ids) != args.episodes:
        raise RuntimeError(
            f"合并后 episode_id 数量错误：得到 {len(episode_ids)}，期望 {args.episodes}"
        )
    batch_id_value = int(episode_ids[0])
    output_path = resolve_output_path(args, batch_id_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    saved_minutes = args.episodes * collection_seconds / 60.0
    print(
        f"批量采集完成：episodes={len(episodes)}, samples={len(merged['timestamp_us'])}, "
        f"有效数据时长={saved_minutes:.3f}min, command_mode={args.command_mode}"
    )
    print(f"本次采集唯一输出：{output_path}")
    return output_path


if __name__ == "__main__":
    collect_episodes(parse_args())
