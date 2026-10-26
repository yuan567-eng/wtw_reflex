"""采集 Detector V2 的正常数据：IMU、命令、步态、关节位置和关节速度。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from imu_joint_collection_env import ImuJointCollectionEnv


GAIT_COMMANDS = {
    "pronking": [0.0, 0.0, 0.0],
    "trotting": [0.5, 0.0, 0.0],
    "bounding": [0.0, 0.5, 0.0],
    "pacing": [0.0, 0.0, 0.5],
}
GAIT_MAPPING = {"pronking": 0, "trotting": 1, "bounding": 2, "pacing": 3}
DETECTOR_COMMAND_INDICES = (0, 1, 2, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="采集 Detector V2 的正常 200 Hz IMU 与关节本体感觉数据"
    )
    parser.add_argument("--gait", choices=tuple(GAIT_COMMANDS), default="trotting")
    parser.add_argument("--steps", type=int, default=500, help="采集的 50 Hz 策略步数")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--gait-frequency", type=float, default=3.0)
    parser.add_argument(
        "--label",
        default="gait-conditioned-agility/pretrain-go2/train",
        help="runs/ 下保存 WTW JIT 策略的父目录",
    )
    parser.add_argument("--sim-device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0:
        parser.error("steps 必须大于 0，warmup-steps 不可为负")
    return args


def find_logdir(label: str) -> Path:
    candidates = sorted(
        Path(path) for path in glob.glob(str(PROJECT_ROOT / "runs" / label / "*"))
    )
    for candidate in candidates:
        if (
            candidate / "parameters.pkl"
        ).is_file() and (candidate / "checkpoints/body_latest.jit").is_file():
            return candidate
    raise FileNotFoundError(f"未找到预训练策略：runs/{label}")


def load_policy(logdir: Path):
    body = torch.jit.load(str(logdir / "checkpoints/body_latest.jit"))
    adaptation = torch.jit.load(str(logdir / "checkpoints/adaptation_module_latest.jit"))

    def policy(obs):
        history = obs["obs_history"].to("cpu")
        latent = adaptation.forward(history)
        return body.forward(torch.cat((history, latent), dim=-1))

    return policy


def load_training_config(logdir: Path, args):
    """加载 WTW 已训练策略配置，并关闭数据集采集不需要的随机化。"""
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

    for name in (
        "push_robots", "randomize_friction", "randomize_gravity",
        "randomize_restitution", "randomize_motor_offset", "randomize_motor_strength",
        "randomize_friction_indep", "randomize_ground_friction",
        "randomize_base_mass", "randomize_Kd_factor", "randomize_Kp_factor",
        "randomize_joint_friction", "randomize_com_displacement",
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


def set_motion_command(env, args, detector_command=None):
    """将 Detector 使用的四维运动命令写入 WTW 环境。

    ``detector_command=None`` 时使用命令行固定值；批量实验采集器可传入
    ``[vx, vy, yaw_rate, gait_frequency]`` 以写入随机且平滑过渡的命令。
    """
    if detector_command is None:
        detector_command = (args.vx, args.vy, args.yaw_rate, args.gait_frequency)
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


def step_policy(env, base_env, policy, obs, args):
    set_motion_command(base_env, args)
    detector_command = (
        base_env.commands[0, list(DETECTOR_COMMAND_INDICES)]
        .detach().cpu().numpy().astype(np.float32, copy=True)
    )
    with torch.no_grad():
        actions = policy(obs)
    obs, _, done, info = env.step(actions)
    if bool(done[0].item()):
        raise RuntimeError("采集期间机器人发生 reset，本 episode 不保存")
    return obs, info, detector_command


def resolve_output(args):
    if args.output is not None:
        return args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    identifier = int(time.time_ns() // 1_000)
    return (
        PROJECT_ROOT
        / "detector_v2/dataset/raw/normal"
        / args.gait
        / f"normal_{identifier}.npz"
    )


def collect(args):
    logdir = find_logdir(args.label)
    load_training_config(logdir, args)
    base_env = ImuJointCollectionEnv(
        sim_device=args.sim_device, headless=args.headless, cfg=Cfg
    )
    env = HistoryWrapper(base_env)
    policy = load_policy(logdir)
    obs = env.reset()

    for _ in tqdm(range(args.warmup_steps), desc="warmup"):
        obs, _, _ = step_policy(env, base_env, policy, obs, args)

    imu_chunks, joint_pos_chunks, joint_vel_chunks = [], [], []
    timestamp_chunks, command_chunks = [], []
    for _ in tqdm(range(args.steps), desc="collect"):
        obs, info, command = step_policy(env, base_env, policy, obs, args)
        imu_chunks.append(info["imu_raw"][:, 0, :])
        joint_pos_chunks.append(info["joint_pos"][:, 0, :])
        joint_vel_chunks.append(info["joint_vel"][:, 0, :])
        timestamp_chunks.append(info["timestamp_us"])
        command_chunks.append(
            np.repeat(command[None, :], len(info["timestamp_us"]), axis=0)
        )

    timestamp_us = np.concatenate(timestamp_chunks).astype(np.int64, copy=False)
    imu_raw = np.concatenate(imu_chunks).astype(np.float32, copy=False)
    joint_pos = np.concatenate(joint_pos_chunks).astype(np.float32, copy=False)
    joint_vel = np.concatenate(joint_vel_chunks).astype(np.float32, copy=False)
    commands = np.concatenate(command_chunks).astype(np.float32, copy=False)
    sample_count = len(timestamp_us)
    expected_samples = args.steps * Cfg.control.decimation

    expected_shapes = {
        "imu_raw": (expected_samples, 6),
        "joint_pos": (expected_samples, 12),
        "joint_vel": (expected_samples, 12),
        "commands": (expected_samples, 4),
    }
    observed = {
        "imu_raw": imu_raw.shape,
        "joint_pos": joint_pos.shape,
        "joint_vel": joint_vel.shape,
        "commands": commands.shape,
    }
    for name, expected in expected_shapes.items():
        if observed[name] != expected:
            raise RuntimeError(f"{name} shape 错误：得到 {observed[name]}，期望 {expected}")
    if not all(np.isfinite(item).all() for item in (imu_raw, joint_pos, joint_vel, commands)):
        raise RuntimeError("采集数据含有 NaN 或 Inf")

    intervals = np.diff(timestamp_us)
    expected_interval_us = int(round(Cfg.sim.dt * 1_000_000))
    if len(intervals) and not np.all(intervals == expected_interval_us):
        raise RuntimeError("时间戳不连续，未达到预期物理步采样频率")

    episode_id = int(time.time_ns() // 1_000)
    # 明确写入采样率，数据加载器无需仅靠时间戳推断 200 Hz。
    sample_rate_hz = np.float32(1_000_000.0 / expected_interval_us)
    episode = {
        "timestamp_us": timestamp_us,
        "imu_raw": imu_raw,
        "commands": commands,
        "gait_id": np.full(sample_count, GAIT_MAPPING[args.gait], dtype=np.int8),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "switching": np.zeros(sample_count, dtype=np.bool_),
        "settling": np.zeros(sample_count, dtype=np.bool_),
        "detector_target": np.zeros(sample_count, dtype=np.int8),
        "loss_mask": np.ones(sample_count, dtype=np.bool_),
        "episode_id": np.full(sample_count, episode_id, dtype=np.int64),
        "joint_names": np.asarray(base_env.dof_names, dtype="<U32"),
        "joint_default_pos": base_env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32),
        "sample_rate_hz": sample_rate_hz,
    }
    output = resolve_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **episode)

    print(f"已保存：{output}")
    print(
        f"samples={sample_count}, rate={sample_rate_hz:.1f}Hz, "
        f"imu={imu_raw.shape}, joint_pos={joint_pos.shape}, joint_vel={joint_vel.shape}"
    )
    print(f"joint_names={base_env.dof_names}")
    return output


if __name__ == "__main__":
    collect(parse_args())
