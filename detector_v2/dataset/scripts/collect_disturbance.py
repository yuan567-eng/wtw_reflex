"""采集 Detector V2 的单次机身外力扰动数据。

输出字段与 ``collect_normal.py`` 完全一致；仅 ``detector_target``、
``settling`` 与 ``loss_mask`` 会随本次扰动而变化。
"""

from __future__ import annotations

import argparse
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

# 只复用 WTW 策略加载和命令写入，不共享任何 TCN / GRU 特征或训练逻辑。
from collect_normal import (
    GAIT_MAPPING,
    find_logdir,
    load_policy,
    load_training_config,
    set_motion_command,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="采集 Detector V2 的单次扰动 200 Hz IMU 与关节本体感觉数据"
    )
    parser.add_argument("--gait", choices=tuple(GAIT_MAPPING), default="trotting")
    parser.add_argument("--steps", type=int, default=500, help="采集的 50 Hz 策略步数")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--gait-frequency", type=float, default=3.0)
    parser.add_argument("--force-y", type=float, default=60.0, help="机体系 y 方向外力，单位 N")
    parser.add_argument("--start-seconds", type=float, default=5.0, help="采集开始后多久施力")
    parser.add_argument("--duration-seconds", type=float, default=0.1, help="施力持续时间")
    parser.add_argument("--recovery-seconds", type=float, default=0.5, help="施力结束后的忽略期")
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
    if args.duration_seconds <= 0 or args.recovery_seconds < 0 or args.start_seconds < 0:
        parser.error("扰动开始时间和恢复时间不可为负，持续时间必须大于 0")
    return args


def step_policy(env, base_env, policy, obs, args):
    """执行一次策略步，返回同一控制周期对应的四帧传感器样本。"""
    set_motion_command(base_env, args)
    detector_command = (
        base_env.commands[0, (0, 1, 2, 4)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
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
        / "detector_v2/dataset/raw/abnormal"
        / args.gait
        / f"abnormal_{identifier}.npz"
    )


def collect(args):
    logdir = find_logdir(args.label)
    load_training_config(logdir, args)
    collection_seconds = args.steps * Cfg.control.decimation * Cfg.sim.dt
    event_end = args.start_seconds + args.duration_seconds + args.recovery_seconds
    if event_end > collection_seconds + 1e-9:
        raise ValueError(
            f"扰动与恢复期结束于 {event_end:.3f}s，超出采集时长 {collection_seconds:.3f}s"
        )

    base_env = ImuJointCollectionEnv(
        sim_device=args.sim_device, headless=args.headless, cfg=Cfg
    )
    env = HistoryWrapper(base_env)
    policy = load_policy(logdir)
    obs = env.reset()

    for _ in tqdm(range(args.warmup_steps), desc="warmup"):
        obs, _, _ = step_policy(env, base_env, policy, obs, args)

    # 外力以机体系 y 方向定义；正负号对应机器人左右相反方向的推力。
    base_env.arm_body_force_disturbance(
        start_delay_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        recovery_seconds=args.recovery_seconds,
        body_force_xyz=[0.0, args.force_y, 0.0],
    )

    chunks = {
        "imu_raw": [], "joint_pos": [], "joint_vel": [], "timestamp_us": [],
        "commands": [], "detector_target": [], "settling": [],
    }
    for _ in tqdm(range(args.steps), desc="collect"):
        obs, info, command = step_policy(env, base_env, policy, obs, args)
        frame_count = len(info["timestamp_us"])
        for name in ("imu_raw", "joint_pos", "joint_vel", "timestamp_us"):
            chunks[name].append(info[name][:, 0, :] if name != "timestamp_us" else info[name])
        chunks["commands"].append(np.repeat(command[None, :], frame_count, axis=0))
        chunks["detector_target"].append(info["disturbance_active"])
        chunks["settling"].append(info["disturbance_settling"])

    episode = {
        name: np.concatenate(values).astype(
            np.int64 if name == "timestamp_us" else np.float32, copy=False
        )
        for name, values in chunks.items()
        if name not in ("detector_target", "settling")
    }
    episode["detector_target"] = np.concatenate(chunks["detector_target"]).astype(np.int8)
    episode["settling"] = np.concatenate(chunks["settling"]).astype(np.bool_)

    sample_count = len(episode["timestamp_us"])
    expected_samples = args.steps * Cfg.control.decimation
    expected_shapes = {
        "imu_raw": (expected_samples, 6),
        "joint_pos": (expected_samples, 12),
        "joint_vel": (expected_samples, 12),
        "commands": (expected_samples, 4),
        "detector_target": (expected_samples,),
        "settling": (expected_samples,),
    }
    for name, expected in expected_shapes.items():
        if episode[name].shape != expected:
            raise RuntimeError(f"{name} shape 错误：得到 {episode[name].shape}，期望 {expected}")
    if not all(np.isfinite(episode[name]).all() for name in ("imu_raw", "joint_pos", "joint_vel", "commands")):
        raise RuntimeError("采集数据含有 NaN 或 Inf")

    intervals = np.diff(episode["timestamp_us"])
    expected_interval_us = int(round(Cfg.sim.dt * 1_000_000))
    if len(intervals) and not np.all(intervals == expected_interval_us):
        raise RuntimeError("时间戳不连续，未达到预期 200 Hz 物理步采样频率")
    expected_target = max(1, int(round(args.duration_seconds / Cfg.sim.dt)))
    expected_settling = int(round(args.recovery_seconds / Cfg.sim.dt))
    if int(episode["detector_target"].sum()) != expected_target:
        raise RuntimeError("扰动标签数与施力时长不一致")
    if int(episode["settling"].sum()) != expected_settling:
        raise RuntimeError("恢复期标签数与设定不一致")

    episode_id = int(time.time_ns() // 1_000)
    episode.update(
        {
            "gait_id": np.full(sample_count, GAIT_MAPPING[args.gait], dtype=np.int8),
            "switching": np.zeros(sample_count, dtype=np.bool_),
            "loss_mask": ~episode["settling"],
            "episode_id": np.full(sample_count, episode_id, dtype=np.int64),
            "joint_names": np.asarray(base_env.dof_names, dtype="<U32"),
            "joint_default_pos": base_env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32),
            "sample_rate_hz": np.float32(1_000_000.0 / expected_interval_us),
        }
    )
    output = resolve_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **episode)
    print(f"已保存：{output}")
    print(
        f"samples={sample_count}, target={int(episode['detector_target'].sum())}, "
        f"settling={int(episode['settling'].sum())}, rate={episode['sample_rate_hz']:.1f}Hz"
    )
    return output


if __name__ == "__main__":
    collect(parse_args())
