"""按 Detector V2 的 B0/B1/B2 实验设计批量采集正式原始数据。

三个预设均将多个独立 10 秒 episode 合并为一个 npz；episode_id 用于阻止
后续 0.5 秒滑动窗口跨 episode。TCN 与 GRU 将使用完全相同的三份原始数据。
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
from collect_normal import (
    GAIT_MAPPING,
    find_logdir,
    load_policy,
    load_training_config,
    set_motion_command,
)


CONTROL_HZ = 50
EPISODE_SECONDS = 10.0
EPISODE_STEPS = int(CONTROL_HZ * EPISODE_SECONDS)
INITIAL_COMMAND = np.array([0.5, 0.0, 0.0, 3.0], dtype=np.float32)
COMMAND_RANGES = np.array(
    [[-1.0, 1.0], [-0.6, 0.6], [-1.0, 1.0], [2.0, 4.0]], dtype=np.float32
)

PRESETS = {
    # 6 * 10 s = 1 min；仅采集随机运动下的正常状态。
    "train_normal_1min": {"episodes": 6, "disturbance": False, "random_commands": True},
    # 24 * 10 s = 4 min；每个 episode 一次 40--80 N 的随机侧向扰动。
    "train_abnormal_4min": {"episodes": 24, "disturbance": True, "random_commands": True},
    # 30 * 10 s = 5 min；每 10 s 一次扰动，按原/增强/弱扰动三段组成。
    "test_5min": {"episodes": 30, "disturbance": True, "random_commands": False},
    # 独立泛化测试：12 段复杂正常运动 + 18 段未见力度的扰动运动。
    # 不可用于训练、阈值选择或模型选型。
    "test_b0_generalization_5min": {
        "episodes": 30, "disturbance": "mixed", "random_commands": True,
    },
    # B3-delta 的最终冻结测试：协议与泛化测试相同，但必须使用新的双随机种子。
    # 该数据只能在模型和阈值固定后采集一次，不得再用于调参。
    "final_b3_delta_5min": {
        "episodes": 30, "disturbance": "mixed", "random_commands": True,
    },
    # 仅验证新增物理特征的数值、shape 与时间同步；不用于训练或任何性能指标。
    "physics_feature_smoke_10s": {"episodes": 1, "disturbance": False, "random_commands": False},
}


class RandomCommandScheduler:
    """在 50 Hz 控制周期随机采样目标命令，并线性平滑过渡。"""

    def __init__(self, seed: int, dt: float):
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.current = INITIAL_COMMAND.copy()
        self.start = self.current.copy()
        self.target = self.current.copy()
        self.transition_steps = max(1, int(round(0.5 / dt)))
        self.transition_index = self.transition_steps
        self.hold_steps_remaining = 0

    def _start_transition(self):
        self.start = self.current.copy()
        self.target = self.rng.uniform(
            COMMAND_RANGES[:, 0], COMMAND_RANGES[:, 1]
        ).astype(np.float32)
        self.transition_index = 0

    def _sample_hold_steps(self):
        return max(1, int(round(self.rng.uniform(1.0, 3.0) / self.dt)))

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


def parse_args():
    parser = argparse.ArgumentParser(description="采集 Detector V2 B0/B1/B2 正式实验数据")
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--gait", choices=tuple(GAIT_MAPPING), default="trotting")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--command-seed", type=int, default=0)
    parser.add_argument("--disturbance-seed", type=int, default=100)
    parser.add_argument(
        "--label", default="gait-conditioned-agility/pretrain-go2/train",
        help="runs/ 下保存 WTW JIT 策略的父目录",
    )
    parser.add_argument("--sim-device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("warmup-steps 不可为负")
    return args


def disturbance_parameters(preset: str, episode_index: int, rng):
    """按预设生成施力时间和机体系 y 向力；正负方向按 episode 平衡。"""
    direction = 1.0 if episode_index % 2 == 0 else -1.0
    if preset == "train_abnormal_4min":
        magnitude = float(rng.uniform(40.0, 80.0))
        start_seconds = float(rng.uniform(3.0, 7.0))
    elif preset == "test_5min":
        if episode_index < 6:
            force_range = (40.0, 80.0)
        elif episode_index < 18:
            force_range = (60.0, 120.0)
        else:
            force_range = (20.0, 40.0)
        magnitude = float(rng.uniform(*force_range))
        # 每段 10 秒 episode 在第 5 秒固定扰动，构成“每 10 秒一次”。
        start_seconds = 5.0
    elif preset in ("test_b0_generalization_5min", "final_b3_delta_5min"):
        # 0--11 是无扰动的复杂正常运动；该函数只会为 12--29 调用。
        if 12 <= episode_index < 18:
            force_range = (20.0, 35.0)  # 训练范围 40--80 N 以下的未见弱扰动
        elif 18 <= episode_index < 24:
            force_range = (35.0, 45.0)  # 跨越训练下边界的临界扰动
        elif 24 <= episode_index < 30:
            force_range = (80.0, 110.0) # 训练范围以上的未见强扰动
        else:
            raise ValueError("复杂正常运动 episode 不应生成扰动参数")
        magnitude = float(rng.uniform(*force_range))
        start_seconds = 5.0
    else:
        raise ValueError(f"{preset} 不包含扰动")
    return start_seconds, direction * magnitude


def has_disturbance(preset: str, episode_index: int) -> bool:
    """判断某个 episode 是否真的施力；mixed 预设保留无扰动 hard negative。"""
    if preset in ("test_b0_generalization_5min", "final_b3_delta_5min"):
        return episode_index >= 12
    return bool(PRESETS[preset]["disturbance"])


def scenario_id(preset: str, episode_index: int) -> int:
    """泛化测试的分段编码；-1 表示普通训练/旧测试预设。"""
    if preset not in ("test_b0_generalization_5min", "final_b3_delta_5min"):
        return -1
    if episode_index < 12:
        return 0  # complex_normal
    if episode_index < 18:
        return 1  # weak_20_35N
    if episode_index < 24:
        return 2  # boundary_35_45N
    return 3      # strong_80_110N


def step_policy(env, base_env, policy, obs, args, command):
    set_motion_command(base_env, args, command)
    detector_command = (
        base_env.commands[0, (0, 1, 2, 4)]
        .detach().cpu().numpy().astype(np.float32, copy=True)
    )
    with torch.no_grad():
        actions = policy(obs)
    obs, _, done, info = env.step(actions)
    if bool(done[0].item()):
        raise RuntimeError("采集期间机器人发生 reset；本 batch 不保存")
    return obs, info, detector_command


def collect_one_episode(args, preset, env, base_env, policy, episode_index, force_rng):
    """采集一个 10 秒 episode，并返回可与其他 episode 拼接的逐帧数组。"""
    obs = env.reset()
    for _ in tqdm(range(args.warmup_steps), desc=f"warmup {episode_index + 1}", leave=False):
        obs, _, _ = step_policy(env, base_env, policy, obs, args, INITIAL_COMMAND)

    config = PRESETS[preset]
    disturbance_force_y = 0.0
    disturbed = has_disturbance(preset, episode_index)
    if disturbed:
        start_seconds, disturbance_force_y = disturbance_parameters(
            preset, episode_index, force_rng
        )
        base_env.arm_body_force_disturbance(
            start_delay_seconds=start_seconds,
            duration_seconds=0.1,
            recovery_seconds=0.5,
            body_force_xyz=[0.0, disturbance_force_y, 0.0],
        )

    scheduler = (
        RandomCommandScheduler(args.command_seed + episode_index, 1.0 / CONTROL_HZ)
        if config["random_commands"] else None
    )
    chunks = {
        "timestamp_us": [], "imu_raw": [], "commands": [], "joint_pos": [],
        "joint_vel": [], "joint_torque": [], "foot_contact_force": [],
        "contact_state": [], "detector_target": [], "settling": [],
    }
    for _ in tqdm(range(EPISODE_STEPS), desc=f"collect {episode_index + 1}", leave=False):
        command = scheduler.next() if scheduler is not None else INITIAL_COMMAND
        obs, info, detector_command = step_policy(env, base_env, policy, obs, args, command)
        frame_count = len(info["timestamp_us"])
        chunks["timestamp_us"].append(info["timestamp_us"])
        for name in (
            "imu_raw", "joint_pos", "joint_vel", "joint_torque",
            "foot_contact_force", "contact_state",
        ):
            chunks[name].append(info[name][:, 0, :])
        chunks["commands"].append(np.repeat(detector_command[None, :], frame_count, axis=0))
        chunks["detector_target"].append(info["disturbance_active"])
        chunks["settling"].append(info["disturbance_settling"])

    episode = {
        "timestamp_us": np.concatenate(chunks["timestamp_us"]).astype(np.int64, copy=False),
        "imu_raw": np.concatenate(chunks["imu_raw"]).astype(np.float32, copy=False),
        "commands": np.concatenate(chunks["commands"]).astype(np.float32, copy=False),
        "joint_pos": np.concatenate(chunks["joint_pos"]).astype(np.float32, copy=False),
        "joint_vel": np.concatenate(chunks["joint_vel"]).astype(np.float32, copy=False),
        "joint_torque": np.concatenate(chunks["joint_torque"]).astype(np.float32, copy=False),
        "foot_contact_force": np.concatenate(chunks["foot_contact_force"]).astype(np.float32, copy=False),
        "contact_state": np.concatenate(chunks["contact_state"]).astype(np.bool_, copy=False),
        "detector_target": np.concatenate(chunks["detector_target"]).astype(np.int8, copy=False),
        "settling": np.concatenate(chunks["settling"]).astype(np.bool_, copy=False),
    }
    expected_samples = EPISODE_STEPS * Cfg.control.decimation
    if episode["imu_raw"].shape != (expected_samples, 6):
        raise RuntimeError("IMU 样本数错误")
    if episode["joint_pos"].shape != (expected_samples, 12) or episode["joint_vel"].shape != (expected_samples, 12):
        raise RuntimeError("关节本体感觉样本数错误")
    if episode["joint_torque"].shape != (expected_samples, 12):
        raise RuntimeError("关节力矩样本数错误")
    if episode["foot_contact_force"].shape != (expected_samples, 4, 3):
        raise RuntimeError("足端接触力样本数错误")
    if episode["contact_state"].shape != (expected_samples, 4):
        raise RuntimeError("足端接触状态样本数错误")
    if not disturbed and (episode["detector_target"].any() or episode["settling"].any()):
        raise RuntimeError("正常 episode 出现了扰动标签")
    if disturbed and (int(episode["detector_target"].sum()) != 20 or int(episode["settling"].sum()) != 100):
        raise RuntimeError("扰动或恢复期标签数错误，期望 target=20、settling=100")
    return episode, disturbance_force_y


def resolve_output(args):
    if args.output is not None:
        return args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    return PROJECT_ROOT / "detector_v2/dataset/raw" / f"{args.preset}.npz"


def collect(args):
    preset_config = PRESETS[args.preset]
    logdir = find_logdir(args.label)
    # 传入预设采集长度，让 WTW 内部 episode 时限高于每个 10 秒片段。
    args.steps = EPISODE_STEPS
    load_training_config(logdir, args)
    # WTW 构造函数的第三个位置参数是 num_envs；cfg 必须使用关键字传入。
    base_env = ImuJointCollectionEnv(
        sim_device=args.sim_device, headless=args.headless, cfg=Cfg
    )
    env = HistoryWrapper(base_env)
    policy = load_policy(logdir)
    force_rng = np.random.default_rng(args.disturbance_seed)

    arrays = []
    force_values = []
    for episode_index in range(preset_config["episodes"]):
        episode, force_y = collect_one_episode(
            args, args.preset, env, base_env, policy, episode_index, force_rng
        )
        arrays.append(episode)
        force_values.append(force_y)
        print(
            f"episode {episode_index + 1}/{preset_config['episodes']}: "
            f"target={int(episode['detector_target'].sum())}, force_y={force_y:.2f}N"
        )

    merged = {name: np.concatenate([item[name] for item in arrays]) for name in arrays[0]}
    samples_per_episode = EPISODE_STEPS * Cfg.control.decimation
    episode_indices = np.repeat(
        np.arange(preset_config["episodes"], dtype=np.int32), samples_per_episode
    )
    unique_episode_id = int(time.time_ns() // 1_000)
    merged.update(
        {
            "gait_id": np.full(len(merged["timestamp_us"]), GAIT_MAPPING[args.gait], dtype=np.int8),
            "switching": np.zeros(len(merged["timestamp_us"]), dtype=np.bool_),
            "loss_mask": ~merged["settling"],
            "episode_id": unique_episode_id + episode_indices.astype(np.int64),
            "collection_episode_index": episode_indices,
            # 泛化测试的正常/弱/临界/强分段，后续评估时不依赖外力绝对值猜测。
            "scenario_id": np.repeat(
                np.asarray(
                    [scenario_id(args.preset, index) for index in range(preset_config["episodes"])],
                    dtype=np.int8,
                ),
                samples_per_episode,
            ),
            "scenario_names": np.asarray(
                ["complex_normal", "weak_20_35N", "boundary_35_45N", "strong_80_110N"],
                dtype="<U20",
            ) if args.preset in ("test_b0_generalization_5min", "final_b3_delta_5min") else np.asarray([], dtype="<U20"),
            # 仅为后续按施力强度分段分析，不作为网络输入特征。
            "disturbance_force_y": np.repeat(np.asarray(force_values, dtype=np.float32), samples_per_episode),
            "joint_names": np.asarray(base_env.dof_names, dtype="<U32"),
            "joint_default_pos": base_env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32),
            "sample_rate_hz": np.float32(round(1.0 / Cfg.sim.dt)),
        }
    )
    output = resolve_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    print(f"已保存：{output}")
    print(
        f"preset={args.preset}, episodes={preset_config['episodes']}, "
        f"samples={len(merged['timestamp_us'])}, "
        f"target={int(merged['detector_target'].sum())}, "
        f"settling={int(merged['settling'].sum())}"
    )
    return output


if __name__ == "__main__":
    collect(parse_args())
