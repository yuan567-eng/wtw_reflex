"""Detector V2 的外挂采集环境：每个物理步同步记录 IMU、关节位置和关节速度。"""

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
import numpy as np
import torch

from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv


def compute_body_imu(
    base_quat,
    world_linear_velocity,
    previous_world_linear_velocity,
    world_angular_velocity,
    gravity_world,
    physics_dt,
):
    """由相邻物理步的 base 状态构造机体系比力和角速度。"""
    world_linear_acceleration = (
        world_linear_velocity - previous_world_linear_velocity
    ) / physics_dt
    body_specific_force = quat_rotate_inverse(
        base_quat, world_linear_acceleration - gravity_world
    )
    body_angular_velocity = quat_rotate_inverse(base_quat, world_angular_velocity)
    return torch.cat((body_specific_force, body_angular_velocity), dim=-1)


class ImuJointCollectionEnv(VelocityTrackingEasyEnv):
    """不修改 WTW 源码的 V2 数据采集环境。

    每次策略控制步内部会执行多个物理步。这里在每个物理步结束后同步保存：
    6 维等价 IMU、12 维关节位置和 12 维关节速度，保证三者均为 200 Hz。
    """

    def _init_buffers(self):
        super()._init_buffers()
        self._previous_world_linear_velocity = self.root_states[
            : self.num_envs, 7:10
        ].clone()
        self._physics_step_count = 0
        self._sample_interval_us = int(
            round(float(self.sim_params.dt) * 1_000_000)
        )
        self._gravity_world = torch.tensor(
            self.cfg.sim.gravity, dtype=torch.float, device=self.device
        ).view(1, 3)
        # 外力张量只由 V2 采集器使用，不影响 WTW 原始环境代码。
        self._external_force_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self._disturbance_armed = False
        self._disturbance_start_step = 0
        self._disturbance_end_step = 0
        self._disturbance_recovery_end_step = 0
        self._disturbance_body_force = torch.zeros(3, dtype=torch.float, device=self.device)
        self._clear_step_samples()

    def _clear_step_samples(self):
        """每个策略控制步开始前清空本步内部的物理步采样缓存。"""
        self._imu_step_samples = []
        self._joint_pos_step_samples = []
        self._joint_vel_step_samples = []
        self._joint_torque_step_samples = []
        self._foot_contact_force_step_samples = []
        self._contact_state_step_samples = []
        self._timestamp_step_samples = []
        self._disturbance_step_active = []
        self._disturbance_step_settling = []

    def arm_body_force_disturbance(
        self,
        start_delay_seconds: float,
        duration_seconds: float,
        recovery_seconds: float,
        body_force_xyz,
    ):
        """安排一次机体系外力扰动，标签与后续物理步采样逐点对齐。"""
        physics_dt = float(self.sim_params.dt)
        delay_steps = int(round(start_delay_seconds / physics_dt))
        duration_steps = max(1, int(round(duration_seconds / physics_dt)))
        recovery_steps = max(0, int(round(recovery_seconds / physics_dt)))
        self._disturbance_start_step = self._physics_step_count + delay_steps
        self._disturbance_end_step = self._disturbance_start_step + duration_steps
        self._disturbance_recovery_end_step = (
            self._disturbance_end_step + recovery_steps
        )
        self._disturbance_body_force = torch.as_tensor(
            body_force_xyz, dtype=torch.float, device=self.device
        )
        self._disturbance_armed = True

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "_previous_world_linear_velocity") and len(env_ids) > 0:
            # reset 会突然重设速度；同步历史值，避免下一帧产生伪加速度尖峰。
            self._previous_world_linear_velocity[env_ids] = self.root_states[
                env_ids, 7:10
            ]

    def step(self, actions):
        """复用 WTW 控制流程，并在 decimation 内采集全部物理步状态。"""
        self._clear_step_samples()

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            active, settling = self._apply_scheduled_body_force()
            self._disturbance_step_active.append(active)
            self._disturbance_step_settling.append(settling)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # contact force 不会随着 dof state 一起刷新，必须显式刷新。
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._capture_physics_sample()

        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        self.extras.update(
            {
                "privileged_obs": self.privileged_obs_buf,
                "imu_raw": torch.stack(self._imu_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                "joint_pos": torch.stack(self._joint_pos_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                "joint_vel": torch.stack(self._joint_vel_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                # joint_torque 是当前策略经 PD/actuator model 计算并施加的关节力矩。
                "joint_torque": torch.stack(self._joint_torque_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                # 四个足端的世界坐标接触力；后续 B5 决定是否使用其完整三维量。
                "foot_contact_force": torch.stack(self._foot_contact_force_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                "contact_state": torch.stack(self._contact_state_step_samples, dim=0)
                .detach()
                .cpu()
                .numpy(),
                "timestamp_us": torch.tensor(
                    self._timestamp_step_samples, dtype=torch.int64
                ).numpy(),
                "sample_rate_hz": int(round(1.0 / self.sim_params.dt)),
                "disturbance_active": np.asarray(
                    self._disturbance_step_active, dtype=np.bool_
                ),
                "disturbance_settling": np.asarray(
                    self._disturbance_step_settling, dtype=np.bool_
                ),
            }
        )
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_scheduled_body_force(self):
        """在本物理步施加外力，并返回与本步样本对应的事件标签。"""
        if not self._disturbance_armed:
            return False, False

        step_index = self._physics_step_count
        active = self._disturbance_start_step <= step_index < self._disturbance_end_step
        settling = (
            self._disturbance_end_step
            <= step_index
            < self._disturbance_recovery_end_step
        )

        self._external_force_tensor.zero_()
        if active:
            # 用户输入为机体系力；Isaac Gym 需要世界系力。
            world_force = quat_apply(
                self.base_quat,
                self._disturbance_body_force.view(1, 3).expand(self.num_envs, -1),
            )
            self._external_force_tensor[:, 0, :] = world_force
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._external_force_tensor),
            None,
            gymapi.ENV_SPACE,
        )
        return active, settling

    def _capture_physics_sample(self):
        """在一个 5 ms 物理步结束后，记录与时间戳严格对齐的状态。"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        base_quat = self.root_states[: self.num_envs, 3:7]
        world_linear_velocity = self.root_states[: self.num_envs, 7:10]
        world_angular_velocity = self.root_states[: self.num_envs, 10:13]

        self._imu_step_samples.append(
            compute_body_imu(
                base_quat,
                world_linear_velocity,
                self._previous_world_linear_velocity,
                world_angular_velocity,
                self._gravity_world,
                self.sim_params.dt,
            ).clone()
        )
        # dof_state_tensor 已在本物理步后刷新，直接保存真实关节本体感觉。
        self._joint_pos_step_samples.append(self.dof_pos.clone())
        self._joint_vel_step_samples.append(self.dof_vel.clone())
        self._joint_torque_step_samples.append(self.torques.clone())
        foot_force = self.contact_forces[:, self.feet_indices, :]
        self._foot_contact_force_step_samples.append(foot_force.clone())
        # 与 WTW velocity_tracking extras 中一致：Fz > 1 N 视为接触。
        self._contact_state_step_samples.append((foot_force[..., 2] > 1.0).clone())

        self._physics_step_count += 1
        self._timestamp_step_samples.append(
            self._physics_step_count * self._sample_interval_us
        )
        self._previous_world_linear_velocity.copy_(world_linear_velocity)
