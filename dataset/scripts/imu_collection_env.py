"""固定步态 IMU 与最小外力扰动采集使用的外挂 Isaac Gym 环境。"""

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
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
    """根据相邻物理步的 base 状态构造机体系比力和角速度。"""
    world_linear_acceleration = (
        world_linear_velocity - previous_world_linear_velocity
    ) / physics_dt
    body_specific_force = quat_rotate_inverse(
        base_quat, world_linear_acceleration - gravity_world
    )
    body_angular_velocity = quat_rotate_inverse(base_quat, world_angular_velocity)
    return torch.cat((body_specific_force, body_angular_velocity), dim=-1)


class ImuCollectionEnv(VelocityTrackingEasyEnv):
    """不修改 WTW 源码，在外挂子类中采集每个物理步的六维等价 IMU。"""

    def _init_buffers(self):
        super()._init_buffers()
        self._imu_prev_world_linear_velocity = self.root_states[:self.num_envs, 7:10].clone()
        self._imu_step_samples = []
        self._imu_step_timestamps_us = []
        self._imu_physics_step_count = 0
        # 使用整数采样周期累加时间戳，避免长批次中 float dt 累乘产生 4999/5001 us 抖动。
        self._imu_interval_us = int(round(float(self.sim_params.dt) * 1_000_000))
        self._imu_gravity_world = torch.tensor(
            self.cfg.sim.gravity, dtype=torch.float, device=self.device
        ).view(1, 3)
        self._external_force_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self._disturbance_armed = False
        self._disturbance_start_step = 0
        self._disturbance_end_step = 0
        self._disturbance_recovery_end_step = 0
        self._disturbance_body_force = torch.zeros(3, dtype=torch.float, device=self.device)

    def arm_body_force_disturbance(
        self,
        start_delay_seconds,
        duration_seconds,
        recovery_seconds,
        body_force_xyz,
    ):
        """从当前物理步开始计时，在 base 质心施加一段机体系外力。"""
        physics_dt = float(self.sim_params.dt)
        delay_steps = int(round(start_delay_seconds / physics_dt))
        duration_steps = max(1, int(round(duration_seconds / physics_dt)))
        recovery_steps = max(0, int(round(recovery_seconds / physics_dt)))
        self._disturbance_start_step = self._imu_physics_step_count + delay_steps
        self._disturbance_end_step = self._disturbance_start_step + duration_steps
        self._disturbance_recovery_end_step = self._disturbance_end_step + recovery_steps
        self._disturbance_body_force = torch.tensor(
            body_force_xyz, dtype=torch.float, device=self.device
        )
        self._disturbance_armed = True

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "_imu_prev_world_linear_velocity") and len(env_ids) > 0:
            # reset 会瞬间重设 base 速度；同步历史值，避免产生虚假的加速度尖峰。
            self._imu_prev_world_linear_velocity[env_ids] = self.root_states[env_ids, 7:10]

    def step(self, actions):
        """复用 WTW 控制语义，同时在 decimation 内保留全部物理步 IMU。"""
        self._imu_step_samples = []
        self._imu_step_timestamps_us = []
        self._disturbance_step_active = []
        self._disturbance_step_settling = []

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            active, settling = self._apply_scheduled_body_force()
            self._disturbance_step_active.append(active)
            self._disturbance_step_settling.append(settling)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._capture_imu_after_simulation_step()

        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        imu_raw = torch.stack(self._imu_step_samples, dim=0)
        timestamp_us = torch.tensor(self._imu_step_timestamps_us, dtype=torch.int64)
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "imu_raw": imu_raw.detach().cpu().numpy(),
            "imu_timestamp_us": timestamp_us.numpy(),
            "imu_sample_rate_hz": int(round(1.0 / self.sim_params.dt)),
            "disturbance_active": torch.tensor(
                self._disturbance_step_active, dtype=torch.bool
            ).numpy(),
            "disturbance_settling": torch.tensor(
                self._disturbance_step_settling, dtype=torch.bool
            ).numpy(),
        })
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_scheduled_body_force(self):
        """在当前物理步施加外力，并返回与本步 IMU 对齐的事件状态。"""
        if not self._disturbance_armed:
            return False, False

        step_index = self._imu_physics_step_count
        active = self._disturbance_start_step <= step_index < self._disturbance_end_step
        settling = self._disturbance_end_step <= step_index < self._disturbance_recovery_end_step

        self._external_force_tensor.zero_()
        if active:
            body_force = self._disturbance_body_force.view(1, 3).expand(self.num_envs, -1)
            world_force = quat_apply(self.base_quat, body_force)
            self._external_force_tensor[:, 0, :] = world_force
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._external_force_tensor),
            None,
            gymapi.ENV_SPACE,
        )
        return active, settling

    def _capture_imu_after_simulation_step(self):
        """在 5 ms 物理步结束后计算一帧机体系 IMU。"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        base_quat = self.root_states[:self.num_envs, 3:7]
        world_linear_velocity = self.root_states[:self.num_envs, 7:10]
        world_angular_velocity = self.root_states[:self.num_envs, 10:13]

        self._imu_step_samples.append(
            compute_body_imu(
                base_quat=base_quat,
                world_linear_velocity=world_linear_velocity,
                previous_world_linear_velocity=self._imu_prev_world_linear_velocity,
                world_angular_velocity=world_angular_velocity,
                gravity_world=self._imu_gravity_world,
                physics_dt=self.sim_params.dt,
            ).clone()
        )
        self._imu_physics_step_count += 1
        timestamp_us = self._imu_physics_step_count * self._imu_interval_us
        self._imu_step_timestamps_us.append(timestamp_us)
        self._imu_prev_world_linear_velocity.copy_(world_linear_velocity)
