"""统一的正常运动与突发扰动 Isaac Gym 环境。

本类是目前 WTW/Go2 的适配实现。它将以前 ``DisturbanceEnv`` 与
``SuddenGeometryEnv`` 的有效行为集中到一处，但不修改 WTW、Detector 或
Reflex 的任何算法。
"""

from __future__ import annotations

import math
from typing import Optional

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply
import torch

from snn.env.normal_walk_env import NormalWalkEnv
from .scenario_config import SimEnvConfig


class UnifiedScenarioEnv(NormalWalkEnv):
    """一个环境类，通过 ``scenario`` 参数切换六种可重复场景。"""

    task_config_keyword = "scenario_cfg"
    # sim_env 只产生场景和事件真值。检测器应由后续具体项目按需外挂，
    # 不能让这个通用环境依赖某一个 Reflex 方案。
    use_reflex_detector = False

    def __init__(self, *args, scenario_cfg: Optional[SimEnvConfig] = None, **kwargs):
        self.scenario_cfg = scenario_cfg or SimEnvConfig()
        c = self.scenario_cfg
        if c.scenario == "sudden_geometry" and c.num_envs != 1:
            raise ValueError("sudden_geometry 当前只支持 num_envs=1。")

        self.manual_trigger_requested = False
        self.walking_released = c.scenario != "foothold_loss"
        self._scenario_ready = False
        super().__init__(*args, normal_cfg=c, **kwargs)

        self._init_scenario_buffers()
        self._scenario_ready = True
        if self.viewer is not None and c.scenario != "normal":
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "trigger_sim_scenario")
            print(f"[sim_env] 当前场景：{c.scenario}；按 T 主动触发事件。")
        self._reset_scenario_state(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------
    # 场景创建：普通/物理事件保持平地；踩空与突发几何额外加入真实碰撞体。
    # ------------------------------------------------------------------
    def create_sim(self):
        super().create_sim()
        c = self.scenario_cfg
        if c.scenario == "foothold_loss":
            self._create_drop_platform()
        elif c.scenario == "sudden_geometry":
            self._create_dynamic_geometry()

    def _create_drop_platform(self):
        d = self.scenario_cfg.foothold_loss
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        platform = self.gym.create_box(self.sim, d.platform_size_xy[0], d.platform_size_xy[1], d.platform_height, options)
        self.platform_handles = []
        for env_id, env_handle in enumerate(self.envs):
            origin = self.env_origins[env_id].detach().cpu().tolist()
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(origin[0], origin[1], d.platform_height / 2.0)
            handle = self.gym.create_actor(env_handle, platform, pose, "foothold_loss_platform", env_id, 0, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_handle, handle)
            for prop in props:
                prop.friction = float(self.cfg.terrain.static_friction)
            self.gym.set_actor_rigid_shape_properties(env_handle, handle, props)
            self.platform_handles.append(handle)

    def _create_dynamic_geometry(self):
        """创建可下沉拼块和隐藏圆环；只创建一次，T 时再修改位置。"""
        c = self.scenario_cfg
        self.tile_actor_indices, self.tile_centers_xy, self.ring_actor_indices = [], [], []
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        g = c.ground_drop
        count_x = max(1, round(g.platform_size_xy[0] / g.tile_size))
        count_y = max(1, round(g.platform_size_xy[1] / g.tile_size))
        tile_x, tile_y = g.platform_size_xy[0] / count_x, g.platform_size_xy[1] / count_y
        tile = self.gym.create_box(self.sim, tile_x, tile_y, g.platform_height, options)
        origin = self.env_origins[0].detach().cpu().tolist()
        for ix in range(count_x):
            for iy in range(count_y):
                x = origin[0] + (ix + .5 - count_x / 2.0) * tile_x
                y = origin[1] + (iy + .5 - count_y / 2.0) * tile_y
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x, y, g.platform_height / 2.0)
                actor = self.gym.create_actor(self.envs[0], tile, pose, "geometry_platform_tile", 0, 0, 0)
                self.tile_actor_indices.append(self.gym.get_actor_index(self.envs[0], actor, gymapi.DOMAIN_SIM))
                self.tile_centers_xy.append((x, y))

        r = c.obstacle_ring
        tangent = 2.0 * math.pi * r.max_inner_radius / r.segments * 1.15
        segment = self.gym.create_box(self.sim, tangent, r.thickness, r.height, options)
        for _ in range(r.segments):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(origin[0], origin[1], -r.height)
            actor = self.gym.create_actor(self.envs[0], segment, pose, "hidden_ring_segment", 0, 0, 0)
            self.ring_actor_indices.append(self.gym.get_actor_index(self.envs[0], actor, gymapi.DOMAIN_SIM))

    def _init_buffers(self):
        """突发几何创建了额外 actor，日常控制时只向 WTW 暴露机器人状态。"""
        super()._init_buffers()
        if self.scenario_cfg.scenario != "sudden_geometry":
            return
        self.all_actor_root_states = self.root_states
        self.root_states = self.all_actor_root_states[:1]
        self.tile_actor_indices_tensor = torch.tensor(self.tile_actor_indices, dtype=torch.int32, device=self.device)
        self.ring_actor_indices_tensor = torch.tensor(self.ring_actor_indices, dtype=torch.int32, device=self.device)
        self.tile_centers_tensor = torch.tensor(self.tile_centers_xy, dtype=torch.float, device=self.device)

    # ------------------------------------------------------------------
    # 回合、按键及事件状态。
    # ------------------------------------------------------------------
    def _init_scenario_buffers(self):
        n = self.num_envs
        self.event_started = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.event_finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.event_trigger_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.event_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.event_end_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.event_target_leg_mask = torch.zeros((n, 4), dtype=torch.bool, device=self.device)
        self.fault_target_action = torch.zeros(n, self.num_actions, device=self.device)
        self.stable_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.recovery_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.external_force_tensor = torch.zeros((n, self.num_bodies, 3), device=self.device)
        self.external_position_tensor = torch.zeros((n, self.num_bodies, 3), device=self.device)
        self.geometry_event_kind = "none"

    def _reset_scenario_state(self, env_ids):
        self.event_started[env_ids] = False
        self.event_finished[env_ids] = False
        self.event_trigger_count[env_ids] = 0
        self.event_step[env_ids] = -1
        self.event_end_step[env_ids] = -1
        self.recovery_step[env_ids] = -1
        self.stable_steps[env_ids] = 0
        self.event_target_leg_mask[env_ids] = False
        scenario = self.scenario_cfg.scenario
        legs = self._selected_legs()
        if scenario in ("slip", "leg_fault"):
            if legs == (-1,):
                random_legs = torch.randint(0, 4, (len(env_ids),), device=self.device)
                self.event_target_leg_mask[env_ids, random_legs] = True
            else:
                self.event_target_leg_mask[env_ids[:, None], torch.tensor(legs, device=self.device)] = True

    def _selected_legs(self):
        c = self.scenario_cfg
        legs = c.slip.target_legs if c.scenario == "slip" else c.leg_fault.target_legs
        if legs != (-1,) and (not legs or any(leg < 0 or leg > 3 for leg in legs)):
            raise ValueError("target_legs 只能为 (-1,) 或由 0--3 组成的腿编号元组。")
        return tuple(legs)

    def reset_idx(self, env_ids):
        dynamic = self.scenario_cfg.scenario == "sudden_geometry" and hasattr(self, "all_actor_root_states")
        if dynamic:
            self.root_states = self.all_actor_root_states
        super().reset_idx(env_ids)
        if dynamic:
            self.root_states = self.all_actor_root_states[:1]
        if not self._scenario_ready or len(env_ids) == 0:
            return
        # 只有 slip / kick 用到了 force tensor。突发几何额外创建了许多 actor，
        # 此时 Isaac Gym 要求的 force tensor 行数也会改变；对它清空一个
        # “仅机器人刚体”的张量反而会报 shape 错误，因此不调用该接口。
        if self.scenario_cfg.scenario in ("slip", "kick"):
            self._clear_external_force()
        self._restore_robot_friction(env_ids)
        self.manual_trigger_requested = False
        self.walking_released = self.scenario_cfg.scenario != "foothold_loss"
        if self.scenario_cfg.scenario == "foothold_loss":
            self.root_states[env_ids, 2] += self.scenario_cfg.foothold_loss.platform_height
            self.root_states[env_ids, 7:13] = 0.0
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids.to(torch.int32)), len(env_ids),
            )
        elif dynamic:
            self._reset_dynamic_geometry(env_ids)
        self._reset_scenario_state(env_ids)

    def _reset_dynamic_geometry(self, env_ids):
        """每回合将机器人放回平台顶面，并把坑和圆环隐藏回初态。"""
        if len(env_ids) == 0:
            return
        g, r = self.scenario_cfg.ground_drop, self.scenario_cfg.obstacle_ring
        self.root_states[env_ids, 2] += g.platform_height
        self.root_states[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.all_actor_root_states),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)), len(env_ids),
        )
        self.all_actor_root_states[self.tile_actor_indices_tensor, 2] = g.platform_height / 2.0
        self.all_actor_root_states[self.ring_actor_indices_tensor, 2] = -r.height
        ids = torch.cat((self.tile_actor_indices_tensor, self.ring_actor_indices_tensor))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.all_actor_root_states), gymtorch.unwrap_tensor(ids), len(ids)
        )
        self.geometry_event_kind = "none"

    def _set_fixed_commands(self, env_ids):
        super()._set_fixed_commands(env_ids)
        if self.scenario_cfg.scenario == "foothold_loss" and not self.walking_released:
            self.commands[env_ids, :3] = 0.0

    def _handle_experiment_viewer_event(self, event):
        if event.action != "trigger_sim_scenario" or event.value <= 0:
            return False
        scenario = self.scenario_cfg.scenario
        if scenario == "normal":
            return True
        if scenario in ("slip", "kick"):
            self.manual_trigger_requested = True
            print(f"[sim_env] 已收到 T：追加一次 {scenario} 脉冲。")
        elif scenario == "sudden_geometry":
            already = bool(torch.any(self.event_trigger_count > 0))
            if self.scenario_cfg.allow_multiple_geometry_triggers or not already:
                self.manual_trigger_requested = True
                print("[sim_env] 已收到 T：下一控制步触发突发几何事件。")
        elif not self.manual_trigger_requested and not torch.any(self.event_started):
            self.manual_trigger_requested = True
            print(f"[sim_env] 已收到 T：准备触发 {scenario}。")
        return True

    # ------------------------------------------------------------------
    # 每控制步更新事件。跌倒不提前重置，便于收集恢复和失败证据。
    # ------------------------------------------------------------------
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self._scenario_ready:
            self._update_scenario()
            self._update_recovery_metric()

    def _update_scenario(self):
        c, scenario = self.scenario_cfg, self.scenario_cfg.scenario
        if self.manual_trigger_requested:
            self.manual_trigger_requested = False
            if scenario == "foothold_loss":
                self.walking_released = True
                print("[sim_env] 踩空场景已放行：机器人将从平台边缘自然走落。")
            elif scenario == "sudden_geometry":
                self._trigger_geometry_event()
            else:
                self._start_event(torch.arange(self.num_envs, device=self.device))

        if scenario == "foothold_loss" and self.walking_released and not torch.any(self.event_started):
            falling = self.base_pos[:, 2] < (c.foothold_loss.platform_height + .22)
            ids = torch.nonzero(falling, as_tuple=False).flatten()
            if len(ids):
                self._start_event(ids)

        active = self.event_started & (~self.event_finished)
        ids = torch.nonzero(active, as_tuple=False).flatten()
        if len(ids):
            if scenario == "kick":
                self._apply_kick(ids)
            elif scenario == "slip":
                self._apply_slip(ids)
        end_ids = torch.nonzero(active & (self.episode_length_buf >= self.event_end_step), as_tuple=False).flatten()
        if len(end_ids):
            if scenario == "slip":
                self._restore_robot_friction(end_ids)
            if scenario in ("slip", "kick"):
                self._clear_external_force()
            self.event_finished[end_ids] = True

    def _start_event(self, env_ids):
        c, scenario = self.scenario_cfg, self.scenario_cfg.scenario
        duration = c.slip.duration_s if scenario == "slip" else c.kick.duration_s
        self.event_started[env_ids] = True
        self.event_finished[env_ids] = False
        self.event_trigger_count[env_ids] += 1
        self.event_step[env_ids] = self.episode_length_buf[env_ids]
        self.recovery_step[env_ids] = -1
        self.stable_steps[env_ids] = 0
        self.event_end_step[env_ids] = (
            self.max_episode_length + 1 if scenario == "leg_fault"
            else self.event_step[env_ids] + max(1, round(duration / self.dt))
        )
        if scenario == "slip":
            self._set_target_foot_friction(env_ids, c.slip.foot_friction)
        elif scenario == "leg_fault":
            self._capture_leg_fault_targets(env_ids)
        print(f"[sim_env] 事件已触发：{scenario}")

    # ------------------------------------------------------------------
    # 四类物理扰动的实现。
    # ------------------------------------------------------------------
    def _apply_kick(self, env_ids):
        c = self.scenario_cfg.kick
        self.external_force_tensor.zero_()
        self.external_position_tensor.zero_()
        roll, pitch, yaw = c.direction_rpy
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        direction_body = torch.tensor([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], device=self.device)
        direction_world = quat_apply(self.base_quat[env_ids], direction_body.expand(len(env_ids), -1))
        self.external_force_tensor[env_ids, 0] = c.force_magnitude_n * direction_world
        offset_body = torch.tensor(c.application_offset_xyz, device=self.device)
        self.external_position_tensor[env_ids, 0] = self.base_pos[env_ids] + quat_apply(self.base_quat[env_ids], offset_body.expand(len(env_ids), -1))
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim, gymtorch.unwrap_tensor(self.external_force_tensor), gymtorch.unwrap_tensor(self.external_position_tensor), gymapi.ENV_SPACE
        )

    def _apply_slip(self, env_ids):
        c = self.scenario_cfg.slip
        self.external_force_tensor.zero_()
        self.external_position_tensor.zero_()
        rows, legs = torch.nonzero(self.event_target_leg_mask[env_ids], as_tuple=True)
        force_env_ids, bodies = env_ids[rows], self.feet_indices[legs]
        force = torch.tensor((*c.tangential_force_xy, 0.0), device=self.device)
        self.external_force_tensor[force_env_ids, bodies] = force
        self.external_position_tensor[force_env_ids, bodies] = self.foot_positions[force_env_ids, legs]
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim, gymtorch.unwrap_tensor(self.external_force_tensor), gymtorch.unwrap_tensor(self.external_position_tensor), gymapi.ENV_SPACE
        )

    def _clear_external_force(self):
        if not hasattr(self, "external_force_tensor"):
            return
        self.external_force_tensor.zero_()
        self.external_position_tensor.zero_()
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.external_force_tensor), None, gymapi.ENV_SPACE)

    def _set_target_foot_friction(self, env_ids, friction):
        for env_id in env_ids.tolist():
            props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])
            for leg in torch.nonzero(self.event_target_leg_mask[env_id], as_tuple=False).flatten().tolist():
                foot = int(self.feet_indices[leg].item())
                if foot >= len(props):
                    raise RuntimeError(f"足端 shape 索引 {foot} 超出 {len(props)} 个 collision shapes")
                props[foot].friction = friction
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], props)

    def _restore_robot_friction(self, env_ids):
        for env_id in env_ids.tolist():
            props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])
            for prop in props:
                prop.friction = float(self.cfg.terrain.static_friction)
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], props)

    def _leg_action_ids(self, env_ids):
        rows, legs = torch.nonzero(self.event_target_leg_mask[env_ids], as_tuple=True)
        affected = env_ids[rows]
        joints = legs.unsqueeze(1) * 3 + torch.arange(3, device=self.device).unsqueeze(0)
        return affected, joints

    def _capture_leg_fault_targets(self, env_ids):
        affected, joints = self._leg_action_ids(env_ids)
        target = self.joint_pos_target[affected.unsqueeze(1), joints] - self.default_dof_pos[0, joints]
        target /= self.cfg.control.action_scale
        target[:, 0] /= self.cfg.control.hip_scale_reduction
        self.fault_target_action[affected.unsqueeze(1), joints] = target

    def _compute_torques(self, actions):
        active = self._scenario_ready and self.scenario_cfg.scenario == "leg_fault" and torch.any(self.event_started & ~self.event_finished)
        if not active:
            return super()._compute_torques(actions)
        ids = torch.nonzero(self.event_started & ~self.event_finished, as_tuple=False).flatten()
        affected, joints = self._leg_action_ids(ids)
        safe_actions = actions.clone()
        safe_actions[affected.unsqueeze(1), joints] = self.fault_target_action[affected.unsqueeze(1), joints]
        torques = super()._compute_torques(safe_actions)
        if self.scenario_cfg.leg_fault.mode == "zero_torque":
            torques[affected.unsqueeze(1), joints] = 0.0
        return torques

    # ------------------------------------------------------------------
    # 突发几何事件：坑洞永远在当前机身前方；圆环按当前足端展开范围生成。
    # ------------------------------------------------------------------
    def _trigger_geometry_event(self):
        c = self.scenario_cfg
        if c.geometry_event == "random":
            kind = "ground_drop" if torch.rand((), device=self.device) < c.ground_drop_probability else "obstacle_ring"
        else:
            kind = c.geometry_event
        if kind == "ground_drop":
            self._trigger_ground_drop()
        else:
            self._trigger_obstacle_ring()
        self.geometry_event_kind = kind
        self.event_started[:] = True
        self.event_trigger_count += 1
        self.event_step[:] = self.episode_length_buf
        print(f"[sim_env] 事件已触发：{kind}")

    def _trigger_ground_drop(self):
        g = self.scenario_cfg.ground_drop
        forward = g.hole_forward_offset_range[0] + torch.rand((), device=self.device) * (g.hole_forward_offset_range[1] - g.hole_forward_offset_range[0])
        lateral = g.hole_lateral_offset_range[0] + torch.rand((), device=self.device) * (g.hole_lateral_offset_range[1] - g.hole_lateral_offset_range[0])
        local = torch.stack((forward, lateral, torch.zeros((), device=self.device)))
        center = self.base_pos[0, :2] + quat_apply(self.base_quat[0].unsqueeze(0), local.unsqueeze(0))[0, :2]
        delta = self.tile_centers_tensor - center
        if g.hole_shape == "circle":
            selected = torch.sum(delta ** 2, dim=1) <= (min(g.hole_size_xy) / 2.0) ** 2
        else:
            selected = (delta[:, 0].abs() <= g.hole_size_xy[0] / 2.0) & (delta[:, 1].abs() <= g.hole_size_xy[1] / 2.0)
        ids = self.tile_actor_indices_tensor[selected]
        if not len(ids):
            ids = self.tile_actor_indices_tensor[torch.argmin(torch.sum(delta ** 2, dim=1)).view(1)]
        self.all_actor_root_states[ids, 2] = g.platform_height / 2.0 - g.drop_depth
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_actor_root_states), gymtorch.unwrap_tensor(ids), len(ids))

    def _trigger_obstacle_ring(self):
        r, g = self.scenario_cfg.obstacle_ring, self.scenario_cfg.ground_drop
        center = self.base_pos[0, :2]
        foot_radius = torch.norm(self.foot_positions[0, :, :2] - center, dim=1).max()
        inner = torch.clamp(foot_radius + r.foot_clearance, r.min_inner_radius, r.max_inner_radius)
        radius = float((inner + r.thickness / 2.0).item())
        for index, actor_id in enumerate(self.ring_actor_indices):
            angle = 2.0 * math.pi * index / r.segments
            self.all_actor_root_states[actor_id, 0] = center[0] + radius * math.cos(angle)
            self.all_actor_root_states[actor_id, 1] = center[1] + radius * math.sin(angle)
            self.all_actor_root_states[actor_id, 2] = g.platform_height + r.height / 2.0
            half = angle + math.pi / 2.0
            self.all_actor_root_states[actor_id, 3:7] = torch.tensor([0.0, 0.0, math.sin(half / 2.0), math.cos(half / 2.0)], device=self.device)
        ids = self.ring_actor_indices_tensor
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_actor_root_states), gymtorch.unwrap_tensor(ids), len(ids))

    # ------------------------------------------------------------------
    # 统一评价：各扰动场景都保留翻倒后的完整过程。
    # ------------------------------------------------------------------
    def check_termination(self):
        super().check_termination()
        self.reset_buf = self.time_out_buf.clone()

    def _update_recovery_metric(self):
        if not torch.any(self.event_started):
            return
        roll_pitch = torch.norm(self.projected_gravity[:, :2], dim=1)
        angular = torch.norm(self.base_ang_vel[:, :2], dim=1)
        vel_error = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        stable = (roll_pitch < .35) & (angular < 1.0) & (vel_error < .35)
        self.stable_steps = torch.where(self.event_started & stable, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        recovered = self.event_started & (self.recovery_step < 0) & (self.stable_steps >= 25)
        self.recovery_step[recovered] = self.episode_length_buf[recovered]

    def get_scenario_metrics(self):
        metrics = self.get_normal_metrics()
        metrics.update({
            "event_started": self.event_started,
            "event_finished": self.event_finished,
            "event_trigger_count": self.event_trigger_count,
            "event_step": self.event_step,
            "recovery_step": self.recovery_step,
            "target_leg_mask": self.event_target_leg_mask,
        })
        return metrics
